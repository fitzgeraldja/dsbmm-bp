import faulthandler
import time

import numpy as np
from numba.typed import List
from scipy import sparse
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score as ari

# from . import (
#     bp,
#     bp_sparse,
#     bp_sparse_parallel,
#     dsbmm,
#     dsbmm_sparse,
#     dsbmm_sparse_parallel,
# )
from .np_par_bp_methods import NumpyBP
from .np_par_dsbmm_methods import NumpyDSBMM

# from utils import nb_ari_local  # , nb_nmi_local

faulthandler.enable()

import matplotlib.pyplot as plt
import yaml  # type: ignore # for reading config file
from tqdm import tqdm

# plt.ion()

try:
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    MAX_MSG_ITER = config["max_msg_iter"]
    MSG_CONV_TOL = config["msg_conv_tol"]
    MAX_ITER = config["max_iter"]
    CONV_TOL = config["conv_tol"]
except FileNotFoundError:
    MAX_ITER = 100  # max num EM iterations
    MAX_MSG_ITER = 30  # max num BP iterations each EM cycle
    MSG_CONV_TOL = 1.0e-5  # conv tol for differences between messages after BP update
    CONV_TOL = 1.0e-5  # total difference tol for params after EM run


class EM:
    def __init__(
        self,
        data,
        init_Z_mode="A",
        msg_init_mode="planted",
        sparse_adj=False,
        use_numba=False,
        try_parallel=False,
        n_runs=5,
        patience=None,
        tuning_param=1.0,
        deg_corr=False,
        leave_directed=False,
        verbose=True,
        use_meta=True,
        max_iter=30,
        max_msg_iter=10,
        trial_Qs=None,
        alpha_use_all=True,
        non_informative_init=True,
        planted_p=0.6,
        auto_tune=False,
        ret_probs=False,
        fe_freq=3,  # calc free energy every fe_freq EM iterations
    ):
        self.verbose = verbose
        self.parallel = try_parallel
        self.patience = patience if patience is not None else max_iter
        self.use_meta = use_meta
        self.use_numba = use_numba
        self.msg_init_mode = msg_init_mode
        self.max_iter = max_iter
        self.max_msg_iter = max_msg_iter
        self.init_Z_mode = init_Z_mode
        self.alpha_use_all = alpha_use_all
        self.auto_tune = auto_tune
        self.ret_probs = ret_probs
        self.A = data["A"]
        self.frozen = False
        self.params_to_set = None
        self.non_informative_init = non_informative_init
        self.planted_p = planted_p
        self.fe_freq = fe_freq
        if type(tuning_param) == float:
            self.tuning_params = [tuning_param]
        else:
            # assume passing list of values to try
            self.tuning_params = tuning_param
        self.deg_corr = deg_corr
        self.directed = leave_directed
        self.n_runs = n_runs
        self.run_idx = 0
        try:
            self.Q = data["Q"]
        except Exception:  # KeyError:
            raise ValueError("Must specify Q in data")
        self.trial_Qs = trial_Qs if trial_Qs is not None else [self.Q]
        self.sparse = sparse_adj
        try:
            if not self.sparse and self.use_numba:
                # NB numpy version takes sparse by default
                assert np.allclose(self.A, self.A.transpose(1, 0, 2))
            else:
                if not self.directed:
                    try:
                        if not self.deg_corr:
                            # binarise and symmetrise
                            self.A = [((A_t + A_t.T) > 0) * 1.0 for A_t in self.A]
                        else:
                            # symmetrise by taking average w transpose
                            self.A = [(A_t + A_t.T) / 2.0 for A_t in self.A]
                    except ValueError:
                        print(*[A_t.shape for A_t in self.A], sep="\n")
                        raise ValueError("Problem w non-square adj matrix input")
                else:
                    if not self.deg_corr:
                        # binarise
                        self.A = [(A_t > 0) * 1.0 for A_t in self.A]

        except Exception:  # AssertionError:
            # symmetrise for this test case
            print(
                "WARNING: provided non-symmetric adjacency,",
                "\nsuggesting directed network:",
                "\nThis is currently unimplemented - symmetrising and binarising...",
            )
            self.A = ((self.A + self.A.transpose(1, 0, 2)) > 0) * 1.0
        # NB expecting A in shape N x N x T
        if not self.sparse and self.use_numba:
            self.N = self.A.shape[0]
            self.T = self.A.shape[2]
        else:
            self.N = self.A[0].shape[0]
            self.T = len(self.A)
        self.X = data["X"]
        if type(self.X) == dict:
            try:
                X_poisson = np.ascontiguousarray(self.X["poisson"].transpose(1, 2, 0))
                X_ib = np.ascontiguousarray(
                    self.X["indep bernoulli"].transpose(1, 2, 0)
                )
                self.X = List()
                self.X.append(X_poisson)
                self.X.append(X_ib)
            except Exception:  # KeyError:
                print(self.X)
                raise KeyError(
                    "X given as dict - expected test run with keys 'poisson' and 'indep bernoulli', or 'categorical'"
                )

        if not self.sparse and self.use_numba:
            self._pres_nodes = (self.A.sum(axis=1) > 0) | (self.A.sum(axis=0) > 0)
        else:
            self._pres_nodes = np.zeros(
                (self.N, self.T), dtype=bool
            )  # N x T boolean array w i,t True if i present in net at time t
            for t in range(self.T):
                indptrs = self.A[t].indptr
                idxs = self.A[t].indices
                indptrsT = self.A[t].transpose().indptr
                idxsT = self.A[t].transpose().indices
                for i in range(self.N):
                    val_locs = idxs[indptrs[i] : indptrs[i + 1]]
                    self._pres_nodes[val_locs, t] = True
                    val_locsT = idxsT[indptrsT[i] : indptrsT[i + 1]]
                    self._pres_nodes[val_locsT, t] = True

        self.initialise_partition(self.Q, init_Z_mode=self.init_Z_mode)
        self.perturb_init_Z()

        try:
            self.meta_types = List()
            for mt in data["meta_types"]:
                self.meta_types.append(mt)
            assert len(self.meta_types) == len(self.X)
        except Exception:  # AssertionError:
            raise ValueError("Must specify types of all metadata")
        try:
            # known params for sim data
            self.p_in = data["p_in"]
            self.p_out = data["p_out"]
            self.p_stay = data["p_stay"]
            if self.verbose:
                # NB expected degree is ~ (N/Q)(p_in + (Q-1)*p_out)
                print(
                    f"With params p_in={self.p_in},",
                    f"p_out={self.p_out},",
                    f"Q={self.Q},",
                    f"N={self.N},",
                )
                exp_deg = (self.N / self.Q) * (self.p_in + (self.Q - 1) * self.p_out)
                print(f"\tExpected degree in data: {exp_deg:.2f}")
                # as is distributed as Poisson Binomial, w roughly N/Q indep samps
                # w p = p_in, and (Q - 1)N/Q indep samps w p = p_out
                # So variance is (N/Q)(p_in*(1-p_in) + (Q-1)*p_out*(1-p_out))
                deg_var = (self.N / self.Q) * (
                    self.p_in * (1 - self.p_in)
                    + (self.Q - 1) * self.p_out * (1 - self.p_out)
                )
                print(f"\twith variance {deg_var:.2f}")

        except Exception:  # AttributeError:
            if self.verbose:
                print("No p_in / p_out provided")
        self.reset_model(self.tuning_params[0], reinit=False)

    def initialise_partition(self, Q, init_Z_mode="A"):
        # set self Q to specified Q as well as initialising Z
        self.Q = Q
        ## Sort init partition
        if init_Z_mode in ["AX", "XA"]:
            kmeans_mat = np.concatenate(
                [
                    *[self.A[:, :, t] for t in range(self.T)],
                    *[
                        Xs.transpose(1, 2, 0).reshape(self.N, -1)
                        for Xs in self.X.values()
                    ],
                ],
                axis=1,
            )  # done for fixing labels over time
        else:
            try:
                if not self.sparse and self.use_numba:
                    kmeans_mat = np.concatenate(
                        [self.A[:, :, t] for t in range(self.T)], axis=1
                    )  # done for fixing labels over time
                else:
                    kmeans_mat = sparse.hstack(self.A)
            except Exception:  # ValueError:
                print("A:", self.A.shape, "T:", self.T)
                raise ValueError("Problem with A passed")
        if self.N > 1e5:
            kmeans_labels = (
                MiniBatchKMeans(
                    n_clusters=self.Q,
                    #   random_state=0, # TODO: consider allowing fixing this for reproducibility
                    batch_size=100,
                    max_iter=10,
                )
                .fit_predict(kmeans_mat)
                .reshape(-1, 1)
            )
        else:
            kmeans_labels = (
                KMeans(
                    n_clusters=self.Q,
                    #   random_state=0, # likewise consider allowing fixing this for reproducibility
                    max_iter=10,
                )
                .fit_predict(kmeans_mat)
                .reshape(-1, 1)
            )
        self.k_means_init_Z = np.tile(kmeans_labels, (1, self.T))
        self.k_means_init_Z[~self._pres_nodes] = -1
        try:
            assert self.k_means_init_Z.shape == (self.N, self.T)
        except Exception:  # AssertionError:
            print(self.k_means_init_Z.shape)
            print((self.N, self.T))
            raise ValueError("Wrong partition shape")

    def perturb_init_Z(self, pert_prop=0.05):
        # add some noise to init clustering
        # w specified proportion of noise to add
        mask = np.random.rand(*self.k_means_init_Z.shape) < pert_prop
        init_Z = self.k_means_init_Z.copy()
        init_Z[mask] += np.random.randint(
            0,
            self.Q,
            size=(self.N, self.T),
        )[mask]
        init_Z[mask] = np.mod(init_Z, self.Q)[mask]
        init_Z[~self._pres_nodes] = -1
        self.init_Z = init_Z

    def reinit(self, tuning_param=1.0, set_Z=None):
        self.perturb_init_Z()
        self.reset_model(tuning_param, set_Z=set_Z)

    def set_params(self, params, freeze=True):
        self.frozen = freeze
        self.params_to_set = params
        self.dsbmm.set_params(params, freeze=True)
        self.dsbmm.update_params(init=True)  # necessary to update the meta lkl suitably

    def reset_model(self, tuning_param, set_Z=None, reinit=True):
        retext = "re" if reinit else ""
        if self.use_numba:
            raise ValueError("Currently pausing numba support")
            # if self.sparse:
            #     if self.parallel:
            #         self.dsbmm = dsbmm_sparse_parallel.DSBMMSparseParallel(
            #             A=self.A,
            #             X=self.X,
            #             Z=self.init_Z.copy() if set_Z is None else set_Z,
            #             Q=self.Q,
            #             deg_corr=self.deg_corr,
            #             meta_types=self.meta_types,
            #             tuning_param=tuning_param,
            #             verbose=self.verbose,
            #             use_meta=self.use_meta,
            #             auto_tune=self.auto_tune,
            #         )  # X=X,
            #         if self.frozen or self.params_to_set is not None:
            #             try:
            #                 assert self.params_to_set is not None
            #             except AssertionError:
            #                 raise ValueError(
            #                     "If freezing EM then must pass suitable DSBMM parameters."
            #                 )
            #             self.set_params(self.params_to_set)
            #         if self.verbose:
            #             print(f"Successfully {retext}instantiated DSBMM...")
            #         self.bp = bp_sparse_parallel.BPSparseParallel(self.dsbmm)
            #         if self.verbose:
            #             print(f"Successfully {retext}instantiated BP system...")
            #     else:
            #         self.dsbmm = dsbmm_sparse.DSBMMSparse(
            #             A=self.A,
            #             X=self.X,
            #             Z=self.init_Z.copy() if set_Z is None else set_Z,
            #             Q=self.Q,
            #             deg_corr=self.deg_corr,
            #             meta_types=self.meta_types,
            #             tuning_param=tuning_param,
            #             verbose=self.verbose,
            #             use_meta=self.use_meta,
            #             auto_tune=self.auto_tune,
            #         )  # X=X,
            #         if self.frozen or self.params_to_set is not None:
            #             try:
            #                 assert self.params_to_set is not None
            #             except AssertionError:
            #                 raise ValueError(
            #                     "If freezing EM then must pass suitable DSBMM parameters."
            #                 )
            #             self.set_params(self.params_to_set)
            #         if self.verbose:
            #             print(f"Successfully {retext}instantiated DSBMM...")
            #         self.bp = bp_sparse.BPSparse(self.dsbmm)
            #         if self.verbose:
            #             print(f"Successfully {retext}instantiated BP system...")
            # else:
            #     self.dsbmm = dsbmm.DSBMM(
            #         A=self.A,
            #         X=self.X,
            #         Z=self.init_Z.copy() if set_Z is None else set_Z,
            #         Q=self.Q,
            #         deg_corr=self.deg_corr,
            #         meta_types=self.meta_types,
            #         tuning_param=tuning_param,
            #         verbose=self.verbose,
            #         use_meta=self.use_meta,
            #         auto_tune=self.auto_tune,
            #     )  # X=X,
            #     if self.frozen or self.params_to_set is not None:
            #         try:
            #             assert self.params_to_set is not None
            #         except AssertionError:
            #             raise ValueError(
            #                 "If freezing EM then must pass suitable DSBMM parameters."
            #             )
            #         self.set_params(self.params_to_set)
            #     if self.verbose:
            #         print(f"Successfully {retext}instantiated DSBMM...")
            #     self.bp = bp.BP(self.dsbmm)
            #     if self.verbose:
            #         print(f"Successfully {retext}instantiated BP system...")
        else:
            if self.verbose:
                print(f"Successfully {retext}instantiated DSBMM...")
            self.dsbmm = NumpyDSBMM(
                A=self.A,
                X=self.X,
                Z=self.init_Z.copy() if set_Z is None else set_Z,
                Q=self.Q,
                deg_corr=self.deg_corr,
                directed=self.directed,
                meta_types=self.meta_types,
                use_meta=self.use_meta,
                tuning_param=tuning_param,
                verbose=self.verbose,
                alpha_use_all=self.alpha_use_all,
                non_informative_init=self.non_informative_init,
                auto_tune=self.auto_tune,
            )
            if self.frozen or self.params_to_set is not None:
                try:
                    assert self.params_to_set is not None
                except AssertionError:
                    raise ValueError(
                        "If freezing EM then must pass suitable DSBMM parameters."
                    )
                self.set_params(self.params_to_set)
            if self.verbose:
                print(f"Successfully {retext}instantiated BP system...")
            self.bp = NumpyBP(self.dsbmm)

        ## Initialise model params
        if self.verbose:
            print(f"Now {retext}initialising model:")
        self.bp.model.update_params(init=True, planted_p=self.planted_p)
        if self.verbose:
            print("\tInitialised all DSBMM params!")
        self.bp.init_messages(mode=self.msg_init_mode)
        if self.verbose:
            print(f"\tInitialised messages and marginals ({self.msg_init_mode})")
        if self.use_numba:
            self.bp.init_h()
        else:
            self.bp.calc_h()
        if self.verbose:
            print("\tInitialised corresponding external fields")
        if self.verbose:
            print("Done, can now run updates")

        if self.verbose:
            # make updating plot of average score (e.g. ARI) if ground truth known
            self.xdata, self.ydata = [], []
            self.figure, self.ax = plt.subplots(dpi=200)
            (self.lines,) = self.ax.plot([], [], "o")
            # Autoscale on unknown axis and known lims on the other
            self.ax.set_autoscaley_on(True)
            self.ax.set_xlim(0, self.max_iter)
            # Other stuff
            self.ax.grid()
            # ...
        if not reinit:
            self.best_Z = None
            self.best_val = 0.0
        else:
            self.run_idx += 1

    def fit(
        self,
        conv_tol=1e-5,
        msg_conv_tol=1e-5,
        true_Z=None,
        learning_rate=0.2,
    ):
        self.true_Z = true_Z

        self.all_best_Zs = np.empty((len(self.trial_Qs), self.N, self.T))
        if np.all(self.trial_Qs == self.trial_Qs[0]):
            self.all_pi = np.empty(
                (len(self.trial_Qs), self.trial_Qs[0], self.trial_Qs[0])
            )
        self.best_tun_pars = np.ones_like(self.trial_Qs, dtype=float)
        if self.ret_probs:
            try:
                assert np.all(self.trial_Qs == self.trial_Qs[0])
            except AssertionError:
                raise ValueError("If returning probs, must have same Q for all trials")
            self.run_probs = np.empty(
                (len(self.trial_Qs), self.N, self.T, self.trial_Qs[0])
            )
        for q_idx, current_Q in enumerate(self.trial_Qs):
            self.q_idx = q_idx
            self.best_val_q = 0.0
            tqdm.write(f"\tCurrent Q: {current_Q}")
            tqdm.write("")
            if current_Q != self.Q:
                self.initialise_partition(current_Q, init_Z_mode=self.init_Z_mode)
                self.reinit(tuning_param=self.tuning_params[0])
            if self.verbose:
                print(
                    "#" * 15,
                    f"Using tuning_param {self.tuning_params[0]}",
                    "#" * 15,
                )
            with tqdm(total=self.n_runs, desc="Run no.", leave=False) as run_pbar:
                while self.run_idx < self.n_runs - 1:
                    if self.verbose:
                        print("%" * 15, f"Starting run {self.run_idx+1}", "%" * 15)
                    self.do_run(conv_tol, msg_conv_tol, learning_rate)
                    self.bp.model.set_Z_by_MAP()
                    run_pbar.update(1)
                    self.reinit(tuning_param=self.tuning_params[0])
                if self.verbose:
                    print("%" * 15, f"Starting run {self.run_idx+1}", "%" * 15)
                # final random init run
                self.do_run(conv_tol, msg_conv_tol, learning_rate)
                self.bp.model.set_Z_by_MAP()
                run_pbar.update(1)
            if len(self.tuning_params) > 1:
                for tuning_param in self.tuning_params[1:]:
                    if self.verbose:
                        print(
                            "#" * 15,
                            f"Using tuning_param {tuning_param}",
                            "#" * 15,
                        )
                        time.sleep(1)
                    self.run_idx = 0

                    self.reinit(tuning_param=tuning_param)
                    while self.run_idx < self.n_runs - 1:
                        self.do_run(
                            conv_tol,
                            msg_conv_tol,
                            learning_rate,
                        )
                        self.bp.model.set_Z_by_MAP()
                        self.reinit(tuning_param=tuning_param)
                    # final random init run
                    self.do_run(
                        conv_tol,
                        msg_conv_tol,
                        learning_rate,
                    )
                    self.bp.model.set_Z_by_MAP()
            # now reinit with best part found from these runs
            if self.best_Z is None:
                self.best_Z = self.bp.model.Z.copy()
                self.pi = self.dsbmm._pi.copy()
            try:
                self.reinit(tuning_param=self.best_tun_param, set_Z=self.best_Z)
            except Exception:  # AttributeError:
                # no tuning param used
                self.reinit(set_Z=self.best_Z)
            self.do_run(conv_tol, msg_conv_tol, learning_rate)
            self.bp.model.set_Z_by_MAP()

    def do_run(self, conv_tol, msg_conv_tol, learning_rate):
        self.poor_iter_ctr = 0  # ctr for iters without reduction in free energy
        diff = 1.0
        for n_iter in tqdm(
            range(self.max_iter), desc=f"EM iter, diff {diff:.3g}", leave=False
        ):
            if self.verbose:
                tqdm.write(f"\n##### At iteration {n_iter+1} #####")
                self.xdata.append(n_iter)
            for msg_iter in tqdm(
                range(self.max_msg_iter), desc="Message iter", leave=False
            ):
                if n_iter < 5 and self.max_iter > 1:
                    if msg_iter > self.max_msg_iter // 3:
                        # Use fewer max msg updates for first few EM steps, where params usually inaccurate
                        break
                if self.verbose:
                    tqdm.write(f"Message update iter {msg_iter + 1}...")
                self.bp.update_node_marg()  # dumping rate?
                if self.verbose:
                    tqdm.write("\tUpdated node marginals, messages and external fields")
                if self.use_numba:
                    msg_diff = self.bp.jit_model.msg_diff
                else:
                    msg_diff = self.bp.msg_diff
                if self.verbose:
                    tqdm.write(f"\tmsg differences: {msg_diff:.4g}")
                if msg_diff < msg_conv_tol:
                    break
                self.bp.zero_diff()
            if self.use_numba:
                self.bp.model.set_node_marg(self.bp.jit_model.node_marg)
            else:
                self.bp.model.set_node_marg(self.bp.node_marg)
            if self.max_iter > 1:
                self.bp.update_twopoint_marginals()  # dumping rate?
                if self.verbose:
                    tqdm.write("Initialised corresponding twopoint marginals")
                if self.use_numba:
                    self.bp.model.set_twopoint_edge_marg(
                        self.bp.jit_model.twopoint_e_marg
                    )
                    self.bp.model.set_twopoint_time_marg(
                        self.bp.jit_model.twopoint_t_marg
                    )
                else:
                    self.bp.model.set_twopoint_edge_marg(self.bp.twopoint_e_marg)
                    self.bp.model.set_twopoint_time_marg(self.bp.twopoint_t_marg)
                if self.verbose:
                    tqdm.write("\tPassed marginals to DSBMM")
                self.bp.model.update_params(init=False, learning_rate=learning_rate)
                if self.verbose:
                    tqdm.write("\tUpdated DSBMM params given marginals")
                if self.use_numba:
                    diff = self.bp.model.jit_model.diff
                else:
                    diff = self.bp.model.diff
                if self.verbose:
                    tqdm.write(f"Successfully completed update! Diff = {diff:.4f}")
                self.bp.model.zero_diff()
            else:
                diff = 1.0
            if self.true_Z is None:
                if n_iter % self.fe_freq == 0:
                    current_energy = self.bp.compute_free_energy()
                    if self.best_val_q == 0.0:
                        self.max_energy = current_energy
                        # first iter, first run
                        self.best_val_q = current_energy
                        self.best_val = current_energy
                        self.poor_iter_ctr = 0
                        self.bp.model.set_Z_by_MAP()
                        self.best_Z = self.bp.model.Z.copy()
                        self.pi = self.dsbmm._pi.copy()
                        tmp_tun_param = self.dsbmm.tuning_param
                        if tmp_tun_param > 1.0e4:
                            tmp_tun_param = 1.0e4
                        elif tmp_tun_param < 1.0e-4:
                            tmp_tun_param = 1.0e-4
                        self.best_tun_param = tmp_tun_param
                        self.best_tun_pars[self.q_idx] = tmp_tun_param
                        self.max_energy_Z = self.bp.model.Z.copy()
                    elif current_energy < self.best_val_q:
                        # new best for q
                        self.poor_iter_ctr = 0
                        self.best_val_q = current_energy
                        self.bp.model.set_Z_by_MAP()
                        self.all_best_Zs[self.q_idx, :, :] = self.bp.model.Z.copy()
                        self.all_pi[self.q_idx, ...] = self.dsbmm._pi.copy()
                        tmp_tun_param = self.dsbmm.tuning_param
                        if tmp_tun_param > 1.0e4:
                            tmp_tun_param = 1.0e4
                        elif tmp_tun_param < 1.0e-4:
                            tmp_tun_param = 1.0e-4
                        self.best_tun_param = tmp_tun_param
                        self.best_tun_pars[self.q_idx] = tmp_tun_param
                        if self.ret_probs:
                            self.run_probs[
                                self.q_idx, ...
                            ] = self.bp.model.node_marg.copy()
                        if self.best_val_q < self.best_val:
                            self.best_val = self.best_val_q
                            self.best_Z = self.bp.model.Z.copy()
                            self.pi = self.dsbmm._pi.copy()
                            self.best_tun_param = self.dsbmm.tuning_param

                    else:
                        self.poor_iter_ctr += 1
                        if self.poor_iter_ctr >= self.patience:
                            tqdm.write(
                                f"~~~~~~ OUT OF PATIENCE, STOPPING EARLY in run {self.run_idx+1} ~~~~~~"
                            )
                            self.poor_iter_ctr = 0
                            break
                    if current_energy > self.max_energy:
                        self.max_energy = current_energy
                        self.max_energy_Z = self.bp.model.Z
            else:
                self.bp.model.set_Z_by_MAP()
                current_score = self.ari_score(self.true_Z)
                if current_score.mean() >= self.best_val:
                    self.best_val = current_score.mean()
                    self.best_Z = self.bp.model.Z.copy()
                    self.pi = self.dsbmm._pi.copy()
            if self.verbose:
                self.bp.model.set_Z_by_MAP()
                if self.true_Z is not None:
                    current_score = self.ari_score(self.true_Z)
                    self.ydata.append(current_score.mean())
                    self.update_score_plot()
                    print()
            if diff < conv_tol or (msg_diff < msg_conv_tol and self.max_iter == 1):
                tqdm.write(f"~~~~~~ CONVERGED in run {self.run_idx+1} ~~~~~~")
                break
            else:
                if n_iter == self.max_iter - 1:
                    tqdm.write(
                        f"~~~~~~ MAX ITERATIONS REACHED in run {self.run_idx+1} ~~~~~~"
                    )

    def ari_score(self, true_Z, pred_Z=None):
        if self.use_numba:
            # wait for a second to execute in case of parallelisation issues
            time.sleep(0.5)
        if pred_Z is None:
            try:
                # could try and replace with numba again but seemed
                # to cause issues
                pred_Z = self.best_Z
                aris = np.array(
                    [
                        ari(true[pred > -1], pred[pred > -1])
                        for true, pred in zip(pred_Z.T, true_Z.T)
                    ]
                )
                if self.verbose:
                    print("ARIs:", np.round_(aris, 2))
                return aris
            except Exception:  # AttributeError:
                print(
                    "Can't provide ARI score - make sure have both",
                    "provided ground truth \nwhen initialising model,",
                    "and have called fit",
                )
        else:
            aris = np.array(
                [
                    ari(true[pred > -1], pred[pred > -1])
                    for true, pred in zip(pred_Z.T, true_Z.T)
                ]
            )
            if self.verbose:
                print("ARIs:", np.round_(aris, 2))
            return aris

    def update_score_plot(self):
        # update score plot with new data
        self.lines.set_xdata(self.xdata)
        self.lines.set_ydata(self.ydata)
        # Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        # We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    # print("NMIs: ", nb_nmi_local(bp_ex.model.Z, Z))

    # print("Overlaps:", (bp_ex.model.Z == Z).sum(axis=0) / Z.shape[0])
