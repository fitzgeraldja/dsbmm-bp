import faulthandler
import time

import bp
import bp_sparse_parallel
import dsbmm
import dsbmm_sparse_parallel
import numpy as np
from numba.typed import List
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score as ari

# import bp_sparse
# import dsbmm_sparse

# from utils import nb_ari_local  # , nb_nmi_local

faulthandler.enable()

import matplotlib.pyplot as plt

# plt.ion()

import yaml  # for reading config file

with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

MAX_MSG_ITER = config["max_msg_iter"]
MSG_CONV_TOL = config["msg_conv_tol"]
MAX_ITER = config["max_iter"]
CONV_TOL = config["conv_tol"]


class EM:
    def __init__(
        self,
        data,
        init_Z_mode="A",
        msg_init_mode="planted",
        sparse_adj=False,
        n_runs=5,
        tuning_param=1.0,
        deg_corr=False,
        verbose=True,
    ):
        self.verbose = verbose
        self.msg_init_mode = msg_init_mode
        self.A = data["A"]
        if type(tuning_param) == float:
            self.tuning_params = [tuning_param]
        else:
            # assume passing list of values to try
            self.tuning_params = tuning_param
        self.deg_corr = deg_corr
        self.n_runs = n_runs
        self.run_idx = 0
        try:
            self.Q = data["Q"]
        except KeyError:
            raise ValueError("Must specify Q in data")
        self.sparse = sparse_adj
        try:
            if not self.sparse:
                assert np.allclose(self.A, self.A.transpose(1, 0, 2))
            else:
                # TODO: fix properly - currently just force symmetrising and binarising
                self.A = [((A_t + A_t.T) > 0) * 1.0 for A_t in self.A]

        except AssertionError:
            # symmetrise for this test case
            print(
                "WARNING: provided non-symmetric adjacency,",
                "\nsuggesting directed network:",
                "\nThis is currently unimplemented - symmetrising and binarising...",
            )
            self.A = ((self.A + self.A.transpose(1, 0, 2)) > 0) * 1.0
        # NB expecting A in shape N x N x T
        if not self.sparse:
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
            except KeyError:
                print(self.X)
                raise KeyError(
                    "X given as dict - expected test run with keys 'poisson' and 'indep bernoulli'"
                )

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
                if not self.sparse:
                    kmeans_mat = np.concatenate(
                        [self.A[:, :, t] for t in range(self.T)], axis=1
                    )  # done for fixing labels over time
                else:
                    kmeans_mat = sparse.hstack(self.A)
            except ValueError:
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
                    #   random_state=0, # TODO: consider allowing fixing this for reproducibility
                    max_iter=10,
                )
                .fit_predict(kmeans_mat)
                .reshape(-1, 1)
            )
        self.k_means_init_Z = np.tile(kmeans_labels, (1, self.T))
        if not self.sparse:
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
        self.k_means_init_Z[~self._pres_nodes] = -1
        try:
            assert self.k_means_init_Z.shape == (self.N, self.T)
        except AssertionError:
            print(self.k_means_init_Z.shape)
            print((self.N, self.T))
            raise ValueError("Wrong partition shape")
        self.perturb_init_Z()

        try:
            self.meta_types = List()
            for mt in data["meta_types"]:
                self.meta_types.append(mt)
            assert len(self.meta_types) == len(self.X)
        except AssertionError:
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

        except AttributeError:
            if self.verbose:
                print("No p_in / p_out provided")
        if self.sparse:
            # TODO: [clean-up] make abstract wrapper with separate dense / sparse impl
            # so can remove redundant code below
            self.dsbmm = dsbmm_sparse_parallel.DSBMMSparseParallel(
                A=self.A,
                X=self.X,
                Z=self.init_Z.copy(),
                Q=self.Q,
                deg_corr=self.deg_corr,
                meta_types=self.meta_types,
                tuning_param=self.tuning_params[0],
                verbose=self.verbose,
            )  # X=X,
            if self.verbose:
                print("Successfully instantiated DSBMM...")
            self.bp = bp_sparse_parallel.BPSparseParallel(self.dsbmm)
            if self.verbose:
                print("Successfully instantiated BP system...")
        else:
            self.dsbmm = dsbmm.DSBMM(
                A=self.A,
                X=self.X,
                Z=self.init_Z.copy(),
                Q=self.Q,
                deg_corr=self.deg_corr,
                meta_types=self.meta_types,
                tuning_param=self.tuning_params[0],
                verbose=self.verbose,
            )  # X=X,
            if self.verbose:
                print("Successfully instantiated DSBMM...")
            self.bp = bp.BP(self.dsbmm)
            if self.verbose:
                print("Successfully instantiated BP system...")
        ## Initialise model params
        if self.verbose:
            print("Now initialising model:")
        self.bp.model.update_params(init=True)
        if self.verbose:
            print("\tInitialised all DSBMM params!")
        self.bp.init_messages(mode=msg_init_mode)
        if self.verbose:
            print(f"\tInitialised messages and marginals ({msg_init_mode})")
        self.bp.init_h()
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
            self.ax.set_xlim(0, MAX_ITER)
            # Other stuff
            self.ax.grid()
            # ...
        self.best_Z = None
        self.best_val = 0.0

    def perturb_init_Z(self, pert_prop=0.05):
        # add some noise to init clustering
        # prop = 0.1  # proportion of noise to add
        mask = np.random.rand(*self.k_means_init_Z.shape) < pert_prop
        init_Z = self.k_means_init_Z.copy()
        init_Z[mask] += np.random.randint(
            -self.Q // 2,
            self.Q // 2 + 1,
            size=(self.N, self.T),
        )[mask]
        init_Z[init_Z < 0] = 0
        init_Z[init_Z > self.Q - 1] = self.Q - 1
        init_Z[~self._pres_nodes] = -1
        self.init_Z = init_Z

    def reinit(self, tuning_param=1.0, set_Z=None):
        self.perturb_init_Z()
        if self.sparse:
            self.dsbmm = dsbmm_sparse_parallel.DSBMMSparseParallel(
                A=self.A,
                X=self.X,
                Z=self.init_Z.copy() if set_Z is None else set_Z,
                Q=self.Q,
                deg_corr=self.deg_corr,
                meta_types=self.meta_types,
                tuning_param=tuning_param,
                verbose=self.verbose,
            )  # X=X,
            if self.verbose:
                print("Successfully reinstantiated DSBMM...")
            self.bp = bp_sparse_parallel.BPSparseParallel(self.dsbmm)
            if self.verbose:
                print("Successfully reinstantiated BP system...")
        else:
            self.dsbmm = dsbmm.DSBMM(
                A=self.A,
                X=self.X,
                Z=self.init_Z.copy() if set_Z is None else set_Z,
                Q=self.Q,
                deg_corr=self.deg_corr,
                meta_types=self.meta_types,
                tuning_param=tuning_param,
                verbose=self.verbose,
            )  # X=X,
            if self.verbose:
                print("Successfully reinstantiated DSBMM...")
            self.bp = bp.BP(self.dsbmm)
            if self.verbose:
                print("Successfully reinstantiated BP system...")
        ## Initialise model params
        if self.verbose:
            print("Now reinitialising model:")
        self.bp.model.update_params(init=True)
        if self.verbose:
            print("\tReinitialised all DSBMM params!")
        self.bp.init_messages(mode=self.msg_init_mode)
        if self.verbose:
            print(f"\tReinitialised messages and marginals ({self.msg_init_mode})")
        self.bp.init_h()
        if self.verbose:
            print("\tReinitialised corresponding external fields")
        if self.verbose:
            print("Done, can now run updates again")
        if self.verbose:
            # make updating plot of average score (e.g. ARI) if ground truth known
            self.xdata, self.ydata = [], []
            self.figure, self.ax = plt.subplots(dpi=200)
            (self.lines,) = self.ax.plot([], [], "o")
            # Autoscale on unknown axis and known lims on the other
            self.ax.set_autoscaley_on(True)
            self.ax.set_xlim(0, MAX_ITER)
            # Other stuff
            self.ax.grid()
            # ...
        self.run_idx += 1

    def fit(
        self,
        max_iter=MAX_ITER,
        max_msg_iter=MAX_MSG_ITER,
        conv_tol=CONV_TOL,
        msg_conv_tol=MSG_CONV_TOL,
        true_Z=None,
        learning_rate=0.2,
    ):
        self.true_Z = true_Z
        if self.verbose:
            print(
                "#" * 15,
                f"Using tuning_param {self.tuning_params[0]}",
                "#" * 15,
            )
        while self.run_idx < self.n_runs - 1:
            if self.verbose:
                print("%" * 15, f"Starting run {self.run_idx+1}", "%" * 15)
            self.do_run(max_iter, max_msg_iter, conv_tol, msg_conv_tol, learning_rate)
            self.bp.model.set_Z_by_MAP()
            self.reinit()
        if self.verbose:
            print("%" * 15, f"Starting run {self.run_idx+1}", "%" * 15)
        # final random init run
        self.do_run(max_iter, max_msg_iter, conv_tol, msg_conv_tol, learning_rate)
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
                        max_iter,
                        max_msg_iter,
                        conv_tol,
                        msg_conv_tol,
                        learning_rate,
                    )
                    self.bp.model.set_Z_by_MAP()
                    self.reinit(tuning_param=tuning_param)
                # final random init run
                self.do_run(
                    max_iter,
                    max_msg_iter,
                    conv_tol,
                    msg_conv_tol,
                    learning_rate,
                )
        # now reinit with best part found from these runs
        if self.best_Z is None:
            self.best_Z = self.bp.model.Z
        try:
            self.reinit(tuning_param=self.best_tun_param, set_Z=self.best_Z)
        except AttributeError:
            # no tuning param used
            self.reinit(set_Z=self.best_Z)
        self.do_run(max_iter, max_msg_iter, conv_tol, msg_conv_tol, learning_rate)

    def do_run(self, max_iter, max_msg_iter, conv_tol, msg_conv_tol, learning_rate):
        for n_iter in range(max_iter):
            if self.verbose:
                print(f"\n##### At iteration {n_iter+1} #####")
                self.xdata.append(n_iter)
            for msg_iter in range(max_msg_iter):
                if self.verbose:
                    print(f"Message update iter {msg_iter + 1}...")
                self.bp.update_node_marg()  # dumping rate?
                if self.verbose:
                    print("\tUpdated node marginals, messages and external fields")
                msg_diff = self.bp.jit_model.msg_diff
                if self.verbose:
                    print(f"\tmsg differences: {msg_diff:.4f}")
                if msg_diff < msg_conv_tol:
                    break
                self.bp.zero_diff()
            self.bp.update_twopoint_marginals()  # dumping rate?
            if self.verbose:
                print("Initialised corresponding twopoint marginals")
            self.bp.model.set_node_marg(self.bp.jit_model.node_marg)
            self.bp.model.set_twopoint_edge_marg(self.bp.jit_model.twopoint_e_marg)
            self.bp.model.set_twopoint_time_marg(self.bp.jit_model.twopoint_t_marg)
            if self.verbose:
                print("\tPassed marginals to DSBMM")
            self.bp.model.update_params(init=False, learning_rate=learning_rate)
            if self.verbose:
                print("\tUpdated DSBMM params given marginals")
            diff = self.bp.model.jit_model.diff
            if self.verbose:
                print(f"Successfully completed update! Diff = {diff:.4f}")
            if diff < conv_tol:
                if self.verbose:
                    print("~~~~~~ CONVERGED ~~~~~~")
                break
            self.bp.model.zero_diff()
            if self.true_Z is None:
                current_energy = self.bp.compute_free_energy()
                if self.best_val == 0.0:
                    # first iter, first run
                    self.best_val = current_energy
                    self.bp.model.set_Z_by_MAP()
                    self.best_Z = self.bp.model.Z
                    self.best_tun_param = self.dsbmm.tuning_param
                elif current_energy < self.best_val:
                    # new best
                    self.best_val = current_energy
                    self.bp.model.set_Z_by_MAP()
                    self.best_Z = self.bp.model.Z
                    self.best_tun_param = self.dsbmm.tuning_param
            else:
                self.bp.model.set_Z_by_MAP()
                current_score = self.ari_score(self.true_Z)
                if current_score.mean() >= self.best_val:
                    self.best_val = current_score.mean()
                    self.best_Z = self.bp.model.Z
            if self.verbose:
                self.bp.model.set_Z_by_MAP()
                if self.true_Z is not None:
                    current_score = self.ari_score(self.true_Z)
                    self.ydata.append(current_score.mean())
                    self.update_score_plot()
                    print()

    def ari_score(self, true_Z, pred_Z=None):
        # wait for a second to execute in case of parallelisation issues
        time.sleep(0.5)
        if pred_Z is None:
            try:
                # could try and replace with numba again but seemed
                # to cause issues
                aris = np.array(
                    [
                        ari(true[pred > -1], pred[pred > -1])
                        for true, pred in zip(self.bp.model.jit_model.Z.T, true_Z.T)
                    ]
                )
                if self.verbose:
                    print("ARIs:", np.round_(aris, 2))
                return aris
            except AttributeError:
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
