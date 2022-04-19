import simulation
import data_processer
import dsbmm
import bp
import dsbmm_sparse
import bp_sparse

# from utils import nb_ari_local  # , nb_nmi_local
from sklearn.metrics import adjusted_rand_score as ari
import numpy as np
from scipy import sparse
from numba.typed import List
from sklearn.cluster import MiniBatchKMeans, KMeans
import time

import faulthandler

faulthandler.enable()

import matplotlib.pyplot as plt

plt.ion()

import pickle

MAX_MSG_ITER = 5
MSG_CONV_TOL = 1e-4
MAX_ITER = 10
CONV_TOL = 1e-4


class EM:
    def __init__(self, data, msg_init_mode="planted", sparse=False, verbose=True):
        self.verbose = verbose
        self.A = data["A"]
        self.sparse = sparse
        try:
            if not sparse:
                assert np.allclose(self.A, self.A.transpose(1, 0, 2))
            else:
                # TODO: fix properly - currently just force symmetrising and binarising
                self.A = [((A_t + A_t.T) > 0) * 1.0 for A_t in self.A]

        except:
            # symmetrise for this test case
            print(
                "WARNING: provided non-symmetric adjacency,",
                "\nsuggesting directed network:",
                "\nThis is currently unimplemented - symmetrising and binarising...",
            )
            self.A = ((self.A + self.A.transpose(1, 0, 2)) > 0) * 1.0
        # NB expecting A in shape N x N x T
        if not sparse:
            self.N = self.A.shape[0]
            self.T = self.A.shape[2]
        else:
            self.N = self.A[0].shape[0]
            self.T = len(self.A)
        self.X = data["X"]
        self.X_poisson = np.ascontiguousarray(self.X["poisson"].transpose(1, 2, 0))
        self.X_ib = np.ascontiguousarray(self.X["indep bernoulli"].transpose(1, 2, 0))
        # self.X_poisson = data["X_poisson"]
        # self.X_ib = data["X_ib"]
        try:
            self.init_Z = data["Z"]
            # print("Z:")
            # print(Z.flags)
        except:
            if self.verbose:
                print("No init partition passed")
        try:
            self.Q = data["Q"]
        except:
            if self.verbose:
                print(
                    "Inferring Q from given Z:",
                    "\nTaking as max(Z) + 1, so -1 may be used for missing data",
                )
            self.Q = self.init_Z.max() + 1
        try:
            self.meta_types = List()
            for mt in data["meta_types"]:
                self.meta_types.append(mt)
            assert len(self.meta_types) == len(self.X)
        except:
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

        except:
            if self.verbose:
                print("No p_in / p_out provided")
        if sparse:
            self.dsbmm = dsbmm_sparse.DSBMMSparse(
                A=self.A,
                X_poisson=self.X_poisson,
                X_ib=self.X_ib,
                Z=self.init_Z.copy(),
                Q=self.Q,
                meta_types=self.meta_types,
                verbose=self.verbose,
            )  # X=X,
            if self.verbose:
                print("Successfully instantiated DSBMM...")
            self.bp = bp_sparse.BPSparse(self.dsbmm)
            if self.verbose:
                print("Successfully instantiated BP system...")
        else:
            self.dsbmm = dsbmm.DSBMM(
                A=self.A,
                X_poisson=self.X_poisson,
                X_ib=self.X_ib,
                Z=self.init_Z.copy(),
                Q=self.Q,
                meta_types=self.meta_types,
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

    def fit(
        self,
        max_iter=MAX_ITER,
        max_msg_iter=MAX_MSG_ITER,
        conv_tol=CONV_TOL,
        msg_conv_tol=MSG_CONV_TOL,
        true_Z=None,
        learning_rate=0.2,
    ):
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
            if self.verbose:
                self.bp.model.set_Z_by_MAP()
                if true_Z is not None:
                    current_score = self.ari_score(true_Z)
                    self.ydata.append(current_score.mean())
                    self.update_score_plot()
                    print()
        self.bp.model.set_Z_by_MAP()

    def ari_score(self, true_Z, pred_Z=None):
        # wait for a second to execute in case of parallelisation issues
        time.sleep(1)
        if pred_Z is None:
            try:
                # could try and replace with numba again but seemed
                # to cause issues
                aris = np.array(
                    [
                        ari(true, pred)
                        for true, pred in zip(self.bp.model.jit_model.Z.T, true_Z.T)
                    ]
                )
                if self.verbose:
                    print("ARIs:", aris)
                return aris
            except:
                print(
                    "Can't provide ARI score - make sure have both",
                    "provided ground truth \nwhen initialising model,",
                    "and have called fit",
                )
        else:
            aris = np.array([ari(true, pred) for true, pred in zip(pred_Z.T, true_Z.T)])
            if self.verbose:
                print("ARIs:", aris)
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


def calc_present(A):
    """Calculate whether nodes present at each time period given adjacency
    (i.e. either send or receive a link)

    Args:
        A (_type_): N x N x T adjacency (assume sparse)
    """
    if type(A) == np.ndarray:
        present = (A.sum(axis=0) > 0) | (A.sum(axis=1) > 0)
    elif type(A) == list:
        present = np.vstack(
            [(A[t].sum(axis=0) > 0) | (A[t].sum(axis=1) > 0) for t in range(len(A))]
        )
    return present


def calc_trans_present(present):
    """Calculate whether nodes present in adjacent time periods and so should be
    counted in transitions

    Args:
        present (_type_): N x T boolean for presence of node i at time t 
    """
    return present[:, :-1] & present[:, 1:]


def effective_pi(Z):
    Q = len(np.unique(Z))
    z_vals = np.unique(Z)
    print("Unique Z vals:", z_vals)
    print(f"Q inferred as {Q}")
    T = Z.shape[1]
    print(f"T inferred as {T}")
    qqprime_trans = np.zeros((Q, Q))
    for q in np.unique(Z):
        for qprime in np.unique(Z):
            for t in range(1, T):
                tm1_idxs = Z[:, t - 1] == q
                t_idxs = Z[:, t] == qprime
                qqprime_trans[z_vals == q, z_vals == qprime] += (
                    tm1_idxs * t_idxs
                ).sum()
    print("Num. trans. inferred:", qqprime_trans)
    qqprime_trans /= np.expand_dims(qqprime_trans.sum(axis=1), 1)
    return qqprime_trans


def effective_beta(A, Z):
    Q = len(np.unique(Z))
    z_vals = np.unique(Z)
    T = Z.shape[1]
    beta = np.zeros((Q, Q, T))
    if type(A) == np.ndarray:
        for q in z_vals:
            for r in z_vals:
                for t in range(T):
                    beta[q, r, t] = A[:, :, t][
                        np.ix_(Z[:, t] == q, Z[:, t] == r)
                    ].mean()
    elif type(A) == list:
        for q in z_vals:
            for r in z_vals:
                for t in range(T):
                    beta[q, r, t] = A[t][np.ix_(Z[:, t] == q, Z[:, t] == r)].mean()
    return beta / 2


if __name__ == "__main__":
    ## Simulate data (for multiple tests)
    default_test_params = simulation.default_test_params
    og_test_params = simulation.og_test_params
    scaling_test_params = simulation.scaling_test_params
    testset_names = ["og", "default", "scaling"]
    testset_name = testset_names[2]
    # choose which set of tests to run
    if testset_name == "og":
        test_params = og_test_params
    elif testset_name == "default":
        test_params = default_test_params
    elif testset_name == "scaling":
        test_params = scaling_test_params

    # NB n_samps, p_out, T, meta_types, L, meta_dims all fixed
    # in default test set - all other params change over 12 tests
    all_samples = []
    params_set = []
    chosen_test_idx = 10
    # print(test_params)
    for i, testno in enumerate(test_params["test_no"]):
        # if i == chosen_test_idx:
        if testset_name in ["default", "scaling"]:
            params = {
                "test_no": testno,
                "N": test_params["N"][i],
                "Q": test_params["Q"][i],
                "p_in": test_params["p_in"][i],
                "p_stay": test_params["p_stay"][i],
                "n_samps": test_params["n_samps"],
                "p_out": test_params["p_out"][i],
                "T": test_params["T"],
                "meta_types": test_params["meta_types"],
                "L": test_params["L"],
                "meta_dims": test_params["meta_dims"],
                "pois_params": test_params["meta_params"][i][0],
                "indep_bern_params": test_params["meta_params"][i][1],
                "meta_aligned": test_params["meta_align"][i],
            }
        elif testset_name == "og":
            params = {
                "test_no": testno,
                "N": test_params["N"],
                "Q": test_params["Q"],
                "beta_mat": test_params["beta_mat"][i],
                "trans_mat": test_params["trans_mat"][i],
                "n_samps": test_params["n_samps"],
                "T": test_params["T"][i],
                "meta_types": test_params["meta_types"],
                "L": test_params["L"],
                "meta_dims": test_params["meta_dims"],
                # "pois_params": og_test_params["meta_params"][i][0],
                # "indep_bern_params": og_test_params["meta_params"][i][1],
                "sample_meta_params": test_params["sample_meta_params"],
            }
        # print(params)
        samples = simulation.gen_test_data(**params)
        if testset_name == "scaling":
            print()
            print(f"Simulated test {testno}")
            for i, sample in enumerate(samples):
                print(f"...converting sample {i+1} to sparse format")
                sample["A"] = [
                    sparse.csr_matrix(sample["A"][:, :, t]) for t in range(params["T"])
                ]
            print("...done")
        # print(samples)
        all_samples.append(samples)
        params_set.append(params)
    # Example of specifying own params for simulating data:
    # N = 500
    # Q = 20
    # T = 5
    # p_in = 0.02
    # p_out = 0.01
    # p_stay = 0.8
    # trans_prob = simulation.gen_trans_mat(p_stay, Q)
    # Z_1 = np.random.randint(0, Q, size=(N,))
    # Z_1 = np.sort(Z_1)
    # meta_types = List()
    # meta_types.append("poisson")
    # meta_types.append("indep bernoulli")
    # L = 5
    # meta_params = [np.random.randint(5, 15, size=(1, Q, T)), np.random.rand(L, Q, T)]
    # data = simulation.sample_dynsbm_meta(
    #     Z_1=Z_1,
    #     Q=Q,
    #     T=T,
    #     p_in=p_in,
    #     p_out=p_out,
    #     trans_prob=trans_prob,
    #     meta_types=meta_types,
    #     meta_params=meta_params,
    # )
    print("Successfully simulated data, now initialising model...")

    ## Process empirical data
    pass  # not needed as simulating directly in form needed

    use_X_init = False
    verbose = False
    if testset_name == "og":
        test_aris = [np.zeros((20, T)) for T in test_params["T"]]
    elif testset_name in ["default", "scaling"]:
        test_aris = np.zeros((len(all_samples), 20, 5))
    test_times = np.zeros((len(all_samples), 19))
    for test_no, (samples, params) in enumerate(zip(all_samples, params_set)):
        # if test_no < 5:
        print()
        print("*" * 15, f"Test {test_no+1}", "*" * 15)
        for samp_no, sample in enumerate(samples):
            if samp_no < len(samples):  # can limit num samples considered
                if samp_no > 0:
                    # drop first run as compiling
                    start_time = time.time()
                if verbose:
                    print("true params:", params)
                print()
                print("$" * 12, "At sample", samp_no + 1, "$" * 12)
                sample.update(params)
                # present = calc_present(sample["A"])
                # trans_present = calc_trans_present(present)
                # print(present)
                # print(trans_present)+
                ## Initialise model
                true_Z = sample["Z"]

                if use_X_init:
                    kmeans_mat = np.concatenate(
                        [
                            *[sample["A"][:, :, t] for t in range(sample["T"])],
                            *[
                                Xs.transpose(1, 2, 0).reshape(params["N"], -1)
                                for Xs in sample["X"].values()
                            ],
                        ],
                        axis=1,
                    )  # done for fixing labels over time
                else:
                    if testset_name != "scaling":
                        kmeans_mat = np.concatenate(
                            [sample["A"][:, :, t] for t in range(sample["T"])], axis=1
                        )  # done for fixing labels over time
                    else:
                        kmeans_mat = sparse.hstack(sample["A"])
                if sample["N"] > 1e5:
                    kmeans_labels = (
                        MiniBatchKMeans(
                            n_clusters=params["Q"],
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
                            n_clusters=params["Q"],
                            #   random_state=0, # TODO: consider allowing fixing this for reproducibility
                            max_iter=10,
                        )
                        .fit_predict(kmeans_mat)
                        .reshape(-1, 1)
                    )
                init_Z = np.tile(kmeans_labels, (1, params["T"]))
                # add some noise to init clustering
                prop = 0.1  # proportion of noise to add
                mask = np.random.rand(*init_Z.shape) < prop
                init_Z[mask] += np.random.randint(
                    -sample["Q"] // 2,
                    sample["Q"] // 2 + 1,
                    size=(sample["N"], sample["T"]),
                )[mask]
                init_Z[init_Z < 0] = 0
                init_Z[init_Z > sample["Q"] - 1] = sample["Q"] - 1
                try:
                    assert init_Z.shape == sample["Z"].shape
                except:
                    print(init_Z.shape)
                    print(sample["Z"].shape)
                    raise ValueError("Wrong partition shape")
                sample["Z"] = init_Z
                ## Initialise
                if testset_name != "scaling":
                    em = EM(sample, verbose=verbose)
                else:
                    em = EM(sample, sparse=True, verbose=verbose)
                ## Score from K means
                print("Before fitting model, K-means init partition has")
                if verbose:
                    em.ari_score(true_Z)
                else:
                    print(em.ari_score(true_Z))
                ## Fit to given data
                em.fit(true_Z=true_Z, learning_rate=0.2)
                ## Score after fit
                try:
                    test_aris[test_no, samp_no, :] = 0.0
                    print("BP Z ARI:")
                    test_aris[test_no, samp_no, :] = em.ari_score(true_Z)
                    if not verbose:
                        print(test_aris[test_no, samp_no, :])
                    print("Init Z ARI:")
                    if verbose:
                        em.ari_score(true_Z, pred_Z=em.init_Z)
                    else:
                        print(em.ari_score(true_Z, pred_Z=em.init_Z))
                except:
                    print("BP Z ARI:")
                    test_aris[test_no][samp_no, :] = em.ari_score(true_Z)
                    if not verbose:
                        print(test_aris[test_no][samp_no, :])
                    print("Init Z ARI:")
                    if verbose:
                        em.ari_score(true_Z, pred_Z=em.init_Z)
                    else:
                        print(em.ari_score(true_Z, pred_Z=em.init_Z))
                # print("Z inferred:", em.bp.model.jit_model.Z)
                if verbose:
                    ## Show transition matrix inferred
                    print("Pi inferred:", em.bp.trans_prob)
                    try:
                        print("Versus true pi:", params["trans_mat"])
                    except:
                        print(
                            "Versus true pi:",
                            simulation.gen_trans_mat(sample["p_stay"], sample["Q"]),
                        )
                    print("True effective pi:", effective_pi(true_Z))
                    print(
                        "Effective pi from partition inferred:",
                        effective_pi(em.bp.model.jit_model.Z),
                    )
                    print(
                        "True effective beta:",
                        effective_beta(em.bp.model.jit_model.A, true_Z).transpose(
                            2, 0, 1
                        ),
                    )
                    print(
                        "Pred effective beta:",
                        effective_beta(
                            em.bp.model.jit_model.A, em.bp.model.jit_model.Z
                        ).transpose(2, 0, 1),
                    )
                if samp_no > 0:
                    test_times[test_no, samp_no - 1] = time.time() - start_time
                time.sleep(
                    1
                )  # sleep a second to allow threads to complete, TODO: properly sort this
        print(f"Finished test {test_no} for true params:")
        print(params)
        print(f"Mean ARIs: {test_aris[test_no].mean(axis=0)}")
    print()
    print("Mean ARIs inferred for each test:")
    try:
        print(test_aris.mean(axis=(1, 2)))
    except:
        print(np.array([aris.mean() for aris in test_aris]))
    print("Mean times for each test:")
    print(test_times.mean(axis=1))

    with open(f"../../results/{testset_name}_test_aris.pkl", "wb") as f:
        pickle.dump(test_aris, f)
    with open(f"../../results/{testset_name}_test_times.pkl", "wb") as f:
        pickle.dump(test_times, f)
