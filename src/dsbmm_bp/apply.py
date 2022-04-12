import simulation
import data_processer
import dsbmm
import bp
from utils import nb_ari_local  # , nb_nmi_local
import numpy as np
from numba.typed import List
from sklearn.cluster import MiniBatchKMeans
import time

import faulthandler
from numba import gdb

faulthandler.enable()

MAX_MSG_ITER = 5
MSG_CONV_TOL = 1e-4
MAX_ITER = 10
CONV_TOL = 1e-4


class EM:
    def __init__(self, data, msg_init_mode="planted", verbose=True):
        self.verbose = verbose
        self.A = data["A"]
        try:
            assert np.allclose(self.A, self.A.transpose(1, 0, 2))
        except:
            # symmetrise for this test case
            print(
                "WARNING: provided non-symmetric adjacency,",
                "\nsuggesting directed network:",
                "\nThis is currently unimplemented - symmetrising and binarising...",
            )
            self.A = ((self.A + self.A.transpose(1, 0, 2)) > 0) * 1.0
        # NB expecting A in shape N x N x T
        self.N = self.A.shape[0]
        self.T = self.A.shape[2]
        self.X = data["X"]
        self.X_poisson = np.ascontiguousarray(self.X["poisson"].transpose(1, 2, 0))
        self.X_ib = np.ascontiguousarray(self.X["indep bernoulli"].transpose(1, 2, 0))
        # self.X_poisson = data["X_poisson"]
        # self.X_ib = data["X_ib"]
        try:
            self.Z = data["Z"]
            # print("Z:")
            # print(Z.flags)
        except:
            if self.verbose:
                print("No ground truth partition passed")
        try:
            self.Q = data["Q"]
        except:
            if self.verbose:
                print(
                    "Inferring Q from given Z:",
                    "\nTaking as max(Z) + 1, so -1 may be used for missing data",
                )
            self.Q = self.Z.max() + 1
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
                print("No known parameters provided")

        # TODO: remove after fix
        # gdb()
        self.dsbmm = dsbmm.DSBMM(
            A=self.A,
            X_poisson=self.X_poisson,
            X_ib=self.X_ib,
            Z=self.Z,
            Q=self.Q,
            meta_types=self.meta_types,
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
            self.bp.model.update_params(learning_rate)
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
                    self.ari_score(true_Z)
        self.bp.model.set_Z_by_MAP()

    def ari_score(self, true_Z):
        # wait for a second to execute in case of parallelisation issues
        time.sleep(1)
        try:
            aris = nb_ari_local(self.bp.model.jit_model.Z, true_Z)
            if self.verbose:
                print("ARIs:", aris)
            return aris
        except:
            print(
                "Can't provide ARI score - make sure have both",
                "provided ground truth \nwhen initialising model,",
                "and have called fit",
            )

    # print("NMIs: ", nb_nmi_local(bp_ex.model.Z, Z))

    # print("Overlaps:", (bp_ex.model.Z == Z).sum(axis=0) / Z.shape[0])


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


if __name__ == "__main__":
    ## Simulate data (for multiple tests)
    default_test_params = simulation.default_test_params
    og_test_params = simulation.og_test_params
    # NB n_samps, p_out, T, meta_types, L, meta_dims all fixed
    # in default test set - all other params change over 12 tests
    all_samples = []
    params_set = []
    for i, testno in enumerate(og_test_params["test_no"]):
        if i < 2:
            # default:
            # params = {
            #     "test_no": testno,
            #     "N": default_test_params["N"][i],
            #     "Q": default_test_params["Q"][i],
            #     "p_in": default_test_params["p_in"][i],
            #     "p_stay": default_test_params["p_stay"][i],
            #     "n_samps": default_test_params["n_samps"],
            #     "p_out": default_test_params["p_out"],
            #     "T": default_test_params["T"],
            #     "meta_types": default_test_params["meta_types"],
            #     "L": default_test_params["L"],
            #     "meta_dims": default_test_params["meta_dims"],
            #     "pois_params": default_test_params["meta_params"][i][0],
            #     "indep_bern_params": default_test_params["meta_params"][i][1],
            #     "meta_aligned": default_test_params["meta_align"][i],
            # }
            # og:
            params = {
                "test_no": testno,
                "N": og_test_params["N"],
                "Q": og_test_params["Q"],
                "beta_mat": og_test_params["beta_mat"][i],
                "trans_mat": og_test_params["trans_mat"][i],
                "n_samps": og_test_params["n_samps"],
                "T": og_test_params["T"][i],
                "meta_types": og_test_params["meta_types"],
                "L": og_test_params["L"],
                "meta_dims": og_test_params["meta_dims"],
                # "pois_params": og_test_params["meta_params"][i][0],
                # "indep_bern_params": og_test_params["meta_params"][i][1],
                "sample_meta_params": og_test_params["sample_meta_params"],
            }
            # print(params)
            samples = simulation.gen_test_data(**params)
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
    test_aris = np.zeros((len(all_samples), 20, 5))
    for test_no, (samples, params) in enumerate(zip(all_samples, params_set)):
        for samp_no, sample in enumerate(samples):
            print()
            print("$" * 12, "At sample", samp_no, "$" * 12)
            sample.update(params)
            ## Initialise model
            true_Z = sample["Z"]

            if use_X_init:
                kmeans_mat = np.concatenate(
                    [
                        sample["A"].reshape(params["N"], -1),
                        *[
                            Xs.transpose(1, 2, 0).reshape(params["N"], -1)
                            for Xs in sample["X"].values()
                        ],
                    ],
                    axis=1,
                )  # done for fixing labels over time
            else:
                kmeans_mat = sample["A"].reshape(
                    params["N"], -1
                )  # done for fixing labels over time
            kmeans_labels = (
                MiniBatchKMeans(
                    n_clusters=params["Q"],
                    #   random_state=0, # TODO: consider allowing fixing this for reproducibility
                    batch_size=20,
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
                -sample["Q"] // 2, sample["Q"] // 2, size=(sample["N"], sample["T"])
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
            em = EM(sample)
            ## Score from K means
            print("Before fitting model, K-means init partition has")
            em.ari_score(true_Z)
            ## Fit to given data
            em.fit(true_Z=true_Z, learning_rate=0.2)
            ## Score after fit
            test_aris[test_no, samp_no, :] = em.ari_score(true_Z)
            print("Z inferred:", em.bp.model.jit_model.Z)
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
            time.sleep(
                1
            )  # sleep a second to allow threads to complete, TODO: properly sort this
        print(f"Finished test {test_no}:")
        print(f"Mean ARIs: {test_aris[test_no].mean(axis=0)}")

    print("Mean ARIs inferred for each test:")
    print(test_aris.mean(axis=(1, 2)))

