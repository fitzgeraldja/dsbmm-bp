import argparse
import pickle
import time
from pathlib import Path

import em
import numpy as np
import simulation
import utils
from numba.typed import List
from scipy import sparse

parser = argparse.ArgumentParser(description="Apply model to data.")


testset_names = ["og", "default", "scaling", "align", "scopus"]
parser.add_argument(
    "--test",
    type=str,
    default="default",
    help=f"Run chosen set of provided tests, options are {testset_names}, default is default.",
    choices=testset_names,
)
parser.add_argument(
    "--data",
    type=str,
    default="./data",
    help="Specify path to data directory. Default is ./data",
)

parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose output")

parser.add_argument(
    "--nb_parallel",
    "-nbp",
    action="store_true",
    help="Try with numba parallelism (currently slower)",
)

parser.add_argument(
    "--tuning_param",
    "-tunp",
    type=float,
    default=1.0,
)

args = parser.parse_args()


if __name__ == "__main__":
    ## Simulate data (for multiple tests)
    default_test_params = simulation.default_test_params
    og_test_params = simulation.og_test_params
    scaling_test_params = simulation.scaling_test_params
    align_test_params = simulation.align_test_params
    testset_name = args.test
    # choose which set of tests to run
    if testset_name == "og":
        test_params = og_test_params
    elif testset_name == "default":
        test_params = default_test_params
    elif testset_name == "scaling":
        test_params = scaling_test_params
    elif testset_name == "align":
        test_params = align_test_params
    else:
        test_params = None
    # NB n_samps, p_out, T, meta_types, L, meta_dims all fixed
    # in default test set - all other params change over 12 tests
    if test_params is not None:
        all_samples = []
        params_set = []
        chosen_test_idx = 10
        # print(test_params)
        for i, testno in enumerate(test_params["test_no"]):
            # if i == chosen_test_idx:
            # if i < 3:  # just take first 3 tests for example to start
            if testset_name in ["default", "scaling", "align"]:
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
            if testset_name in ["scaling", "align"]:
                try:
                    with open(
                        f"../../results/{testset_name}_{testno}_samples.pkl",
                        "rb",
                    ) as f:
                        samples = pickle.load(f)
                except Exception:  # FileNotFoundError: NB only base exception supported by numba
                    samples = simulation.gen_test_data(**params)
                    print()
                    print(f"Simulated test {testno}")
                    if testset_name == "scaling":
                        for i, sample in enumerate(samples):
                            print(f"...converting sample {i+1} to sparse format")
                            sample["A"] = [
                                sparse.csr_matrix(sample["A"][:, :, t])
                                for t in range(params["T"])
                            ]
                        print("...done")
                    with open(
                        f"../../results/{testset_name}_{testno}_samples.pkl",
                        "wb",
                    ) as f:
                        pickle.dump(samples, f)
            else:
                samples = simulation.gen_test_data(**params)
            # print(samples)
            all_samples.append(samples)
            params_set.append(params)
        print("Successfully simulated data, now initialising model...")
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

    ## Load empirical data
    if testset_name == "scopus":
        data = {}
        DATA_PATH = Path("../../tests/data/scopus")
        link_choice = "ref"  # 'ref' or 'au'
        if link_choice == "au":
            with open(DATA_PATH / "col_A.pkl", "rb") as f:
                data["A"] = [sparse.csr_matrix(A) for A in pickle.load(f)]
            with open(DATA_PATH / "col_ages.pkl", "rb") as f:
                # given as full time series so only take 4 most recent as done
                # for A
                data["X_ages"] = pickle.load(f)[-4:].transpose(1, 0, 2)
            with open(DATA_PATH / "col_insts.pkl", "rb") as f:
                data["X_insts"] = pickle.load(f)[-4:].transpose(1, 0, 2)
            with open(DATA_PATH / "col_subjs.pkl", "rb") as f:
                data["X_subjs"] = pickle.load(f)[-4:].transpose(1, 0, 2)
        elif link_choice == "ref":
            with open(DATA_PATH / "col_ref_A.pkl", "rb") as f:
                data["A"] = [sparse.csr_matrix(A) for A in pickle.load(f)]
            with open(DATA_PATH / "col_ref_ages.pkl", "rb") as f:
                # given as full time series so only take 4 most recent as done
                # for A
                data["X_ages"] = pickle.load(f)[-4:].transpose(1, 0, 2)
            with open(DATA_PATH / "col_ref_insts.pkl", "rb") as f:
                data["X_insts"] = pickle.load(f)[-4:].transpose(1, 0, 2)
            with open(DATA_PATH / "col_ref_subjs.pkl", "rb") as f:
                data["X_subjs"] = pickle.load(f)[-4:].transpose(1, 0, 2)
        else:
            raise ValueError("Unknown link choice passed")
        # data['X'] = [v for k,v in data.items() if 'X' in k]
        tmp = List()
        tmp.append(np.ascontiguousarray(data["X_ages"]))
        tmp.append(np.ascontiguousarray(data["X_insts"]))
        tmp.append(np.ascontiguousarray(data["X_subjs"]))
        data["X"] = tmp
        data["meta_types"] = ["poisson", "indep bernoulli", "indep bernoulli"]
        data["Q"] = 22 if link_choice == "au" else 19 if link_choice == "ref" else None

    try_parallel = args.nb_parallel

    use_X_init = False
    verbose = args.verbose
    if testset_name != "scopus":
        if testset_name == "og":
            test_aris = [np.zeros((20, T)) for T in test_params["T"]]
        elif testset_name in ["default", "scaling", "align"]:
            test_aris = np.zeros(
                (len(all_samples), test_params["n_samps"], test_params["T"])
            )
        test_times = np.zeros((len(all_samples), test_params["n_samps"] - 1))
        init_times = np.zeros_like(test_times)
        if testset_name == "scaling":
            test_Ns = [param["N"] for param in params_set]
            with open(f"../../results/{testset_name}_N.pkl", "wb") as f:
                pickle.dump(test_Ns, f)
        elif testset_name == "align":
            test_Z = np.zeros(
                (
                    len(all_samples),
                    params_set[0]["n_samps"],
                    params_set[0]["N"],
                    test_params["T"],
                )
            )
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
                    true_Z = sample.pop("Z")
                    ## Initialise
                    if testset_name != "scaling":
                        model = em.EM(sample, verbose=verbose)
                    elif testset_name == "align":
                        print(f"alignment = {params['meta_aligned']}")
                        model = em.EM(
                            sample, verbose=verbose, tuning_param=args.tuning_param
                        )
                    else:
                        print(f"N = {params['N']}")
                        model = em.EM(
                            sample,
                            sparse_adj=True,
                            try_parallel=try_parallel,
                            verbose=verbose,
                        )
                    if samp_no > 0:
                        init_times[test_no, samp_no - 1] = time.time() - start_time
                    ## Score from K means
                    print("Before fitting model, K-means init partition has")
                    if verbose:
                        model.ari_score(true_Z, pred_Z=model.k_means_init_Z)
                    else:
                        print(
                            np.round_(
                                model.ari_score(true_Z, pred_Z=model.k_means_init_Z),
                                3,
                            )
                        )
                    ## Fit to given data
                    model.fit(true_Z=true_Z, learning_rate=0.2)
                    ## Score after fit
                    try:
                        test_aris[test_no, samp_no, :] = 0.0
                        print("BP Z ARI:")
                        test_aris[test_no, samp_no, :] = model.ari_score(true_Z)
                        if not verbose:
                            print(np.round_(test_aris[test_no, samp_no, :], 3))
                        print("Init Z ARI:")
                        if verbose:
                            model.ari_score(true_Z, pred_Z=model.k_means_init_Z)
                        else:
                            print(
                                np.round_(
                                    model.ari_score(
                                        true_Z, pred_Z=model.k_means_init_Z
                                    ),
                                    3,
                                )
                            )
                    except Exception:  # IndexError:
                        print("BP Z ARI:")
                        test_aris[test_no][samp_no, :] = model.ari_score(true_Z)
                        if not verbose:
                            print(np.round_(test_aris[test_no][samp_no, :], 3))
                        print("Init Z ARI:")
                        if verbose:
                            model.ari_score(true_Z, pred_Z=model.k_means_init_Z)
                        else:
                            print(
                                np.round_(
                                    model.ari_score(
                                        true_Z, pred_Z=model.k_means_init_Z
                                    ),
                                    3,
                                )
                            )
                    # print("Z inferred:", model.bp.model.jit_model.Z)
                    if verbose:
                        ## Show transition matrix inferred
                        print("Pi inferred:", model.bp.trans_prob)
                        try:
                            print("Versus true pi:", params["trans_mat"])
                        except Exception:  # KeyError:
                            print(
                                "Versus true pi:",
                                simulation.gen_trans_mat(sample["p_stay"], sample["Q"]),
                            )
                        print("True effective pi:", utils.effective_pi(true_Z))
                        print(
                            "Effective pi from partition inferred:",
                            utils.effective_pi(model.bp.model.jit_model.Z),
                        )
                        print(
                            "True effective beta:",
                            utils.effective_beta(
                                model.bp.model.jit_model.A, true_Z
                            ).transpose(2, 0, 1),
                        )
                        print(
                            "Pred effective beta:",
                            utils.effective_beta(
                                model.bp.model.jit_model.A,
                                model.bp.model.jit_model.Z,
                            ).transpose(2, 0, 1),
                        )
                    if samp_no > 0:
                        test_times[test_no, samp_no - 1] = time.time() - start_time
                        if testset_name == "scaling":
                            print(f"Sample took ~{test_times[test_no,samp_no-1]:.2f}s")
                    time.sleep(
                        0.5
                    )  # sleep a bit to allow threads to complete, TODO: properly sort this
                    # save after every sample in case of crash
                    tp_str = (
                        "_tp" + str(args.tuning_param)
                        if args.tuning_param is not None
                        else ""
                    )
                    if testset_name == "align":
                        test_Z[test_no, samp_no, :, :] = model.bp.model.jit_model.Z
                        with open(
                            f"../../results/{testset_name}_test_Z{tp_str}.pkl",
                            "wb",
                        ) as f:
                            pickle.dump(test_Z, f)
                    with open(
                        f"../../results/{testset_name}_test_aris{tp_str}.pkl", "wb"
                    ) as f:
                        pickle.dump(test_aris, f)
                    with open(
                        f"../../results/{testset_name}_test_times{tp_str}.pkl", "wb"
                    ) as f:
                        pickle.dump(test_times, f)
                    with open(
                        f"../../results/{testset_name}_init_times{tp_str}.pkl", "wb"
                    ) as f:
                        pickle.dump(init_times, f)
            print(f"Finished test {test_no+1} for true params:")
            print(params)
            print(f"Mean ARIs: {test_aris[test_no].mean(axis=0)}")
        print()
        print("Mean ARIs inferred for each test:")
        try:
            print(test_aris.mean(axis=(1, 2)))
        except Exception:  # AttributeError:
            print(np.array([aris.mean() for aris in test_aris]))
        print("Mean times for each test:")
        print(test_times.mean(axis=1))
    else:
        T = len(data["A"])
        N = data["A"][0].shape[0]
        n_runs = 5
        test_Z = np.zeros((n_runs, N, T))
        print("*" * 15, "Running Scopus data", "*" * 15)
        ## Initialise
        model = em.EM(
            data,
            sparse_adj=True,
            tuning_param=np.linspace(0.8, 1.8, 11),
            n_runs=n_runs,
            deg_corr=True,
            verbose=verbose,
        )
        ## Fit to given data
        model.fit(learning_rate=0.2)
        pred_Z = model.bp.model.jit_model.Z
        print(f"Best tuning param: {model.best_tun_param}")
        with open(f"../../results/{testset_name}_{link_choice}_Z.pkl", "wb") as f:
            pickle.dump(pred_Z, f)

    # TODO: clean up code and improve documentation, then
    # run poetry publish
