import argparse
import os
import pickle
import time
import warnings
from pathlib import Path

import csr
import em
import mlflow
import numpy as np
import simulation
import utils
from mlflow import log_artifacts, log_metric, log_param
from numba.typed import List
from scipy import sparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Apply model to data.")

parser.add_argument(
    "--data",
    type=str,
    default="./data",
    help="Specify path to data directory. Default is ./data",
)

parser.add_argument(
    "--max_msg_iter",
    type=int,
    default=300,
    help="Maximum number of message updates to run. Default is 300.",
)

parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose output")

parser.add_argument(
    "--use-numba",
    "-nb",
    action="store_true",
    help="Try with numba rather than numpy - currently seems slower for all, maybe other than v large nets",
)

parser.add_argument(
    "--nb_parallel",
    "-nbp",
    action="store_true",
    help="Try with numba parallelism (currently slower)",
)

parser.add_argument(
    "--n_threads", type=int, help="Set number of threads used for parallelism"
)

parser.add_argument(
    "--n_runs", type=int, default=5, help="Set number of runs, default = 5"
)

parser.add_argument(
    "--tuning_param",
    "-tunp",
    type=float,
    default=1.0,
    help="Set metadata tuning parameter value, default = 1.0",
)

parser.add_argument(
    "--ignore_meta",
    action="store_true",
    help="Ignore metadata, use only network edges for fitting model",
)

parser.add_argument(
    "--check_conn",
    action="store_true",
    help="Check connectivity of samples for each test prior to application.",
)

args = parser.parse_args()

if args.n_threads is not None:
    from numba import set_num_threads

    set_num_threads(args.n_threads)


if __name__ == "__main__":
    # if not os.path.exists("../../results/mlruns"):
    #     os.mkdir("../../results/mlruns")
    # dir_path = os.path.abspath("../../results/mlruns")
    # mlflow.set_tracking_uri(f"file://{dir_path}")
    # Set an experiment name, which must be unique
    # and case-sensitive.
    log_runs = False
    if log_runs:
        try:
            experiment_id = mlflow.create_experiment(f"dsbmm-alpha-toy")
            experiment = mlflow.get_experiment(experiment_id)
        except:
            experiment = mlflow.set_experiment(f"dsbmm-alpha-toy")
    # Get Experiment Details
    # print("Experiment_id: {}".format(experiment.experiment_id))

    # params = {
    #     "test_no": testno,
    #     "N": test_params["N"][i],
    #     "Q": test_params["Q"][i],
    #     "p_in": test_params["p_in"][i],
    #     "p_stay": test_params["p_stay"][i],
    #     "n_samps": test_params["n_samps"],
    #     "p_out": test_params["p_out"][i],
    #     "T": test_params["T"],
    #     "meta_types": test_params["meta_types"],
    #     "L": test_params["L"],
    #     "meta_dims": test_params["meta_dims"],
    #     "pois_params": test_params["meta_params"][i][0],
    #     "indep_bern_params": test_params["meta_params"][i][1],
    #     "meta_aligned": test_params["meta_align"][i],
    # }
    N = 200
    T = 10
    Q = 2
    c = 10
    n_samps = 20
    eps_grid = np.linspace(0.3, 0.6, 10)
    eta_grid = np.linspace(0.4, 0.8, 10)
    rho_grid = np.linspace(0.6, 1.0, 10)

    def get_p_in(eps):
        return Q * c / N * (1 + (Q - 1) * eps)

    def get_p_out(eps):
        return eps * Q * c / N * (1 + (Q - 1) * eps)

    data_path = Path("/scratch/fitzgeraldj/data") / "toy_sims"
    data_path.mkdir(exist_ok=True)
    results_dir = data_path.parent / "toy_results"
    results_dir.mkdir(exist_ok=True)

    try:
        with open(data_path / "toy_samples.pkl", "rb") as f:
            all_samples = pickle.load(f)
        with open(data_path / "toy_param_grid.pkl", "rb") as f:
            param_grid = pickle.load(f)
    except FileNotFoundError:
        print("Existing samples not found, generating data...")
        all_samples, param_grid = simulation.toy_tests(
            N=N,
            T=T,
            Q=Q,
            c=c,
            n_samps=n_samps,
            eps_grid=eps_grid,
            eta_grid=eta_grid,
            rho_grid=rho_grid,
        )
        with open(data_path / "toy_samples.pkl", "wb") as f:  # type: ignore
            pickle.dump(all_samples, f)
        with open(data_path / "toy_param_grid.pkl", "wb") as f:  # type: ignore
            pickle.dump(param_grid, f)

    if args.check_conn:
        print("Checking connectivity of test samples...")
        for testno, samples in enumerate(tqdm(all_samples)):
            # print(samples)
            try:
                if not np.all(
                    [
                        utils.is_connected_dense(sample["A"][:, :, t])
                        for sample in samples
                        for t in range(sample["A"].shape[-1])
                    ]
                ):
                    warnings.warn(
                        f"Some matrices not connected in test set for test no. {testno}"
                    )
            except Exception:
                # assert np.all(
                #     samples[0]["A"][0].todense() == samples[0]["A"][0].T.todense()
                # )
                if not np.all(
                    [
                        utils.is_connected_sparse(csr.CSR.from_scipy(sample["A"][t]))
                        for sample in samples
                        for t in range(len(sample["A"]))
                    ]
                ):
                    warnings.warn(
                        f"Some matrices not connected in test set for test no. {testno}"
                    )
                    # print(f"Components for {samples[0].get('A')[0].shape[0]} nodes:")
                    # print(
                    #     *[
                    #         [
                    #             list(
                    #                 map(
                    #                     len,
                    #                     list(
                    #                         utils.connected_components_sparse(
                    #                             csr.CSR.from_scipy(sample["A"][t])
                    #                         )
                    #                     ),
                    #                 )
                    #             )
                    #             for t in range(len(sample["A"]))
                    #         ]
                    #         for sample in samples
                    #     ],
                    #     sep="\n",
                    # )
                    # raise Exception("Stop here for now")

    print("Successfully loaded data, now initialising model...")

    # tmp = List()
    # tmp.append(np.ascontiguousarray(data["X_ages"]))
    # tmp.append(np.ascontiguousarray(data["X_insts"]))
    # tmp.append(np.ascontiguousarray(data["X_subjs"]))
    # data["X"] = tmp
    # data["meta_types"] = ["poisson", "indep bernoulli", "indep bernoulli"]

    try_parallel = args.nb_parallel

    use_X_init = False
    verbose = args.verbose

    n_tests = len(all_samples)
    test_aris = np.zeros((n_tests, n_samps, T))  # n_tests x n_samps x T
    max_en_aris = np.zeros_like(test_aris)
    test_times = np.zeros((n_tests, n_samps - 1))
    init_times = np.zeros_like(test_times)

    testset_name = "toy"
    for test_no, (samples, params) in enumerate(zip(tqdm(all_samples), param_grid)):
        eps, eta, rho = params
        tqdm.write()
        tqdm.write("*" * 15, f"Test {test_no+1}", "*" * 15)
        # Create nested runs for each test + sample
        # if log_runs:
        #     experiment_id = experiment.experiment_id
        # else:
        #     experiment_id = 0
        # with mlflow.start_run(
        #     run_name=f"PARENT_RUN_{test_no}", experiment_id=experiment_id
        # ) as parent_run:
        # mlflow.log_param("parent", "yes")
        for samp_no, sample in enumerate(samples):
            # with mlflow.start_run(
            #     run_name=f"CHILD_RUN_{test_no}:{samp_no}",
            #     experiment_id=experiment_id,
            #     nested=True,
            # ) as child_run:
            # mlflow.log_param("child", "yes")
            if samp_no < len(samples):  # can limit num samples considered
                if samp_no > 0:
                    # drop first run as compiling
                    start_time = time.time()
                if verbose:
                    tqdm.write("true eps, eta, rho:", params)
                tqdm.write()
                tqdm.write("$" * 12, "At sample", samp_no + 1, "$" * 12)
                sample.update({"Q": Q, "meta_types": ["categorical"]})
                # present = calc_present(sample["A"])
                # trans_present = calc_trans_present(present)
                # print(present)
                # print(trans_present)+
                ## Initialise model
                true_Z = sample.pop("Z")
                ## Initialise

                model = em.EM(
                    sample,
                    verbose=verbose,
                    n_runs=args.n_runs,
                    sparse_adj=True,
                    max_iter=1,
                    max_msg_iter=args.max_msg_iter,
                    use_numba=args.use_numba,
                    tuning_param=args.tuning_param
                    if args.tuning_param is not None
                    else 1.0,
                )

                if samp_no > 0:
                    init_times[test_no, samp_no - 1] = time.time() - start_time
                ## Score from K means
                tqdm.write("Before fitting model, K-means init partition has")
                if verbose:
                    model.ari_score(true_Z, pred_Z=model.k_means_init_Z)
                else:
                    tqdm.write(
                        np.round_(
                            model.ari_score(true_Z, pred_Z=model.k_means_init_Z),
                            3,
                        )
                    )

                tqdm.write("Freezing params...")
                alpha = 1 / Q
                p_in = get_p_in(eps)
                p_out = get_p_out(eps)

                beta = simulation.gen_beta_mat(Q, p_in, p_out)
                pi = simulation.gen_trans_mat(eta, Q)

                meta_params = [
                    np.tile(
                        np.expand_dims(simulation.gen_trans_mat(rho, Q), 2), (1, 1, T)
                    )
                ]
                true_params = {
                    "alpha": alpha,
                    "beta": beta,
                    "pi": pi,
                    "meta_params": meta_params,
                }

                model.dsbmm.set_params(true_params, freeze=True)
                ## Fit to given data
                model.fit(learning_rate=0.2)
                ## Score after fit
                test_aris[test_no, samp_no, :] = 0.0  # type: ignore
                tqdm.write("BP Z ARI:")
                test_aris[test_no, samp_no, :] = model.ari_score(true_Z)  # type: ignore
                if not verbose:
                    tqdm.write(np.round_(test_aris[test_no, samp_no, :], 3))  # type: ignore
                tqdm.write("BP max energy Z ARI:")
                max_en_aris[test_no, samp_no, :] = model.ari_score(true_Z, pred_Z=model.max_energy_Z)  # type: ignore
                if not verbose:
                    tqdm.write(np.round_(max_en_aris[test_no, samp_no, :], 3))  # type: ignore
                tqdm.write("Init Z ARI:")
                if verbose:
                    model.ari_score(true_Z, pred_Z=model.k_means_init_Z)
                else:
                    tqdm.write(
                        np.round_(
                            model.ari_score(true_Z, pred_Z=model.k_means_init_Z),
                            3,
                        )
                    )

                # print("Z inferred:", model.bp.model.jit_model.Z)
                if verbose:
                    ## Show transition matrix inferred
                    tqdm.write("Pi inferred:", model.bp.trans_prob)
                    try:
                        tqdm.write("Versus true pi:", params["trans_mat"])
                    except Exception:  # KeyError:
                        tqdm.write(
                            "Versus true pi:",
                            simulation.gen_trans_mat(sample["p_stay"], sample["Q"]),
                        )
                    tqdm.write("True effective pi:", utils.effective_pi(true_Z))
                    if args.use_numba:
                        tqdm.write(
                            "Effective pi from partition inferred:",
                            utils.effective_pi(model.bp.model.jit_model.Z),
                        )
                        tqdm.write(
                            "True effective beta:",
                            utils.effective_beta(
                                model.bp.model.jit_model.A, true_Z
                            ).transpose(2, 0, 1),
                        )
                        tqdm.write(
                            "Pred effective beta:",
                            utils.effective_beta(
                                model.bp.model.jit_model.A,
                                model.bp.model.jit_model.Z,
                            ).transpose(2, 0, 1),
                        )
                    else:
                        tqdm.write(
                            "Effective pi from partition inferred:",
                            utils.effective_pi(model.bp.model.Z),
                        )
                        tqdm.write(
                            "True effective beta:",
                            utils.effective_beta(model.bp.model.A, true_Z).transpose(
                                2, 0, 1
                            ),
                        )
                        tqdm.write(
                            "Pred effective beta:",
                            utils.effective_beta(
                                model.bp.model.A,
                                model.bp.model.Z,
                            ).transpose(2, 0, 1),
                        )
                if samp_no > 0:
                    test_times[test_no, samp_no - 1] = time.time() - start_time
                if args.use_numba:
                    # wait a bit for numba to stop thread errors
                    time.sleep(0.5)
                # save after every sample in case of crash
                tp_str = (
                    "_tp" + str(args.tuning_param)
                    if args.tuning_param is not None
                    else ""
                )

                with open(  # type: ignore
                    results_dir / f"{testset_name}_test_aris{tp_str}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(test_aris, f)
                with open(  # type: ignore
                    results_dir / f"{testset_name}_max_en_aris{tp_str}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(max_en_aris, f)
                with open(  # type: ignore
                    results_dir / f"{testset_name}_test_times{tp_str}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(test_times, f)
                with open(  # type: ignore
                    results_dir / f"{testset_name}_init_times{tp_str}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(init_times, f)

        tqdm.write(f"Finished test {test_no+1} for true params:")
        tqdm.write(params)
        tqdm.write(f"Mean ARIs: {test_aris[test_no].mean(axis=0)}")
        tqdm.write(f"Mean max energy ARIs: {max_en_aris[test_no].mean(axis=0)}")
        # logging_params = {
        #     "test_no": params["test_no"],
        #     "meta_aligned": params.get("meta_aligned", None),
        #     "p_in": params.get("p_in", None),
        #     "p_stay": params.get("p_stay", None),
        # }
        # mlflow.log_params(logging_params)
        # mlflow.log_metric(
        #     key=f"Test {params['test_no']} ARI", value=test_aris[test_no].mean()
        # )
    print()
    print("Mean ARIs inferred for each test:")
    print(test_aris.mean(axis=(1, 2)))  # type: ignore

    print("Mean max energy ARIs inferred for each test:")
    print(max_en_aris.mean(axis=(1, 2)))  # type: ignore

    print("Mean times for each test:")
    print(test_times.mean(axis=1))

    print("Mean init times for each test:")
    print(init_times.mean(axis=1))
