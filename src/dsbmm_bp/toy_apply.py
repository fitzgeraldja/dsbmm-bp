import argparse
import os
import pickle
import time
import warnings
from itertools import permutations
from pathlib import Path

import csr
import em
import mlflow
import numpy as np
from mlflow import log_artifacts, log_metric, log_param
from numba.typed import List
from scipy import sparse
from tqdm import tqdm

from dsbmm_bp import simulation, utils
from dsbmm_bp.utils import max_overlap_over_perms

parser = argparse.ArgumentParser(
    description="Apply model to data simulated from toy model."
)

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

parser.add_argument(
    "--msg_conv_tol",
    type=float,
    default=1e-7,
    help="Convergence criterion for messages",
)

parser.add_argument(
    "--msg_init_mode",
    type=str,
    choices=["planted", "random", "uniform", "planted_meta"],
    default="planted",
    help="Initialization mode for messages. Default is planted.",
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
    "--n_runs", type=int, default=1, help="Set number of initial runs, default is 1"
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

parser.add_argument(
    "--name_ext", default="", type=str, help="Extension to add to file names"
)

parser.add_argument(
    "--params_path",
    default=None,
    type=str,
    help="Path to set of parameters to test. Default is None (use default params).",
)

parser.add_argument(
    "--unfreeze",
    action="store_true",
    help="Allow parameter updating",
)

parser.add_argument(
    "--unfreeze_meta",
    action="store_true",
    help="Allow parameter updating for metadata only (to circumvent permutation fixing problem).",
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
    N = 512
    T = 40
    Q = 2
    c = 16
    n_samps = 20
    if args.params_path is None:
        eps_grid = np.linspace(0.3, 0.6, 10)
        eta_grid = np.linspace(0.4, 0.8, 10)
        rho_grid = np.linspace(0.6, 1.0, 10)
        mesh_grid = None
    else:
        with open(args.params_path, "rb") as f:
            mesh_grid = pickle.load(f)
        eps_grid = None
        eta_grid = None
        rho_grid = None

    def get_p_in(eps):
        return Q * c / N * (1 + (Q - 1) * eps)

    def get_p_out(eps):
        return eps * Q * c / N * (1 + (Q - 1) * eps)

    def get_p_stay(eta):
        return eta + (1 - eta) / Q

    data_path = Path("/scratch/fitzgeraldj/data") / "toy_sims"
    data_path.mkdir(exist_ok=True)
    results_dir = data_path.parent / "toy_results"
    results_dir.mkdir(exist_ok=True)

    try:
        if args.params_path is not None:
            params_ext = "_" + str(Path(args.params_path).stem).split("_")[-1]
        else:
            params_ext = args.name_ext
        with open(data_path / f"toy_samples{params_ext}.pkl", "rb") as f:
            all_samples = pickle.load(f)
        with open(data_path / f"toy_param_grid{params_ext}.pkl", "rb") as f:
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
            mesh_grid=mesh_grid,
        )
        with open(data_path / f"toy_samples{params_ext}.pkl", "wb") as f:  # type: ignore
            pickle.dump(all_samples, f)
        with open(data_path / f"toy_param_grid{params_ext}.pkl", "wb") as f:  # type: ignore
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
    max_ols = np.zeros((n_tests, n_samps))

    testset_name = "toy"
    for test_no, (samples, params) in enumerate(zip(tqdm(all_samples), param_grid)):
        eps, eta, rho = params
        tqdm.write("")
        tqdm.write(f"{'*' * 15} Test {test_no+1} {'*' * 15}")
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
                tqdm.write(f"true eps, eta, rho: {params}")
                tqdm.write("")
                tqdm.write(f"{'$' * 12} At sample {samp_no + 1} {'$' * 12}")
                sample.update({"Q": Q, "meta_types": ["categorical"]})

                true_Z = sample.pop("Z")
                sample["X"] = [sample["X"]["categorical"].transpose(1, 2, 0)]
                if rho == 1.0:
                    try:
                        assert np.all(true_Z == np.argmax(sample["X"][0], axis=-1))
                        # print("Meta for rho=1.0 showing true labels correctly.")
                    except AssertionError:
                        print(sample["X"][0].shape)
                        print(true_Z.shape)
                        raise ValueError("Problem w metadata for rho=1.0")

                ## Initialise

                model = em.EM(
                    sample,
                    verbose=verbose,
                    msg_init_mode=args.msg_init_mode,
                    n_runs=args.n_runs,
                    sparse_adj=True,
                    max_iter=1 if not args.unfreeze or not args.unfreeze_meta else 50,
                    patience=100,
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
                        f"{np.round_(model.ari_score(true_Z, pred_Z=model.k_means_init_Z),3)}"
                    )

                tqdm.write("Freezing params...")
                alpha = (1 / Q) * np.ones(Q)
                p_in = get_p_in(eps)
                p_out = get_p_out(eps)
                p_stay = get_p_stay(eta)

                beta = np.tile(
                    simulation.gen_beta_mat(Q, p_in, p_out)[:, :, np.newaxis], (1, 1, T)
                )
                pi = simulation.gen_trans_mat(p_stay, Q)

                meta_params = [
                    np.tile(
                        simulation.gen_trans_mat(rho, Q)[:, np.newaxis, :], (1, T, 1)
                    )
                ]
                true_params = {
                    "alpha": alpha,
                    "beta": beta,
                    "pi": pi,
                    "meta_params": meta_params,
                }

                if args.unfreeze_meta:
                    model.bp.model.unfreeze_meta()
                model.set_params(true_params, freeze=not args.unfreeze)
                if args.unfreeze:
                    model.bp.model.frozen = False
                ## Fit to given data
                model.fit(learning_rate=0.2, msg_conv_tol=args.msg_conv_tol)
                ## Score after fit
                test_aris[test_no, samp_no, :] = 0.0  # type: ignore
                tqdm.write("BP Z ARI:")
                test_aris[test_no, samp_no, :] = model.ari_score(true_Z)  # type: ignore
                if not verbose:
                    tqdm.write(f"{np.round_(test_aris[test_no, samp_no, :], 3)}")  # type: ignore
                tqdm.write("BP max energy Z ARI:")
                max_en_aris[test_no, samp_no, :] = model.ari_score(true_Z, pred_Z=model.max_energy_Z)  # type: ignore
                if not verbose:
                    tqdm.write(f"{np.round_(max_en_aris[test_no, samp_no, :], 3)}")  # type: ignore
                tqdm.write("Init Z ARI:")
                if verbose:
                    model.ari_score(true_Z, pred_Z=model.k_means_init_Z)
                else:
                    tqdm.write(
                        f"{np.round_(model.ari_score(true_Z, pred_Z=model.k_means_init_Z),3)}"
                    )
                ol = max_overlap_over_perms(true_Z, model.best_Z)
                max_ols[test_no, samp_no] = ol
                tqdm.write("Max overlap:")
                tqdm.write(f"{ol:.3f}")

                # print("Z inferred:", model.bp.model.jit_model.Z)
                if verbose:
                    ## Show transition matrix inferred
                    tqdm.write(f"Pi inferred: {model.bp.trans_prob}")
                    tqdm.write(f"Versus true pi: {true_params['pi']}")

                    tqdm.write(f"True effective pi: {utils.effective_pi(true_Z)}")
                    if args.use_numba:
                        tqdm.write(
                            f"Effective pi from partition inferred: {utils.effective_pi(model.bp.model.jit_model.Z)}"
                        )
                        tqdm.write(
                            f"True effective beta: {utils.effective_beta(model.bp.model.jit_model.A, true_Z).transpose(2, 0, 1)}"
                        )
                        tqdm.write(
                            f"Pred effective beta: {utils.effective_beta(model.bp.model.jit_model.A,model.bp.model.jit_model.Z).transpose(2, 0, 1)}"
                        )
                    else:
                        tqdm.write(
                            f"Effective pi from partition inferred: {utils.effective_pi(model.bp.model.Z)}"
                        )
                        tqdm.write(
                            f"True effective beta: {utils.effective_beta(model.bp.model.A, true_Z).transpose(2, 0, 1)}"
                        )
                        tqdm.write(
                            f"Pred effective beta: {utils.effective_beta(model.bp.model.A,model.bp.model.Z).transpose(2, 0, 1)}"
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
                    results_dir
                    / f"{testset_name}_test_aris{tp_str}{args.name_ext}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(test_aris, f)
                with open(  # type: ignore
                    results_dir
                    / f"{testset_name}_max_en_aris{tp_str}{args.name_ext}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(max_en_aris, f)
                with open(  # type: ignore
                    results_dir
                    / f"{testset_name}_test_times{tp_str}{args.name_ext}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(test_times, f)
                with open(  # type: ignore
                    results_dir
                    / f"{testset_name}_init_times{tp_str}{args.name_ext}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(init_times, f)
                with open(  # type: ignore
                    results_dir / f"{testset_name}_max_ols{tp_str}{args.name_ext}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(max_ols, f)

        tqdm.write(f"Finished test {test_no+1} for true params:")
        tqdm.write(f"{params}")
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
