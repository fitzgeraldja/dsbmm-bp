import argparse
import os
import pickle
import time
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

parser = argparse.ArgumentParser(description="Apply model to data.")

parser.add_argument(
    "--data",
    type=str,
    default="./data",
    help="Specify path to data directory. Default is ./data",
)

parser.add_argument(
    "--max_iter",
    type=int,
    default=30,
    help="Maximum number of EM iterations to run. Default is 30.",
)

parser.add_argument(
    "--max_msg_iter",
    type=int,
    default=10,
    help="Maximum number of message updates to run each EM iteration. Default is 10. Early iterations will use //3 of this value.",
)

parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose output")

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
    "--freeze",
    action="store_true",
    help="Supply model true DSBMM parameters with exception of group labels, and do not update these",
)

args = parser.parse_args()

if args.n_threads is not None:
    from numba import set_num_threads

    set_num_threads(args.n_threads)

if __name__ == "__main__":
    if not os.path.exists("../../results/mlruns"):
        os.mkdir("../../results/mlruns")
    dir_path = os.path.abspath("../../results/mlruns")
    mlflow.set_tracking_uri(f"file://{dir_path}")
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
    try:
        with open("../../results/toy_samples.pkl", "rb") as f:
            all_samples = pickle.load(f)
        with open("../../results/toy_param_grid.pkl", "rb") as f:
            param_grid = pickle.load(f)
    except FileNotFoundError:
        all_samples, param_grid = simulation.toy_tests()

    for testno, samples in enumerate(all_samples):
        # print(samples)
        try:
            if not np.all(
                [
                    utils.is_connected_dense(sample["A"][:, :, t])
                    for sample in samples
                    for t in range(sample["A"].shape[-1])
                ]
            ):
                print(
                    "WARNING: some matrices not connected in test set for test no. ",
                    testno,
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
                print(
                    "WARNING: some matrices not connected in test set for test no. ",
                    testno,
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

    test_aris = np.zeros(
        (len(all_samples), len(all_samples[0]), all_samples[0]["A"].shape[-1])
    )
    test_times = np.zeros((len(all_samples), len(all_samples[0]) - 1))
    init_times = np.zeros_like(test_times)
    # for test_no, (samples, params) in enumerate(zip(all_samples, params_set)):
    #     # if test_no < 5:
    #     print()
    #     print("*" * 15, f"Test {test_no+1}", "*" * 15)
    #     # Create nested runs for each test + sample
    #     if log_runs:
    #         experiment_id = experiment.experiment_id
    #     else:
    #         experiment_id = 0
    #     with mlflow.start_run(
    #         run_name=f"PARENT_RUN_{test_no}", experiment_id=experiment_id
    #     ) as parent_run:
    #         # mlflow.log_param("parent", "yes")
    #         for samp_no, sample in enumerate(samples):
    #             with mlflow.start_run(
    #                 run_name=f"CHILD_RUN_{test_no}:{samp_no}",
    #                 experiment_id=experiment_id,
    #                 nested=True,
    #             ) as child_run:
    #                 # mlflow.log_param("child", "yes")
    #                 if samp_no < len(samples):  # can limit num samples considered
    #                     if samp_no > 0:
    #                         # drop first run as compiling
    #                         start_time = time.time()
    #                     if verbose:
    #                         print("true params:", params)
    #                     print()
    #                     print("$" * 12, "At sample", samp_no + 1, "$" * 12)
    #                     sample.update(params)
    #                     # present = calc_present(sample["A"])
    #                     # trans_present = calc_trans_present(present)
    #                     # print(present)
    #                     # print(trans_present)+
    #                     ## Initialise model
    #                     true_Z = sample.pop("Z")
    #                     ## Initialise
    #                     if testset_name not in ["scaling", "align"]:
    #                         model = em.EM(
    #                             sample,
    #                             verbose=verbose,
    #                             n_runs=args.n_runs,
    #                             max_iter=args.max_iter,
    #                             max_msg_iter=args.max_msg_iter,
    #                         )
    #                     elif testset_name == "align":
    #                         print(f"alignment = {params['meta_aligned']}")
    #                         model = em.EM(
    #                             sample,
    #                             verbose=verbose,
    #                             tuning_param=args.tuning_param,
    #                             n_runs=args.n_runs,
    #                             max_iter=args.max_iter,
    #                             max_msg_iter=args.max_msg_iter,
    #                         )
    #                     else:
    #                         print(f"N = {params['N']}")
    #                         model = em.EM(
    #                             sample,
    #                             sparse_adj=True,
    #                             try_parallel=try_parallel,
    #                             verbose=verbose,
    #                             n_runs=args.n_runs,
    #                         )
    #                     if samp_no > 0:
    #                         init_times[test_no, samp_no - 1] = (
    #                             time.time() - start_time
    #                         )
    #                     ## Score from K means
    #                     print("Before fitting model, K-means init partition has")
    #                     if verbose:
    #                         model.ari_score(true_Z, pred_Z=model.k_means_init_Z)
    #                     else:
    #                         print(
    #                             np.round_(
    #                                 model.ari_score(
    #                                     true_Z, pred_Z=model.k_means_init_Z
    #                                 ),
    #                                 3,
    #                             )
    #                         )
    #                     if args.freeze:
    #                         if testset_name == "og":
    #                             raise NotImplementedError(
    #                                 "Freeze not implemented for OG testset"
    #                             )
    #                         else:
    #                             # "n_samps": test_params["n_samps"],
    #                             print("Freezing params...")
    #                             alpha = 1 / params["Q"]
    #                             beta = simulation.gen_beta_mat(
    #                                 params["Q"], params["p_in"], params["p_out"]
    #                             )
    #                             pi = simulation.gen_trans_mat(
    #                                 params["p_stay"], params["Q"]
    #                             )
    #                             meta_params = [
    #                                 params["pois_params"],
    #                                 params["indep_bern_params"],
    #                             ]
    #                             true_params = {
    #                                 "alpha": alpha,
    #                                 "beta": beta,
    #                                 "pi": pi,
    #                                 "meta_params": meta_params,
    #                             }

    #                         model.dsbmm.set_params(true_params, freeze=True)
    #                     ## Fit to given data
    #                     model.fit(true_Z=true_Z, learning_rate=0.2)
    #                     ## Score after fit
    #                     try:
    #                         test_aris[test_no, samp_no, :] = 0.0  # type: ignore
    #                         print("BP Z ARI:")
    #                         test_aris[test_no, samp_no, :] = model.ari_score(true_Z)  # type: ignore
    #                         if not verbose:
    #                             print(np.round_(test_aris[test_no, samp_no, :], 3))  # type: ignore
    #                         print("Init Z ARI:")
    #                         if verbose:
    #                             model.ari_score(true_Z, pred_Z=model.k_means_init_Z)
    #                         else:
    #                             print(
    #                                 np.round_(
    #                                     model.ari_score(
    #                                         true_Z, pred_Z=model.k_means_init_Z
    #                                     ),
    #                                     3,
    #                                 )
    #                             )
    #                     except Exception:  # IndexError:
    #                         print("BP Z ARI:")
    #                         test_aris[test_no][samp_no, :] = model.ari_score(true_Z)
    #                         if not verbose:
    #                             print(np.round_(test_aris[test_no][samp_no, :], 3))
    #                         print("Init Z ARI:")
    #                         if verbose:
    #                             model.ari_score(true_Z, pred_Z=model.k_means_init_Z)
    #                         else:
    #                             print(
    #                                 np.round_(
    #                                     model.ari_score(
    #                                         true_Z, pred_Z=model.k_means_init_Z
    #                                     ),
    #                                     3,
    #                                 )
    #                             )
    #                     # print("Z inferred:", model.bp.model.jit_model.Z)
    #                     if verbose:
    #                         ## Show transition matrix inferred
    #                         print("Pi inferred:", model.bp.trans_prob)
    #                         try:
    #                             print("Versus true pi:", params["trans_mat"])
    #                         except Exception:  # KeyError:
    #                             print(
    #                                 "Versus true pi:",
    #                                 simulation.gen_trans_mat(
    #                                     sample["p_stay"], sample["Q"]
    #                                 ),
    #                             )
    #                         print("True effective pi:", utils.effective_pi(true_Z))
    #                         print(
    #                             "Effective pi from partition inferred:",
    #                             utils.effective_pi(model.bp.model.jit_model.Z),
    #                         )
    #                         print(
    #                             "True effective beta:",
    #                             utils.effective_beta(
    #                                 model.bp.model.jit_model.A, true_Z
    #                             ).transpose(2, 0, 1),
    #                         )
    #                         print(
    #                             "Pred effective beta:",
    #                             utils.effective_beta(
    #                                 model.bp.model.jit_model.A,
    #                                 model.bp.model.jit_model.Z,
    #                             ).transpose(2, 0, 1),
    #                         )
    #                     if samp_no > 0:
    #                         test_times[test_no, samp_no - 1] = (
    #                             time.time() - start_time
    #                         )
    #                         if testset_name == "scaling":
    #                             print(
    #                                 f"Sample took ~{test_times[test_no,samp_no-1]:.2f}s"
    #                             )
    #                     time.sleep(
    #                         0.5
    #                     )  # sleep a bit to allow threads to complete, TODO: properly sort this
    #                     # save after every sample in case of crash
    #                     tp_str = (
    #                         "_tp" + str(args.tuning_param)
    #                         if args.tuning_param is not None
    #                         else ""
    #                     )
    #                     if testset_name == "align":
    #                         test_Z[
    #                             test_no, samp_no, :, :
    #                         ] = model.bp.model.jit_model.Z
    #                         with open(
    #                             f"../../results/{testset_name}_test_Z{tp_str}.pkl",
    #                             "wb",
    #                         ) as f:
    #                             pickle.dump(test_Z, f)
    #                     with open(
    #                         f"../../results/{testset_name}_test_aris{tp_str}.pkl",
    #                         "wb",
    #                     ) as f:
    #                         pickle.dump(test_aris, f)
    #                     with open(
    #                         f"../../results/{testset_name}_test_times{tp_str}.pkl",
    #                         "wb",
    #                     ) as f:
    #                         pickle.dump(test_times, f)
    #                     with open(
    #                         f"../../results/{testset_name}_init_times{tp_str}.pkl",
    #                         "wb",
    #                     ) as f:
    #                         pickle.dump(init_times, f)

    #     print(f"Finished test {test_no+1} for true params:")
    #     print(params)
    #     print(f"Mean ARIs: {test_aris[test_no].mean(axis=0)}")
    #     logging_params = {
    #         "test_no": params["test_no"],
    #         "meta_aligned": params["meta_aligned"],
    #         "p_in": params["p_in"],
    #         "p_stay": params["p_stay"],
    #     }
    #     mlflow.log_params(logging_params)
    #     mlflow.log_metric(
    #         key=f"Test {params['test_no']} ARI", value=test_aris[test_no].mean()
    #     )
    # print()
    # print("Mean ARIs inferred for each test:")
    # try:
    #     print(test_aris.mean(axis=(1, 2)))  # type: ignore
    # except Exception:  # AttributeError:
    #     print(np.array([aris.mean() for aris in test_aris]))
    # print("Mean times for each test:")
    # print(test_times.mean(axis=1))
