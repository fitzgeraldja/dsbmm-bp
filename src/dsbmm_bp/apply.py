import argparse
import os
import pickle
import time
import warnings
from functools import reduce
from pathlib import Path

import csr
import em
import mlflow
import networkx as nx
import numpy as np
import simulation
import utils
from mlflow import log_artifacts, log_metric, log_param
from numba.typed import List
from scipy import sparse

parser = argparse.ArgumentParser(description="Apply model to data.")


testset_names = ["og", "default", "scaling", "align", "scopus", "empirical"]
parser.add_argument(
    "--test",
    type=str,
    default="default",
    help=f"Run chosen set of provided tests, options are {testset_names}, default is default.",
    choices=testset_names,
)
parser.add_argument(
    "--scopus_link_choice",
    type=str,
    default="ref",
    help=f"Choice of type of link for Scopus data, options are 'ref' or 'au', default is ref.",
    choices=["ref", "au"],
)
parser.add_argument(
    "--data",
    type=str,
    default="./data",
    help="Specify path to data directory. Default is ./data",
)

parser.add_argument(
    "--num_groups",
    type=int,
    default=4,
    help="Number of groups to use in the model. Default is 4, or whatever suitable for specified testset.",
    # NB for scopus, MFVI used 22 if link_choice == "au"
    # else 19 if link_choice == "ref"
)

parser.add_argument(
    "--min_Q",
    type=int,
    default=None,
    help="Minimum number of groups to use, will search from here",
)
parser.add_argument(
    "--max_Q",
    type=int,
    default=None,
    help="Maximum number of groups to use, will search up to here",
)

parser.add_argument(
    "--max_iter",
    type=int,
    default=150,
    help="Maximum number of EM iterations to run. Default is 50.",
)

parser.add_argument(
    "--max_msg_iter",
    type=int,
    default=10,
    help="Maximum number of message updates to run each EM iteration. Default is 10. Early iterations will use //3 of this value.",
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
    help="Try with numba parallelism",
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
    help="Set metadata tuning parameter value",
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

parser.add_argument(
    "--edge-weight",
    type=str,
    default=None,
    help="Name of edge attribute to use as weight when constructing adjacency matrices, defaults to None (binary edges). Only important in DC model.",
)

args = parser.parse_args()

if args.n_threads is not None:
    from numba import set_num_threads

    set_num_threads(args.n_threads)

if __name__ == "__main__":
    if not os.path.exists("../../results/mlruns"):
        os.mkdir("../../results/mlruns")
    dir_path = os.path.abspath("../../results/mlruns")
    # mlflow.set_tracking_uri(f"file://{dir_path}")
    # Set an experiment name, which must be unique
    # and case-sensitive.
    log_runs = False
    # if log_runs:
    #     try:
    #         experiment_id = mlflow.create_experiment(f"dsbmm-alpha-{args.test}")
    #         experiment = mlflow.get_experiment(experiment_id)
    #     except:
    #         experiment = mlflow.set_experiment(f"dsbmm-alpha-{args.test}")
    # Get Experiment Details
    # print("Experiment_id: {}".format(experiment.experiment_id))

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
            if not args.use_numba and testset_name != "scaling":
                for i, sample in enumerate(samples):
                    print(f"...converting sample {i+1} to sparse format")
                    sample["A"] = [
                        sparse.csr_matrix(sample["A"][:, :, t])
                        for t in range(params["T"])
                    ]
                print("...done")
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
        link_choice = args.scopus_link_choice
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
        data["Q"] = args.num_groups

    elif testset_name == "empirical":
        DATA_DIR = Path(args.data)
        data = {}
        data["Q"] = args.num_groups
        print(f"Loading data from {DATA_DIR}:")
        print(
            "\tNB data must be pickled networkx graph in form 'net_[timestamp: int].pkl', with all metadata as node attributes."
        )
        print(
            "\tFurther, there must be an additional 'meta_dists.pkl' file containing a dict of the form {meta_name : (dist_name, dim)},"
        )
        print(
            "\twhere dist_name is one of the accepted metadata distribution types, and dim is the corresponding dimension."
        )
        net_files = list(DATA_DIR.glob("net_*.pkl"))
        print(
            f"Found {len(net_files)} network files, for timestamps {sorted(list(map(lambda x: int(x.stem.split('_')[1]), net_files)))}"
        )
        net_files = sorted(
            net_files, key=lambda x: int(x.stem.split("_")[-1])
        )  # sort by timestamp
        meta_file = DATA_DIR / "meta_dists.pkl"
        nets = []
        for nf in net_files:
            with open(nf, "rb") as f:
                net = pickle.load(f)
            nets.append(net)
        with open(meta_file, "rb") as f:
            meta_info = pickle.load(f)
        meta_names = list(meta_info.keys())
        meta_types = [meta_info[mn][0] for mn in meta_names]
        data["meta_types"] = meta_types
        meta_dims = [int(meta_info[mn][1]) for mn in meta_names]
        try:
            assert set(next(iter(nets[0].nodes.data(default=np.nan)))[1].keys()) == set(
                meta_names
            )
        except AssertionError:
            warnings.warn(
                "Metadata names in meta_dists.pkl do not match those in the networkx graph. Will only use metadata names in meta_dists.pkl."
            )
        edge_attrs = list(next(iter(nets[0].edges(data=True)))[-1].keys())
        node_order = list(
            reduce(lambda res, x: set(res) | set(x), [list(net.nodes) for net in nets])  # type: ignore
        )
        # add in missing nodes at each timestep so matrices not changing size - won't drastically increase memory reqs as using sparse
        # mats
        for net in nets:
            missing_nodes = set(node_order) - set(net.nodes)
            net.add_nodes_from(missing_nodes, **{mn: np.nan for mn in meta_names})
        # get metadata
        metas = [[nx.get_node_attributes(net, mn) for net in nets] for mn in meta_names]
        X = [
            np.array(
                [
                    [
                        metas[meta_idx][net_idx].get(
                            node, np.nan * np.ones(meta_dims[meta_idx])
                        )
                        for net_idx, net in enumerate(nets)
                    ]
                    for node in node_order
                ]
            )
            for meta_idx, mn in enumerate(meta_names)
        ]
        print([x.shape for x in X])
        data["X"] = X
        # get sparse adj mats
        A = [
            nx.to_scipy_sparse_array(net, nodelist=node_order, weight=args.edge_weight)
            for net in nets
        ]
        data["A"] = A

    try_parallel = args.nb_parallel

    use_X_init = False
    verbose = args.verbose
    if testset_name not in ["scopus", "empirical"]:
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
                    if testset_name not in ["scaling", "align"]:
                        model = em.EM(
                            sample,
                            verbose=verbose,
                            n_runs=args.n_runs,
                            max_iter=args.max_iter,
                            max_msg_iter=args.max_msg_iter,
                            use_numba=args.use_numba,
                            tuning_param=args.tuning_param
                            if args.tuning_param is not None
                            else 1.0,
                        )
                    elif testset_name == "align":
                        print(f"alignment = {params['meta_aligned']}")
                        model = em.EM(
                            sample,
                            verbose=verbose,
                            tuning_param=args.tuning_param,
                            n_runs=args.n_runs,
                            max_iter=args.max_iter,
                            max_msg_iter=args.max_msg_iter,
                            use_numba=args.use_numba,
                        )
                    else:
                        # scaling tests
                        print(f"N = {params['N']}")
                        model = em.EM(
                            sample,
                            sparse_adj=True,
                            try_parallel=try_parallel,
                            verbose=verbose,
                            n_runs=args.n_runs,
                            use_numba=args.use_numba,
                            tuning_param=args.tuning_param
                            if args.tuning_param is not None
                            else 1.0,
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
                    if args.freeze:
                        if testset_name == "og":
                            raise NotImplementedError(
                                "Freeze not implemented for OG testset"
                            )
                        else:
                            # "n_samps": test_params["n_samps"],
                            print("Freezing params...")
                            alpha = 1 / params["Q"]
                            beta = simulation.gen_beta_mat(
                                params["Q"], params["p_in"], params["p_out"]
                            )
                            pi = simulation.gen_trans_mat(params["p_stay"], params["Q"])
                            meta_params = [
                                params["pois_params"],
                                params["indep_bern_params"],
                            ]
                            true_params = {
                                "alpha": alpha,
                                "beta": beta,
                                "pi": pi,
                                "meta_params": meta_params,
                            }

                        model.dsbmm.set_params(true_params, freeze=True)
                    ## Fit to given data
                    model.fit(true_Z=true_Z, learning_rate=0.2)
                    ## Score after fit
                    try:
                        test_aris[test_no, samp_no, :] = 0.0  # type: ignore
                        print("BP Z ARI:")
                        test_aris[test_no, samp_no, :] = model.ari_score(true_Z)  # type: ignore
                        print("BP max energy Z ARI:")
                        model.ari_score(true_Z, pred_Z=model.max_energy_Z)  # type: ignore
                        if not verbose:
                            print(np.round_(test_aris[test_no, samp_no, :], 3))  # type: ignore
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
                        print("BP max energy Z ARI:")
                        max_en_aris = model.ari_score(true_Z, pred_Z=model.max_energy_Z)  # type: ignore
                        if not verbose:
                            print(np.round_(max_en_aris, 3))
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
                        if args.use_numba:
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
                        else:
                            print(
                                "Effective pi from partition inferred:",
                                utils.effective_pi(model.bp.model.Z),
                            )
                            print(
                                "True effective beta:",
                                utils.effective_beta(
                                    model.bp.model.A, true_Z
                                ).transpose(2, 0, 1),
                            )
                            print(
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
                    if testset_name == "align":
                        if args.use_numba:
                            test_Z[test_no, samp_no, :, :] = model.bp.model.jit_model.Z
                            with open(
                                f"../../results/{testset_name}_test_Z{tp_str}.pkl",
                                "wb",
                            ) as f:
                                pickle.dump(test_Z, f)
                        else:
                            test_Z[test_no, samp_no, :, :] = model.bp.model.Z
                            with open(
                                f"../../results/{testset_name}_test_Z{tp_str}.pkl",
                                "wb",
                            ) as f:
                                pickle.dump(test_Z, f)
                    with open(
                        f"../../results/{testset_name}_test_aris{tp_str}.pkl",
                        "wb",
                    ) as f:
                        pickle.dump(test_aris, f)
                    with open(
                        f"../../results/{testset_name}_test_times{tp_str}.pkl",
                        "wb",
                    ) as f:
                        pickle.dump(test_times, f)
                    with open(
                        f"../../results/{testset_name}_init_times{tp_str}.pkl",
                        "wb",
                    ) as f:
                        pickle.dump(init_times, f)
            # TODO: finish MLflow logging
            # Use mlflow.set_tag to mark runs that miss
            # reasonable accuracy

            print(f"Finished test {test_no+1} for true params:")
            print(params)
            print(f"Mean ARIs: {test_aris[test_no].mean(axis=0)}")
            logging_params = {
                "test_no": params["test_no"],
                "meta_aligned": params.get("meta_aligned", None),
                "p_in": params.get("p_in", None),
                "p_stay": params.get("p_stay", None),
            }
            # mlflow.log_params(logging_params)
            # mlflow.log_metric(
            #     key=f"Test {params['test_no']} ARI", value=test_aris[test_no].mean()
            # )
        print()
        print("Mean ARIs inferred for each test:")
        try:
            print(test_aris.mean(axis=(1, 2)))  # type: ignore
        except Exception:  # AttributeError:
            print(np.array([aris.mean() for aris in test_aris]))
        print("Mean times for each test:")
        print(test_times.mean(axis=1))
    else:
        T = len(data["A"])
        N = data["A"][0].shape[0]
        n_runs = 5
        test_Z = np.zeros((n_runs, N, T))
        print("*" * 15, "Running empirical data", "*" * 15)
        ## Initialise
        model = em.EM(
            data,
            sparse_adj=True,
            try_parallel=try_parallel,
            tuning_param=np.linspace(0.8, 1.8, 11),
            n_runs=n_runs,
            deg_corr=True if args.use_numba else False,
            verbose=verbose,
            use_meta=not args.ignore_meta,
            use_numba=args.use_numba,
        )
        ## Fit to given data
        model.fit(learning_rate=0.2)
        pred_Z = model.bp.model.jit_model.Z if args.use_numba else model.bp.model.Z
        print(f"Best tuning param: {model.best_tun_param}")
        if testset_name == "scopus":
            with open(f"../../results/{testset_name}_{link_choice}_Z.pkl", "wb") as f:
                pickle.dump(pred_Z, f)
        else:
            with open(f"../../results/{testset_name}_Z.pkl", "wb") as f:
                pickle.dump(pred_Z, f)

    # TODO: clean up code and improve documentation, then
    # run poetry publish
