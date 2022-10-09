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
from tqdm import tqdm

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
    help="Number of groups to use in the model. Default is 4, or whatever suitable for specified testset. Irrelevant if specify search over range of Q using min_Q etc.",
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
    "--deg-corr", action="store_true", help="Use degree corrected version of model."
)

parser.add_argument(
    "--max_trials",
    type=int,
    default=None,
    help="Maximum number of full trials to run, only used if min_Q, max_Q specified, in which case will search np.linspace(min_Q,max_Q,max_trials,dtype=int).",
)

parser.add_argument(
    "--ret_best_only",
    action="store_true",
    help="Only return single best overall inferred partition, else return best partition for each Q trialled.",
)

parser.add_argument(
    "--max_iter",
    type=int,
    default=150,
    help="Maximum number of EM iterations to run. Default is 150.",
)

parser.add_argument(
    "--patience",
    type=int,
    default=None,
    help="Number of EM iterations to tolerate without reduction in energy before giving up. Default is None, i.e. complete all max iterations.",
)

parser.add_argument(
    "--max_msg_iter",
    type=int,
    default=100,
    help="Maximum number of message updates to run each EM iteration. Default is 100. Early iterations will use //3 of this value.",
)

parser.add_argument(
    "--learning_rate",
    "-lr",
    type=float,
    default=0.2,
    help="Learning rate for updating params at each EM iter. Default is 0.2.",
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
    "--n_runs",
    type=int,
    default=5,
    help="Set number of runs for each value of Q, default = 5",
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

parser.add_argument(
    "--alpha_use_first",
    action="store_true",
    help="Use only first and previously missing node marginals to update alpha, rather than all marginals.",
)

parser.add_argument(
    "--partial_informative_dsbmm_init",
    "-pidsbmm",
    action="store_true",
    help="Use initial partition to partially plant DSBMM params, otherwise use minimally informative initialisation.",
)

parser.add_argument(
    "--planted_p",
    type=float,
    default=0.6,
    help="Weighting of parameter initialisation from initial partition vs non-informative initialisation, default is 0.6.",
)

parser.add_argument(
    "--h_l", type=int, default=None, help="Max number of layers in hierarchy"
)

parser.add_argument(
    "--h_Q",
    type=int,
    default=8,
    help="Max number of groups to look for at each layer in the hierarchy, NB if h_Q>N/h_min_N then take h_Q = N//h_min_N instead",
)

parser.add_argument(
    "--h_min_N",
    type=int,
    default=20,
    help="Minimum number of nodes in group to be considered for recursion.",
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
        try:
            assert len(net_files) > 0
        except AssertionError:
            raise FileNotFoundError(
                "No network files of required form found in specified data directory."
            )
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
            net.add_nodes_from(
                missing_nodes,
                **{
                    mn: np.nan * np.ones(meta_dims[meta_idx])
                    if meta_dims[meta_idx] > 1
                    else np.nan
                    for meta_idx, mn in enumerate(meta_names)
                },
            )
        # get metadata
        metas = [[nx.get_node_attributes(net, mn) for net in nets] for mn in meta_names]
        X = [
            np.stack(
                [
                    np.stack(
                        [
                            metas[meta_idx][net_idx].get(
                                node, np.nan * np.ones(meta_dims[meta_idx])
                            )
                            if meta_dims[meta_idx] > 1
                            else np.array([metas[meta_idx][net_idx].get(node, np.nan)])
                            for net_idx in range(len(nets))
                        ],
                        axis=0,
                    )
                    for node in node_order
                ],
                axis=0,
            )
            for meta_idx, mn in enumerate(meta_names)
        ]
        for s, meta_type in enumerate(meta_types):
            # remove null dimensions
            null_dims = np.nansum(X[s], axis=(0, 1)) == 0
            if np.count_nonzero(null_dims) > 0:
                warnings.warn(
                    f"The following empty dimensions were found for metadata {meta_names[s]}: {np.flatnonzero(null_dims)}. Removing these dimensions."
                )
                X[s] = X[s][:, :, ~null_dims]
                meta_dims[s] -= np.count_nonzero(null_dims)
            # now convert suitably according to specified distribution
            L = X[s].shape[-1]
            missing_meta = np.isnan(X[s])
            if meta_type == "indep bernoulli":
                # restrict to a maximum of 10 dims 'present' for each node, else in high cardinality case likely equally weighting important and noisy meta
                if L > 10:
                    tmpx = np.zeros_like(X[s])
                    k = 10
                    topkidxs = np.argsort(
                        X[s], axis=-1
                    )  # will place nans at end, but should be OK as should only have either whole row nan or nothing
                    np.put_along_axis(tmpx, topkidxs[..., -k:], 1, axis=-1)
                    tmpx[X[s] == 0] = 0
                    tmpx[missing_meta] = np.nan
                    X[s] = tmpx
                else:
                    X[s] = (X[s] > 0) * 1.0
                    X[s][missing_meta] = np.nan
            elif meta_type == "categorical":
                tmpx = np.zeros_like(X[s])
                k = 1
                topkidxs = np.argsort(X[s], axis=-1)
                np.put_along_axis(tmpx, topkidxs[..., -k:], 1, axis=-1)
                tmpx[X[s] == 0] = 0
                tmpx[missing_meta] = np.nan
                X[s] = tmpx
            elif meta_type == "multinomial":
                # first convert to a form of count dist
                int_prop_thr = 0.7  # if proportion of integer values is above this, assume integer count dist
                if np.nanmean(np.mod(X[s][X[s] > 0], 1) == 0) > int_prop_thr:
                    X[s] = np.round(
                        X[s]
                    )  # NB can't just cast to int else nans cause problems
                else:
                    # assume need to convert to something similar - as can be floats, will just enforce some precision
                    n_tot = 1000
                    tmpx = np.round(
                        (
                            X[s]
                            - np.nanmin(X[s], axis=-1, keepdims=True, where=X[s] != 0.0)
                        )
                        / (
                            np.nanmax(X[s], axis=-1, keepdims=True)
                            - np.nanmin(X[s], axis=-1, keepdims=True, where=X[s] != 0.0)
                        )
                        + 1 * n_tot
                    )
                    tmpx[X[s] == 0.0] = 0.0
                    tmpx[missing_meta] = np.nan
                    X[s] = tmpx

            elif meta_type == "poisson":
                int_prop_thr = 0.7  # if proportion of integer values is above this, assume integer count dist
                if np.nanmean(np.mod(X[s][X[s] != 0], 1) == 0) > int_prop_thr:
                    X[s] = np.round(X[s] - np.nanmin(X[s], keepdims=True))
                else:
                    warnings.warn(
                        "Poisson dist being used for non-integer values - no error thrown, but possible problem in dataset."
                    )
                    X[s] = np.round(X[s] - np.nanmin(X[s], keepdims=True))

        # print([x.shape for x in X])
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
        for test_no, (samples, params) in enumerate(
            zip(tqdm(all_samples, desc="Test no."), params_set)
        ):
            # if test_no < 5:
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
            for samp_no, sample in enumerate(
                tqdm(samples, desc="Sample no.", leave=False)
            ):
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
                    tqdm.write("")
                    # tqdm.write("$" * 12, "At sample", samp_no + 1, "$" * 12)
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
                            patience=args.patience,
                            max_iter=args.max_iter,
                            max_msg_iter=args.max_msg_iter,
                            use_numba=args.use_numba,
                            tuning_param=args.tuning_param
                            if args.tuning_param is not None
                            else 1.0,
                            alpha_use_all=not args.alpha_use_first,
                            non_informative_init=not args.partial_informative_dsbmm_init,
                            planted_p=args.planted_p,
                        )
                    elif testset_name == "align":
                        tqdm.write(f"alignment = {params['meta_aligned']}")
                        model = em.EM(
                            sample,
                            verbose=verbose,
                            n_runs=args.n_runs,
                            patience=args.patience,
                            max_iter=args.max_iter,
                            max_msg_iter=args.max_msg_iter,
                            use_numba=args.use_numba,
                            tuning_param=args.tuning_param,
                            alpha_use_all=not args.alpha_use_first,
                            non_informative_init=not args.partial_informative_dsbmm_init,
                            planted_p=args.planted_p,
                        )
                    else:
                        # scaling tests
                        tqdm.write(f"N = {params['N']}")
                        model = em.EM(
                            sample,
                            sparse_adj=True,
                            try_parallel=try_parallel,
                            verbose=verbose,
                            n_runs=args.n_runs,
                            patience=args.patience,
                            max_iter=args.max_iter,
                            max_msg_iter=args.max_msg_iter,
                            use_numba=args.use_numba,
                            tuning_param=args.tuning_param
                            if args.tuning_param is not None
                            else 1.0,
                            alpha_use_all=not args.alpha_use_first,
                            non_informative_init=not args.partial_informative_dsbmm_init,
                            planted_p=args.planted_p,
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
                    if args.freeze:
                        if testset_name == "og":
                            raise NotImplementedError(
                                "Freeze not implemented for OG testset"
                            )
                        else:
                            # "n_samps": test_params["n_samps"],
                            tqdm.write("Freezing params...")
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
                    model.fit(learning_rate=args.learning_rate)
                    ## Score after fit
                    try:
                        test_aris[test_no, samp_no, :] = 0.0  # type: ignore
                        tqdm.write("BP Z ARI:")
                        test_aris[test_no, samp_no, :] = model.ari_score(true_Z)  # type: ignore
                        if not verbose:
                            tqdm.write(f"{np.round_(test_aris[test_no, samp_no, :], 3)}")  # type: ignore
                        tqdm.write("BP max energy Z ARI:")
                        max_en_aris = model.ari_score(true_Z, pred_Z=model.max_energy_Z)  # type: ignore
                        if not verbose:
                            tqdm.write(f"{np.round_(max_en_aris, 3)}")  # type: ignore
                        tqdm.write("Init Z ARI:")
                        if verbose:
                            model.ari_score(true_Z, pred_Z=model.k_means_init_Z)
                        else:
                            tqdm.write(
                                f"{np.round_(model.ari_score(true_Z, pred_Z=model.k_means_init_Z),3)}"
                            )
                    except Exception:  # IndexError:
                        tqdm.write("BP Z ARI:")
                        test_aris[test_no][samp_no, :] = model.ari_score(true_Z)
                        if not verbose:
                            tqdm.write(
                                f"{np.round_(test_aris[test_no][samp_no, :], 3)}"
                            )
                        tqdm.write("BP max energy Z ARI:")
                        max_en_aris = model.ari_score(true_Z, pred_Z=model.max_energy_Z)  # type: ignore
                        if not verbose:
                            tqdm.write(f"{np.round_(max_en_aris, 3)}")
                        tqdm.write("Init Z ARI:")
                        if verbose:
                            model.ari_score(true_Z, pred_Z=model.k_means_init_Z)
                        else:
                            tqdm.write(
                                f"{np.round_(model.ari_score(true_Z, pred_Z=model.k_means_init_Z),3)}"
                            )
                    # print("Z inferred:", model.bp.model.jit_model.Z)
                    if verbose:
                        ## Show transition matrix inferred
                        tqdm.write(f"Pi inferred: {model.bp.trans_prob}")
                        try:
                            tqdm.write(f"Versus true pi: {params['trans_mat']}")
                        except Exception:  # KeyError:
                            tqdm.write(
                                f"Versus true pi: {simulation.gen_trans_mat(sample['p_stay'], sample['Q'])}"
                            )
                        print(f"True effective pi: {utils.effective_pi(true_Z)}")
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
                            tqdm.write(
                                f"Effective pi from partition inferred: {utils.effective_pi(model.bp.model.Z)}"
                            )
                            tqdm.write(
                                f"True effective beta: {utils.effective_beta(model.bp.model.A, true_Z).transpose(2, 0, 1)}"
                            )
                            tqdm.write(
                                f"Pred effective beta: {utils.effective_beta(model.bp.model.A,model.bp.model.Z,).transpose(2, 0, 1)}"
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

            tqdm.write(f"Finished test {test_no+1} for true params:")
            tqdm.write(f"{params}")
            tqdm.write(f"Mean ARIs: {test_aris[test_no].mean(axis=0)}")
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
        if args.ret_best_only:
            if args.h_l is None:
                pred_Z = np.zeros((N, T))
            else:
                pred_Z = np.zeros((args.h_l, N, T))
        else:
            if args.h_l is None:
                pred_Z = np.zeros((args.max_trials, N, T))
            else:
                pred_Z = np.zeros((args.n_runs, args.h_l, N, T))
        # args.h_l, default=None, max. no. layers in hier
        # args.h_Q, default=8, max. no. groups at hier layer,
        # = 4 if h_Q > N_l / 4
        # args.h_min_N, default=20, min. nodes for split

        if (
            args.min_Q is not None
            or args.max_Q is not None
            or args.max_trials is not None
        ):
            try:
                assert args.h_l is None
            except AssertionError:
                raise NotImplementedError(
                    "Hierarchical search over range of Q at each level currently not supported"
                )
            try:
                assert args.min_Q is not None
                assert args.max_Q is not None
                assert args.max_trials is not None
            except AssertionError:
                raise ValueError(
                    "If specifying search for Q, must specify all of --min_Q, --max_Q, and --max_trials"
                )
            trial_Qs = np.linspace(args.min_Q, args.max_Q, args.max_trials, dtype=int)
        else:
            if args.h_l is None:
                trial_Qs = [args.num_groups]
            else:
                trial_Qs = [
                    args.h_Q if args.h_Q < N / args.h_min_N else N // args.h_min_N
                ] * args.n_runs
                try:
                    assert trial_Qs[0] > 0
                except AssertionError:
                    raise ValueError(
                        "Minimum number of nodes to consider at each level of hierarchy must be less than total number of nodes."
                    )
                args.ret_best_only = args.n_runs == 1
                args.n_runs = 1
                # args.h_min_N
        data["Q"] = trial_Qs[0]

        print("*" * 15, "Running empirical data", "*" * 15)
        ## Initialise
        if args.h_l is None:
            hierarchy_layers = [0]
        else:
            hierarchy_layers = np.arange(args.h_l)
        for layer in hierarchy_layers:
            if args.h_l is not None:
                print(f"{'%'*15} At hierarchy layer {layer+1}/{args.h_l} {'%'*15}")

            if layer == 0:
                model = em.EM(
                    data,
                    sparse_adj=True,
                    try_parallel=try_parallel,
                    tuning_param=args.tuning_param,
                    n_runs=args.n_runs,
                    patience=args.patience,
                    deg_corr=args.deg_corr,
                    verbose=verbose,
                    max_iter=args.max_iter,
                    max_msg_iter=args.max_msg_iter,
                    use_meta=not args.ignore_meta,
                    use_numba=args.use_numba,
                    trial_Qs=trial_Qs,
                    alpha_use_all=not args.alpha_use_first,
                    non_informative_init=not args.partial_informative_dsbmm_init,
                    planted_p=args.planted_p,
                )
                try:
                    ## Fit to given data
                    model.fit(learning_rate=args.learning_rate)
                except KeyboardInterrupt:
                    print("Keyboard interrupt, stopping early")
                    current_energy = model.bp.compute_free_energy()
                    if model.best_val_q == 0.0:
                        model.max_energy = current_energy
                        # first iter, first run
                        model.best_val_q = current_energy
                        model.best_val = current_energy
                        model.poor_iter_ctr = 0
                        model.bp.model.set_Z_by_MAP()
                        model.best_Z = model.bp.model.Z.copy()
                        model.best_tun_param = model.dsbmm.tuning_param
                        model.max_energy_Z = model.bp.model.Z.copy()
                    elif current_energy < model.best_val_q:
                        # new best for q
                        model.poor_iter_ctr = 0
                        model.best_val_q = current_energy
                        model.bp.model.set_Z_by_MAP()
                        model.all_best_Zs[model.q_idx, :, :] = model.bp.model.Z.copy()
                        model.best_tun_pars[model.q_idx] = model.dsbmm.tuning_param
                        if model.best_val_q < model.best_val:
                            model.best_val = model.best_val_q
                            model.best_Z = model.bp.model.Z.copy()
                            model.best_tun_param = model.dsbmm.tuning_param
                if args.ret_best_only:
                    if args.h_l is None:
                        pred_Z = model.best_Z
                    else:
                        pred_Z[layer, :, :] = model.best_Z
                else:
                    if args.h_l is None:
                        pred_Z = model.all_best_Zs
                    else:
                        pred_Z[:, layer, :, :] = model.all_best_Zs
                if args.ret_best_only:
                    print(f"Best tuning param: {model.best_tun_param}")
                else:
                    print(f"Best tuning params for each Q:")
                    print(
                        *[
                            f"Q = {q}:  {tunpar}"
                            for q, tunpar in zip(trial_Qs, model.best_tun_pars)
                        ],
                        sep="\n",
                    )
            else:
                if len(pred_Z.shape) == 3:
                    # only one run
                    pred_Z = [pred_Z]
                for run_Z in pred_Z:
                    old_Z = run_Z[layer - 1, :, :]
                    qs = np.unique(old_Z[old_Z != -1])
                    node_group_cnts = np.stack(
                        [(old_Z == q).sum(axis=1) for q in qs], axis=0
                    )
                    old_node_labels = np.argmax(
                        node_group_cnts, axis=0
                    )  # assigns each node its most common label
                    q_idxs, group_cnts = np.unique(old_node_labels, return_counts=True)
                    suff_large_q_idxs = q_idxs[group_cnts > args.h_min_N]
                    for no_q, q in enumerate(suff_large_q_idxs):
                        print(
                            f"\t At group {no_q+1}/{len(suff_large_q_idxs)} in level {layer}:"
                        )
                        N = group_cnts[q]
                        sub_data = {
                            "A": [
                                data["A"][t][
                                    np.ix_(old_node_labels == q, old_node_labels == q)
                                ]
                                for t in range(T)
                            ],
                            "X": [
                                data["X"][s][old_node_labels == q, :, :]
                                for s in range(len(data["X"]))
                            ],
                            "Q": args.h_Q
                            if args.h_Q < N / args.h_min_N
                            else max(N // args.h_min_N, 2),
                            "meta_types": data["meta_types"],
                        }
                        model = em.EM(
                            sub_data,
                            sparse_adj=True,
                            try_parallel=try_parallel,
                            tuning_param=args.tuning_param,
                            n_runs=args.n_runs,
                            patience=args.patience,
                            deg_corr=args.deg_corr,
                            verbose=verbose,
                            max_iter=args.max_iter,
                            max_msg_iter=args.max_msg_iter,
                            use_meta=not args.ignore_meta,
                            use_numba=args.use_numba,
                            alpha_use_all=not args.alpha_use_first,
                            non_informative_init=not args.partial_informative_dsbmm_init,
                            planted_p=args.planted_p,
                        )
                        try:
                            ## Fit to given data
                            model.fit(learning_rate=args.learning_rate)
                        except KeyboardInterrupt:
                            print("Keyboard interrupt, stopping early")
                            current_energy = model.bp.compute_free_energy()
                            if model.best_val_q == 0.0:
                                model.max_energy = current_energy
                                # first iter, first run
                                model.best_val_q = current_energy
                                model.best_val = current_energy
                                model.poor_iter_ctr = 0
                                model.bp.model.set_Z_by_MAP()
                                model.best_Z = model.bp.model.Z.copy()
                                model.best_tun_param = model.dsbmm.tuning_param
                                model.max_energy_Z = model.bp.model.Z.copy()
                            elif current_energy < model.best_val_q:
                                # new best for q
                                model.poor_iter_ctr = 0
                                model.best_val_q = current_energy
                                model.bp.model.set_Z_by_MAP()
                                model.all_best_Zs[
                                    model.q_idx, :, :
                                ] = model.bp.model.Z.copy()
                                model.best_tun_pars[
                                    model.q_idx
                                ] = model.dsbmm.tuning_param
                                if model.best_val_q < model.best_val:
                                    model.best_val = model.best_val_q
                                    model.best_Z = model.bp.model.Z.copy()
                                    model.best_tun_param = model.dsbmm.tuning_param

                        print(f"Best tuning param: {model.best_tun_param}")
                        tmp_Z = model.best_Z
                        missing_nodes = tmp_Z == -1
                        tmp_Z += (
                            args.h_Q * layer * len(suff_large_q_idxs) + args.h_Q * no_q
                        )  # shift labels to avoid overlap
                        tmp_Z[missing_nodes] = -1
                        run_Z[layer, old_node_labels == q, :] = tmp_Z
            RESULTS_DIR = DATA_DIR / "results"
            RESULTS_DIR.mkdir(exist_ok=True)
            if testset_name == "scopus":
                with open(
                    RESULTS_DIR / f"{testset_name}_{link_choice}_Z.pkl", "wb"
                ) as f:
                    pickle.dump(pred_Z, f)
            else:
                with open(RESULTS_DIR / f"{testset_name}_Z.pkl", "wb") as f:
                    pickle.dump(pred_Z, f)

    # TODO: clean up code and improve documentation, then
    # run poetry publish
