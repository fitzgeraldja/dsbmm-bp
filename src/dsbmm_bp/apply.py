import argparse
import logging
import os
import pickle
import sys
import time
import warnings
from functools import reduce
from pathlib import Path

import csr
import mlflow
import networkx as nx
import numpy as np
from mlflow import log_artifacts, log_metric, log_param
from numba.typed import List
from scipy import sparse
from tqdm import tqdm

# local package imports
from . import data_processor, em, simulation, utils


def init_pred_Z(N, T, ret_best_only=False, h_l=None, max_trials=None, n_runs=1):
    if ret_best_only:
        if h_l is None:
            pred_Z = np.zeros((N, T))
        else:
            pred_Z = np.zeros((h_l, N, T))
    else:
        if h_l is None:
            pred_Z = np.zeros((max_trials, N, T))
        else:
            pred_Z = np.zeros((n_runs, h_l, N, T))
    return pred_Z


def init_trial_Qs(
    min_Q=None,
    max_Q=None,
    max_trials=None,
    h_l=None,
    num_groups=None,
    h_Q=None,
    h_min_N=None,
    n_runs=1,
):
    if min_Q is not None or max_Q is not None or max_trials is not None:
        try:
            assert h_l is None
        except AssertionError:
            raise NotImplementedError(
                "Hierarchical search over range of Q at each level currently not supported"
            )
        try:
            assert min_Q is not None
            assert max_Q is not None
            assert max_trials is not None
        except AssertionError:
            raise ValueError(
                "If specifying search for Q, must specify all of --min_Q, --max_Q, and --max_trials"
            )
        trial_Qs = np.linspace(min_Q, max_Q, max_trials, dtype=int)
    else:
        if h_l is None:
            trial_Qs = [num_groups]
        else:
            trial_Qs = [h_Q if h_Q < N / h_min_N else N // h_min_N] * n_runs
            try:
                assert trial_Qs[0] > 0
            except AssertionError:
                raise ValueError(
                    "Minimum number of nodes to consider at each level of hierarchy must be less than total number of nodes."
                )
    return trial_Qs


def prepare_for_run(
    data,
    DATA_DIR,
    trial_Qs,
    h_l=None,
):
    data["Q"] = trial_Qs[0]

    tqdm.write("*" * 15, "Running empirical data", "*" * 15)
    ## Initialise
    file_handler = logging.FileHandler(filename="./empirical_dsbmm.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(message)s", handlers=handlers  # type: ignore
    )
    if h_l is None:
        hierarchy_layers = [0]
    else:
        hierarchy_layers = np.arange(h_l)

    RESULTS_DIR = DATA_DIR / "results"
    RESULTS_DIR.mkdir(exist_ok=True)
    return hierarchy_layers, RESULTS_DIR


def run_hier_model(
    testset_name,
    data,
    N,
    T,
    pred_Z,
    trial_Qs,
    hierarchy_layers,
    RESULTS_DIR,
    verbose=False,
    link_choice=None,
    h_l=None,
    h_Q=None,
    h_min_N=None,
    ret_best_only=False,
    tuning_param=1.0,
    learning_rate=0.2,
    n_runs=1,
    patience=3,
    deg_corr=False,
    directed=False,
    max_iter=100,
    max_msg_iter=30,
    ignore_meta=False,
    alpha_use_first=False,
    partial_informative_dsbmm_init=False,
    planted_p=0.7,
    auto_tune=True,
    use_numba=False,
    try_parallel=False,
    ret_Z=False,
    ret_probs=False,
    ret_trans=False,
    save_to_file=True,
    datetime_str=None,
):
    model_settings = dict(
        sparse_adj=True,
        try_parallel=try_parallel,
        tuning_param=tuning_param,
        n_runs=n_runs,
        patience=patience,
        deg_corr=deg_corr,
        leave_directed=directed,
        verbose=verbose,
        max_iter=max_iter,
        max_msg_iter=max_msg_iter,
        use_meta=not ignore_meta,
        use_numba=use_numba,
        trial_Qs=trial_Qs,
        alpha_use_all=not alpha_use_first,
        non_informative_init=not partial_informative_dsbmm_init,
        planted_p=planted_p,
        auto_tune=auto_tune,
        ret_Z=ret_Z,
        ret_probs=ret_probs,
    )
    if ret_probs:
        if h_l is not None:
            if len(pred_Z.shape) == 3:
                node_probs = np.zeros((N, T, np.power(h_Q, h_l, dtype=int)))
            else:
                node_probs = np.zeros(
                    (pred_Z.shape[0], N, T, np.power(h_Q, h_l, dtype=int))
                )
        else:
            # TODO: allow ret probs for non-hier model
            node_probs = [np.zeros(N, T, q) for q in trial_Qs]
    for layer in tqdm(hierarchy_layers, desc="Hier. lvl"):
        if h_l is not None:
            logging.info(f"{'%'*15} At hierarchy level {layer+1}/{h_l} {'%'*15}")
        if layer == 0:
            model = em.EM(data, **model_settings)
            fit_model_allow_interrupt(learning_rate, model)
            if ret_best_only:
                tot_Q = model.Q
                if h_l is None:
                    pred_Z = model.best_Z
                    pi = model.pi
                else:
                    pred_Z[layer, :, :] = model.best_Z
                    if ret_probs:
                        node_probs[:, :, : model.Q] = model.run_probs[0, ...]
                    if ret_trans:
                        pi = model.pi
            else:
                tot_Q = np.repeat([model.Q], n_runs)
                if h_l is None:
                    pred_Z = model.all_best_Zs
                    if ret_trans:
                        pi = model.all_pi
                else:
                    pred_Z[:, layer, :, :] = model.all_best_Zs
                    if ret_probs:
                        node_probs[:, :, :, : model.Q] = model.run_probs
                    if ret_trans:
                        pi_1 = model.all_pi
                        pis = [*pi_1]
                        hier_pis = [[pi] for pi in pis]
            if ret_best_only:
                tqdm.write(f"Best tuning param: {model.best_tun_param}")
            else:
                tqdm.write(f"Best tuning params for each Q:")
                tqdm.write(
                    "\n".join(
                        [
                            f"Q = {q}:  {tunpar}"
                            for q, tunpar in zip(trial_Qs, model.best_tun_pars)
                        ]
                    )
                )
        else:
            if len(pred_Z.shape) == 3:
                # only one run
                pred_Z = np.expand_dims(pred_Z, 0)
            new_Q = tot_Q.copy()
            for run_idx in range(pred_Z.shape[0]):
                old_Z = pred_Z[run_idx, layer - 1, :, :]
                qs = np.unique(old_Z[old_Z != -1])
                node_group_cnts = np.stack(
                    [(old_Z == q).sum(axis=1) for q in qs], axis=0
                )
                old_node_labels = np.argmax(
                    node_group_cnts, axis=0
                )  # assigns each node its most common label
                q_idxs, group_cnts = np.unique(old_node_labels, return_counts=True)
                if np.all(group_cnts <= h_min_N):
                    tqdm.write(
                        f"No remaining groups larger than minimum size specified in run {run_idx+1} at level {layer} -- run complete"
                    )
                    continue
                suff_large_q_idxs = q_idxs[group_cnts > h_min_N]
                n_suff_large = len(suff_large_q_idxs)
                new_Q[run_idx] += n_suff_large * h_Q
                small_q_idxs = q_idxs[group_cnts <= h_min_N]
                n_small = (group_cnts <= h_min_N).sum()
                # mark nodes belonging to small groups as unassigned at this level
                pred_Z[run_idx, layer, np.isin(old_node_labels, small_q_idxs), :] = -1
                if ret_trans:
                    pi_lq = {q_idx: [] for q_idx in suff_large_q_idxs}
                    hier_pis[run_idx].append(pi_lq)

                for no_q, q in enumerate(tqdm(suff_large_q_idxs, desc="q_l")):
                    logging.info(
                        f"\t At group {no_q+1}/{len(suff_large_q_idxs)} in level {layer}:"
                    )
                    N = group_cnts[q]
                    logging.info(f"\t\t Considering {N} nodes...")

                    sub_data = subset_data(data, N, T, h_Q, h_min_N, old_node_labels, q)

                    if np.all([A_t.sum() == 0 for A_t in sub_data["A"]]):
                        logging.info("No edges in subgraph, skipping...")
                        tqdm.write("No edges in subgraph, skipping...")
                        tmp_Z = -1 * np.ones(N, T)
                        pred_Z[run_idx, layer, old_node_labels == q, :] = tmp_Z
                        continue

                    model = em.EM(sub_data, **model_settings)
                    fit_model_allow_interrupt(learning_rate, model)

                    tqdm.write(f"Best tuning param: {model.best_tun_param}")

                    tmp_Z = model.best_Z
                    missing_nodes = tmp_Z == -1
                    q_shift = h_Q * (tot_Q[run_idx] + no_q)
                    tmp_Z += q_shift  # shift labels to avoid overlap
                    tmp_Z[missing_nodes] = -1

                    logging.info(f"Found {len(np.unique(tmp_Z[tmp_Z!=-1]))} groups")
                    pred_Z[run_idx, layer, old_node_labels == q, :] = tmp_Z

                    if ret_probs:
                        node_probs[
                            run_idx,
                            old_node_labels == q,
                            :,
                            q_shift : q_shift + model.Q,
                        ] = model.run_probs[0, ...]
                    if ret_trans:
                        pi_lq[q] = model.pi
                        pis[run_idx] = utils.construct_hier_trans(
                            hier_pis[run_idx],
                            pred_Z[run_idx, : layer + 1, ...],
                            h_min_N,
                        )

                    logging.info(
                        f"Transferred {len(np.unique(pred_Z[run_idx, layer, old_node_labels == q, :][pred_Z[run_idx, layer, old_node_labels == q, :]!=-1]))} groups"
                    )

                    # save after each iteration in case of errors
                    if datetime_str is None:
                        datetime_str = time.strftime(
                            "%d-%m_%H-%M", time.gmtime(time.time())
                        )
                    if save_to_file:
                        save_emp_Z(
                            testset_name,
                            link_choice,
                            RESULTS_DIR,
                            pred_Z,
                            datetime_str,
                            h_l=h_l,
                        )
                        if ret_probs:
                            save_node_probs(
                                testset_name,
                                RESULTS_DIR,
                                node_probs,
                                datetime_str,
                                h_l=h_l,
                            )
                        if ret_trans:
                            save_trans(
                                testset_name,
                                RESULTS_DIR,
                                pis,
                                datetime_str,
                                h_l=h_l,
                            )
                tot_Q[run_idx] = new_Q[run_idx]
        logging.info(f"Run complete for level {layer}, saving...")
        if save_to_file:
            save_emp_Z(
                testset_name, link_choice, RESULTS_DIR, pred_Z, datetime_str, h_l=h_l
            )
            if ret_probs:
                save_node_probs(
                    testset_name, RESULTS_DIR, node_probs, datetime_str, h_l=h_l
                )
            if ret_trans:
                save_trans(
                    testset_name,
                    RESULTS_DIR,
                    pis,
                    datetime_str,
                    h_l=h_l,
                )
    if ret_Z:
        if ret_probs:
            if ret_trans:
                return pred_Z, node_probs, pis
            else:
                return pred_Z, node_probs
        else:
            if ret_trans:
                return pred_Z, pis
            else:
                return pred_Z
    else:
        if ret_probs:
            if ret_trans:
                return node_probs, pis
            else:
                return node_probs
        else:
            if ret_trans:
                return pis


def subset_data(data, N, T, h_Q, h_min_N, old_node_labels, q):
    sub_data = {
        "A": [
            data["A"][t][np.ix_(old_node_labels == q, old_node_labels == q)]
            for t in range(T)
        ],
        "X": [data["X"][s][old_node_labels == q, :, :] for s in range(len(data["X"]))],
        "Q": h_Q if h_Q < N / h_min_N else max(N // h_min_N, 2),
        "meta_types": data["meta_types"],
    }

    return sub_data


def fit_model_allow_interrupt(learning_rate, model):
    try:
        ## Fit to given data
        model.fit(learning_rate=learning_rate)
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt, stopping early")
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


def save_emp_Z(testset_name, link_choice, RESULTS_DIR, pred_Z, datetime_str, h_l=None):
    if testset_name == "scopus":
        with open(RESULTS_DIR / f"{testset_name}_{link_choice}_Z.pkl", "wb") as f:
            pickle.dump(pred_Z, f)
    else:
        with open(
            RESULTS_DIR / f"{testset_name}_Z{'_h' if h_l else ''}_{datetime_str}.pkl",
            "wb",
        ) as f:
            pickle.dump(pred_Z, f)


def save_node_probs(testset_name, RESULTS_DIR, node_probs, datetime_str, h_l=None):
    with open(
        RESULTS_DIR / f"{testset_name}_probs{'_h' if h_l else ''}_{datetime_str}.pkl",
        "wb",
    ) as f:
        pickle.dump(node_probs, f)


def save_trans(testset_name, RESULTS_DIR, pi, datetime_str, h_l=None):
    with open(
        RESULTS_DIR / f"{testset_name}_trans{'_h' if h_l else ''}_{datetime_str}.pkl",
        "wb",
    ) as f:
        pickle.dump(pi, f)


def save_test_results(
    testset_name,
    test_aris,
    test_times,
    init_times,
    test_Z,
    test_no,
    samp_no,
    model,
    tp_str,
    use_numba=False,
):
    if testset_name == "align":
        if use_numba:
            test_Z[test_no, samp_no, :, :] = model.bp.model.jit_model.Z
            with open(  # type: ignore
                f"../../results/{testset_name}_test_Z{tp_str}.pkl",
                "wb",
            ) as f:
                pickle.dump(test_Z, f)
        else:
            test_Z[test_no, samp_no, :, :] = model.bp.model.Z
            with open(  # type: ignore
                f"../../results/{testset_name}_test_Z{tp_str}.pkl",
                "wb",
            ) as f:
                pickle.dump(test_Z, f)
    with open(  # type: ignore
        f"../../results/{testset_name}_test_aris{tp_str}.pkl",
        "wb",
    ) as f:
        pickle.dump(test_aris, f)
    with open(  # type: ignore
        f"../../results/{testset_name}_test_times{tp_str}.pkl",
        "wb",
    ) as f:
        pickle.dump(test_times, f)
    with open(  # type: ignore
        f"../../results/{testset_name}_init_times{tp_str}.pkl",
        "wb",
    ) as f:
        pickle.dump(init_times, f)


def show_true_vs_effective_params(
    model,
    params,
    sample,
    true_Z,
    use_numba=False,
):
    ## Show transition matrix inferred
    tqdm.write(f"Pi inferred: {model.bp.trans_prob}")
    try:
        tqdm.write(f"Versus true pi: {params['trans_mat']}")
    except Exception:  # KeyError:
        tqdm.write(
            f"Versus true pi: {simulation.gen_trans_mat(sample['p_stay'], sample['Q'])}"
        )
    print(f"True effective pi: {utils.effective_pi(true_Z)}")
    if use_numba:
        print(
            "Effective pi from partition inferred:",
            utils.effective_pi(model.bp.model.jit_model.Z),
        )
        print(
            "True effective beta:",
            utils.effective_beta(model.bp.model.jit_model.A, true_Z).transpose(2, 0, 1),
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


def update_test_scores(verbose, test_aris, test_no, samp_no, true_Z, model):
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
            tqdm.write(f"{np.round_(test_aris[test_no][samp_no, :], 3)}")
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
    return test_aris, max_en_aris


if __name__ == "__main__":
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
        "--directed",
        action="store_true",
        help="Use directed version of chosen model, otherwise will force symmetrise network",
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

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print verbose output"
    )

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
        "--auto_tune",
        action="store_true",
        help="Choose tuning parameter automatically according to heuristic suggested",
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
    testset_name = args.test
    # choose which set of tests to run
    if testset_name in ["default", "og", "scaling", "align"]:
        test_params = simulation.get_testset_params(testset_name)
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
                                sparse.csr_array(sample["A"][:, :, t])
                                for t in range(params["T"])
                            ]
                        print("...done")
                    with open(  # type: ignore
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
                        sparse.csr_array(sample["A"][:, :, t])
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
                data["A"] = [sparse.csr_array(A) for A in pickle.load(f)]
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
                data["A"] = [sparse.csr_array(A) for A in pickle.load(f)]
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

        data = data_processor.load_data(DATA_DIR, edge_weight_choice=args.edge_weight)
        data["Q"] = args.num_groups
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
            with open(f"../../results/{testset_name}_N.pkl", "wb") as f:  # type: ignore
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
        test_model_settings = dict(
            verbose=verbose,
            n_runs=args.n_runs,
            patience=args.patience,
            max_iter=args.max_iter,
            max_msg_iter=args.max_msg_iter,
            use_numba=args.use_numba,
            tuning_param=args.tuning_param if args.tuning_param is not None else 1.0,
            alpha_use_all=not args.alpha_use_first,
            non_informative_init=not args.partial_informative_dsbmm_init,
            planted_p=args.planted_p,
            auto_tune=args.auto_tune,
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
                        model = em.EM(sample, **test_model_settings)
                    elif testset_name == "align":
                        tqdm.write(f"alignment = {params['meta_aligned']}")
                        model = em.EM(sample, **test_model_settings)
                    else:
                        # scaling tests
                        tqdm.write(f"N = {params['N']}")
                        model = em.EM(sample, **test_model_settings)
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
                    test_aris, max_en_aris = update_test_scores(
                        verbose, test_aris, test_no, samp_no, true_Z, model
                    )
                    # print("Z inferred:", model.bp.model.jit_model.Z)
                    if verbose:
                        show_true_vs_effective_params(
                            model,
                            params,
                            sample,
                            true_Z,
                            use_numba=args.use_numba,
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
                    save_test_results(
                        testset_name,
                        test_aris,
                        test_times,
                        init_times,
                        test_Z,
                        test_no,
                        samp_no,
                        model,
                        tp_str,
                        use_numba=args.use_numba,
                    )
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
        # empirical data block
        T = len(data["A"])
        N = data["A"][0].shape[0]
        pred_Z = init_pred_Z(
            N,
            T,
            ret_best_only=args.ret_best_only,
            h_l=args.h_l,
            max_trials=args.max_trials,
            n_runs=args.n_runs,
        )
        # args.h_l, default=None, max. no. layers in hier
        # args.h_Q, default=8, max. no. groups at hier layer,
        # = 4 if h_Q > N_l / 4
        # args.h_min_N, default=20, min. nodes for split
        trial_Qs = init_trial_Qs(
            min_Q=args.min_Q,
            max_Q=args.max_Q,
            max_trials=args.max_trials,
            h_l=args.h_l,
            num_groups=args.num_groups,
            h_Q=args.h_Q,
            h_min_N=args.h_min_N,
            n_runs=args.n_runs,
        )
        if (
            not (
                args.min_Q is not None
                or args.max_Q is not None
                or args.max_trials is not None
            )
            and args.h_l is not None
        ):
            args.ret_best_only = args.n_runs == 1
            args.n_runs = 1
            # args.h_min_N

        hierarchy_layers, RESULTS_DIR = prepare_for_run(
            data, DATA_DIR, trial_Qs, h_l=args.h_l
        )

        run_hier_model(
            testset_name,
            data,
            N,
            T,
            pred_Z,
            trial_Qs,
            hierarchy_layers,
            RESULTS_DIR,
            verbose=False,
            link_choice=None,
            h_l=args.h_l,
            h_Q=args.h_Q,
            h_min_N=args.h_min_N,
            ret_best_only=args.ret_best_only,
            tuning_param=args.tuning_param,
            learning_rate=args.learning_rate,
            n_runs=args.n_runs,
            patience=args.patience,
            deg_corr=args.deg_corr,
            directed=args.directed,
            max_iter=args.max_iter,
            max_msg_iter=args.max_msg_iter,
            ignore_meta=args.ignore_meta,
            alpha_use_first=args.alpha_use_first,
            partial_informative_dsbmm_init=args.partial_informative_dsbmm_init,
            planted_p=args.planted_p,
            auto_tune=args.auto_tune,
            use_numba=args.use_numba,
            try_parallel=try_parallel,
        )

    # TODO: clean up code and improve documentation, then
    # run poetry publish
