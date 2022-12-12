import pickle
import warnings
from functools import reduce
from pathlib import Path

import csr
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

# local pkg imports
from dsbmm_bp import utils


def load_data(data_dir, edge_weight_choice=None, test_conn=False):
    """Load data from files in form described below from data_dir,
    and load into correct form for DSBMM usage, using edge_weight_choice
    to determine how to weight edges, and test_conn to determine whether
    to check connectivity of the resulting adjacencies or not.

    :param data_dir: path to the data directory
    :type data_dir: Path
    :param edge_weight_choice: name of edge metadata to use for edge weights,
                               defaults to None (binary edges)
    :type edge_weight_choice: str, optional
    :param test_conn: Test connectivity of networks at each timestep, defaults to False
    :type test_conn: bool, optional
    :raises FileNotFoundError: Necessary files not found in data_dir
    :return: data, dictionary of data in correct form for DSBMM inputs
    :rtype: dict
    """
    data = {}
    tqdm.write(f"Loading data from {data_dir}:")
    tqdm.write(
        "\tNB data must be pickled networkx graph in form 'net_[timestamp: int].pkl', with all metadata as node attributes."
    )
    tqdm.write(
        "\tFurther, there must be an additional 'meta_dists.pkl' file containing a dict of the form {meta_name : (dist_name, dim)},"
    )
    tqdm.write(
        "\twhere dist_name is one of the accepted metadata distribution types, and dim is the corresponding dimension."
    )
    data_dir = Path(data_dir)
    net_files = list(data_dir.glob("net_*.pkl"))
    try:
        assert len(net_files) > 0
    except AssertionError:
        raise FileNotFoundError(
            "No network files of required form found in specified data directory."
        )
    tqdm.write(
        f"Found {len(net_files)} network files, for timestamps {sorted(list(map(lambda x: int(x.stem.split('_')[1]), net_files)))}"
    )
    net_files = sorted(
        net_files, key=lambda x: int(x.stem.split("_")[-1])
    )  # sort by timestamp
    meta_file = data_dir / "meta_dists.pkl"
    nets = []
    for nf in net_files:
        with open(nf, "rb") as f:
            net = pickle.load(f)
        nets.append(net)
    with open(meta_file, "rb") as f:
        meta_info = pickle.load(f)
    meta_names = list(meta_info.keys())
    data["meta_names"] = meta_names
    meta_types = [meta_info[mn][0] for mn in meta_names]
    data["meta_types"] = meta_types
    meta_dims = [int(meta_info[mn][1]) for mn in meta_names]
    all_net_meta_names = set(next(iter(nets[0].nodes.data(default=np.nan)))[1].keys())
    try:
        assert all_net_meta_names == set(meta_names)
    except AssertionError:
        warnings.warn(
            f"""
            Metadata names in meta_dists.pkl do not match those in the networkx graph.
            \nWill only use metadata names in meta_dists.pkl:
            \n{meta_names}
            \n-- {all_net_meta_names - set(meta_names)} present but not used.
            """
        )
    edge_attrs = list(next(iter(nets[0].edges(data=True)))[-1].keys())
    tqdm.write("")
    tqdm.write(f"Available edge attributes: {edge_attrs}")
    if edge_weight_choice is not None:
        tqdm.write(f"Choice passed for edge weights: {edge_weight_choice}")
    else:
        tqdm.write("No choice passed for edge weights -- will assume binary edges")
    tqdm.write("")
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
    X = extract_meta(nets, meta_names, meta_dims, node_order)
    X = clean_meta(meta_names, meta_types, meta_dims, X)

    # tqdm.write([x.shape for x in X])
    data["X"] = X
    # get sparse adj mats
    A = [
        nx.to_scipy_sparse_array(net, nodelist=node_order, weight=edge_weight_choice)
        for net in nets
    ]
    if test_conn:
        test_connectivity(A)

    data["A"] = A

    return data


def save_to_pif_form(
    A,
    X,
    out_dir,
    meta_names,
    region_col_id="region",
    age_col_id="career_age",
    tpc_col_id="tpc_clust_id_cnt_vec",
    synth=True,
):
    """
    Take temporal adjacency, A, up until final time period
    in list of sparse matrices form, and temporal metadata,
    X, in dense array form, and save in out_dir converted
    to the form expected for PIF, both synth and real data:

    edgelist file
    - npz format, file named 'citation_links.npz', and array named 'edge_list'
    - first col idx of author sending citation,
    - second col idx of author receiving citation,
    - third col timestep at which this happens

    au_profs file
    - pickled pandas df, named 'au_profs.pkl'
    - columns are 'auid_idx', 'windowed_year', 'region', 'career_age'
    - no duplicate (auid,windowed_year) combinations

    au_pubs file
    - pickled pandas df, named 'au_pubs.pkl'
    - columns are 'auid_idx', 'windowed_year', 'tpc_idx'
    - no duplicate (auid,windowed_year,tpc_idx) combinations

    There should be one more timestep in the au_profs file than the edgelist file,
    and up to the final timestep they should either match, or it is assumed that
    they can be paired according to their rank. Here just output all edges,
    as can remove later if necessary.

    :param A: temporal adjacency matrices, in length T list of sparse mats
              shape (N,N) form
    :type A: List[sparse.csr_array]
    :param X: temporal metadata, in length S list of (dense) arrays
              shape (N,T,Ds) form
    :type X: List[np.ndarray]
    :param out_dir: directory in which files are created
    :type out_dir: Path
    :param meta_names: List of identifiers for each element of X
    :type meta_names: List[str]
    :param region_col_id: Name for metadata chosen to be relabelled 'region'
                          (considered as either 'country' or 'adm1' in the thesis)
                          defaults to 'region'
    :type region_col_id: str, optional
    :param age_col_id: Name for metadata chosen to be relabelled 'career_age'
                          (taken directly as 'career_age' in the thesis)
                          defaults to 'career_age'
    :type age_col_id: str, optional
    :param tpc_col_id: Name for metadata chosen to be relabelled 'tpc_idx'
    :type tpc_col_id: str, optional
    :param synth: Whether data is synthetic or real, defaults to True
    :type synth: bool, optional

    """
    out_dir = Path(out_dir)

    # convert edges to right form, but w care as nans are
    # counted as nonzero, and can't use np.isnan on sparse mat
    tmp_A = [(A_t != 0) * (utils.sparse_isnan(A_t, take_not=True)) for A_t in A]
    edgelist = np.concatenate(
        [
            np.stack([*A_t.nonzero(), t * np.ones(A_t.nnz)]).T
            for t, A_t in enumerate(tmp_A)
        ]
    )
    # save edgelist
    np.savez(out_dir / "citation_links.npz", edge_list=edgelist)

    # convert meta to right form
    if synth:
        age_idx = meta_names.index(age_col_id)
        region_idx = meta_names.index(region_col_id)
        age_meta = X[age_idx]  # should be in shape (N,T,1)
        region_meta = X[region_idx]  # should be one-hot in shape (N,T,num_regions)
        nz_age = (age_meta != 0) & (~np.isnan(age_meta))
        nz_region = (region_meta != 0) & (~np.isnan(region_meta))
    else:
        tpc_idx = meta_names.index(tpc_col_id)
        tpc_meta = X[tpc_idx]  # should be multi-hot in shape (N,T,num_tpcs)
        nz_tpcs = (tpc_meta != 0) & (~np.isnan(tpc_meta))
    # pair authors and timestep
    if synth:
        author_age_data = np.stack(
            [*nz_age[..., 0].nonzero(), age_meta[nz_age.nonzero()]], axis=1
        )
        author_region_data = np.stack(nz_region.nonzero(), axis=1)
    else:
        author_tpc_data = np.stack(nz_tpcs.nonzero(), axis=1)
    # convert to dfs then save
    if synth:
        age_df = pd.DataFrame(
            author_age_data, columns=["auid_idx", "windowed_year", "career_age"]
        )
        region_df = pd.DataFrame(
            author_region_data, columns=["auid_idx", "windowed_year", "region"]
        )
        au_profs_df = age_df.join(
            region_df.set_index(["auid_idx", "windowed_year"]),
            on=["auid_idx", "windowed_year"],
            how="outer",
        )
        # save author profiles
        with open(out_dir / "au_profs.pkl", "wb") as f:
            pickle.dump(au_profs_df, f)
    else:
        tpc_df = pd.DataFrame(
            author_tpc_data, columns=["auid_idx", "windowed_year", "tpc_idx"]
        )
        # save author publications
        with open(out_dir / "au_pubs.pkl", "wb") as f:
            pickle.dump(tpc_df, f)


def extract_meta(nets, meta_names, meta_dims, node_order):
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

    return X


def clean_meta(meta_names, meta_types, meta_dims, X, max_cats=20):
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
            # restrict to a maximum of max_cats dims 'present' for
            # each node, else in high cardinality case likely
            # equally weighting important and noisy meta
            if L > max_cats:
                tmpx = np.zeros_like(X[s])
                k = max_cats
                topkidxs = np.argsort(
                    X[s], axis=-1
                )  # will place nans at end, but should be OK as should only have either whole row nan or nothing
                np.put_along_axis(tmpx, topkidxs[..., -k:], 1, axis=-1)
                tmpx[X[s] == 0] = 0
                tmpx[missing_meta] = np.nan
                X[s] = tmpx
                # now remove any new null dims
                null_dims = np.nansum(X[s], axis=(0, 1)) == 0
                if np.count_nonzero(null_dims) > 0:
                    warnings.warn(
                        f"After restricting {meta_names[s]} to top {max_cats} for each node, following empty dimensions were found for metadata {meta_names[s]}: {np.flatnonzero(null_dims)}. Removing these dimensions."
                    )
                    X[s] = X[s][:, :, ~null_dims]
                    meta_dims[s] -= np.count_nonzero(null_dims)
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
                    (X[s] - np.nanmin(X[s], axis=-1, keepdims=True, where=X[s] != 0.0))
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
    return X


def test_connectivity(A):
    try:
        if not np.all(
            [utils.is_connected_dense(A[:, :, t]) for t in range(A.shape[-1])]
        ):
            warnings.warn("Some time periods not connected in adjacencies provided")
    except Exception:
        # assert np.all(
        #     samples[0]["A"][0].todense() == samples[0]["A"][0].T.todense()
        # )
        if not np.all(
            [utils.is_connected_sparse(csr.CSR.from_scipy(A_t)) for A_t in A]
        ):
            warnings.warn("Some time periods not connected in adjacencies provided")
