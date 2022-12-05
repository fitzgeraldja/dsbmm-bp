from itertools import chain, combinations_with_replacement, permutations, product
from typing import Generator

import numpy as np
from csr import CSR
from numba import bool_, int32, njit, prange
from numba.typed import List
from scipy import sparse


@njit(parallel=True, cache=True)
def numba_ix(arr, rows, cols):
    """
    Numba compatible implementation of arr[np.ix_(rows, cols)] for 2D arrays.
    :param arr: 2D array to be indexed
    :param rows: Row indices
    :param cols: Column indices
    :return: 2D array with the given rows and columns of the input array

    From https://github.com/numba/numba/issues/5894
    """
    one_d_index = np.zeros(len(rows) * len(cols), dtype=np.int32)
    for i, r in enumerate(rows):
        start = i * len(cols)
        one_d_index[start : start + len(cols)] = cols + arr.shape[1] * r

    arr_1d = arr.reshape((arr.shape[0] * arr.shape[1], 1))
    slice_1d = np.take(arr_1d, one_d_index)
    return slice_1d.reshape((len(rows), len(cols)))


@njit
def is_connected_dense(G: np.ndarray):
    """Check if a graph is connected

    Args:
        G (np.array): (Dense) adjacency matrix of the
                      graph

    Returns:
        bool: True if graph connected, False otherwise
    """
    # TODO: check if actually faster in native python
    # using deques
    # NB only checking weak connectivity
    visited = np.zeros(G.shape[0], dtype=bool_)
    visited[0] = True
    stack = [0]
    while stack:
        node = stack.pop()
        for n in np.nonzero(G[node])[0]:
            if not visited[n]:
                visited[n] = True
                stack.append(n)
        for n in np.nonzero(G.T[node])[0]:
            if not visited[n]:
                visited[n] = True
                stack.append(n)
    return np.all(visited)


@njit
def is_connected_sparse(G: CSR):
    """Check if a graph is connected

    Args:
        G (CSR): (Sparse) adjacency matrix of the graph

    Returns:
        bool: True if graph is connected, False otherwise
    """
    # NB only checking weak connectedness
    G_T = G.transpose()
    visited = np.zeros(G.nrows, dtype=bool_)
    visited[0] = True
    stack = [0]
    while stack:
        node = stack.pop()
        for n in G.row_cs(node):
            if not visited[n]:
                visited[n] = True
                stack.append(n)
        for n in G_T.row_cs(node):
            if not visited[n]:
                visited[n] = True
                stack.append(n)
    return np.all(visited)


@njit
def connected_components(G: np.ndarray) -> Generator[set, None, None]:
    seen = set()
    for v in range(G.shape[0]):
        if v not in seen:
            c = _plain_bfs(G, v)
            seen.update(c)
            yield c


@njit
def _plain_bfs(G: np.ndarray, source: int) -> set:
    """A fast BFS node generator"""
    seen = set()
    nextlevel = {source}
    while nextlevel:
        thislevel = nextlevel
        nextlevel = set()
        for v in thislevel:
            if v not in seen:
                seen.add(v)
                nextlevel.update(np.nonzero(G[v, :])[0])
    return seen


@njit
def connected_components_sparse(G: CSR) -> Generator[set, None, None]:
    # TODO: consider return to generator formulation
    seen = set()
    all_comps = []
    for v in range(G.nrows):
        v = int32(v)
        if v not in seen:
            c = _plain_bfs_sparse(G, v)
            seen.update(c)
            # yield c
            all_comps.append(c)
    return all_comps  # type: ignore


@njit
def _plain_bfs_sparse(G: CSR, source: int):
    """A fast BFS node generator"""
    # only seek weak connectivity
    G_T = G.transpose()
    seen = set()
    nextlevel = {source}
    while nextlevel:
        thislevel = nextlevel
        nextlevel = set()
        for v in thislevel:
            if v not in seen:
                seen.add(v)
                nextlevel.update(G.row_cs(v))
                nextlevel.update(G_T.row_cs(v))
    return seen


@njit(fastmath=True, error_model="numpy", parallel=True)
def gammaln_nb_p_vec(z: np.ndarray):
    """Numerical Recipes 6.1
    Code from https://stackoverflow.com/questions/55048299/why-is-this-log-gamma-numba-function-slower-than-scipy-for-large-arrays-but-fas"""
    # Don't use global variables.. (They only can be changed if you recompile the function)
    coefs = np.array(
        [
            57.1562356658629235,
            -59.5979603554754912,
            14.1360979747417471,
            -0.491913816097620199,
            0.339946499848118887e-4,
            0.465236289270485756e-4,
            -0.983744753048795646e-4,
            0.158088703224912494e-3,
            -0.210264441724104883e-3,
            0.217439618115212643e-3,
            -0.164318106536763890e-3,
            0.844182239838527433e-4,
            -0.261908384015814087e-4,
            0.368991826595316234e-5,
        ]
    )

    out = np.empty(z.shape[0])

    for i in prange(z.shape[0]):
        y = z[i]
        tmp = z[i] + 5.24218750000000000
        tmp = (z[i] + 0.5) * np.log(tmp) - tmp
        ser = 0.999999999999997092

        n = coefs.shape[0]
        for j in range(n):
            y = y + 1.0
            ser = ser + coefs[j] / y

        out[i] = tmp + np.log(2.5066282746310005 * ser / z[i])
    return out


@njit(cache=True)
def _logpoispmf_vec(k, mu):
    # log lkl is k*log(mu)-log(k!)-mu
    # can use that gamma(k)=(k-1)! for k +ve integer
    # and approx for log(gamma(k)) below
    Pk = k * np.log(mu) - gammaln_nb_p_vec(k + 1) - mu
    return Pk


@njit(cache=True)
def nb_poisson_lkl_vec(k, mu):
    """Calculate Poisson lkl of observing k given param mu

    Args:
        k (_type_): _description_
        mu (_type_): _description_
    """

    # p(k) = exp(-mu)*mu^k/k!
    return np.exp(_logpoispmf_vec(k, mu))


@njit(fastmath=True, error_model="numpy")
def gammaln_nb_p_int(z):
    """Numerical Recipes 6.1
    Code from https://stackoverflow.com/questions/55048299/why-is-this-log-gamma-numba-function-slower-than-scipy-for-large-arrays-but-fas"""
    # Don't use global variables.. (They only can be changed if you recompile the function)
    # This version is for z an integer rather than vector
    coefs = np.array(
        [
            57.1562356658629235,
            -59.5979603554754912,
            14.1360979747417471,
            -0.491913816097620199,
            0.339946499848118887e-4,
            0.465236289270485756e-4,
            -0.983744753048795646e-4,
            0.158088703224912494e-3,
            -0.210264441724104883e-3,
            0.217439618115212643e-3,
            -0.164318106536763890e-3,
            0.844182239838527433e-4,
            -0.261908384015814087e-4,
            0.368991826595316234e-5,
        ]
    )
    # out = np.empty(1)
    y = z
    tmp = z + 5.24218750000000000
    tmp = (z + 0.5) * np.log(tmp) - tmp
    ser = 0.999999999999997092

    n = coefs.shape[0]
    for j in range(n):
        y += 1.0
        # assert y != z
        ser += coefs[j] / y

    out = tmp + np.log(2.5066282746310005 * ser / z)
    return out


@njit(cache=True)
def _logpoispmf_int(k, mu):
    # log lkl is k*log(mu)-log(k!)-mu
    # can use that gamma(k)=(k-1)! for k +ve integer
    # and approx for log(gamma(k)) below
    Pk = k * np.log(mu) - gammaln_nb_p_int(k + 1) - mu
    return Pk


@njit(cache=True)
def nb_poisson_lkl_int(k, mu):
    """Calculate Poisson lkl of observing k given param mu

    Args:
        k (_type_): _description_
        mu (_type_): _description_
    """
    # p(k) = exp(-mu)*mu^k/k!
    return np.exp(_logpoispmf_int(k, mu))


@njit(cache=True)
def nb_ib_lkl(xs, ps):
    """Calculate IB lkl of observing xs given probs ps

    Args:
        xs (_type_): _description_
        ps (_type_): _description_

    Returns:
        _type_: _description_
    """
    # assume L dim vecs passed for each
    return np.prod(np.power(ps, xs)) * np.prod(np.power(1 - ps, 1 - xs))


@njit
def nb_contingency_matrix(labels_true, labels_pred):
    """Build a contingency matrix describing the relationship between labels.
        NB possibly less efficient than sklearn implementation via numpy

    Args:
        labels_true (_type_): _description_
        labels_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    N = len(labels_true)
    assert len(labels_pred) == N
    labels_true = labels_true.astype(np.int64)
    labels_pred = labels_pred.astype(np.int64)
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    contingency = np.zeros((n_classes, n_clusters))
    for i in range(N):
        contingency[labels_true[i], labels_pred[i]] += 1
    return contingency


@njit
def nb_mi(labels_true, labels_pred, contingency=None):
    # TODO: fix
    """Calculate mutual information between two integer vectors
    Modified from sklearn impl

    Args:
        labels_true (_type_): _description_
        labels_pred (_type_): _description_
    """

    """Mutual Information between two clusterings.
    The Mutual Information is a measure of the similarity between two labels
    of the same data. Where :math:`|U_i|` is the number of the samples
    in cluster :math:`U_i` and :math:`|V_j|` is the number of the
    samples in cluster :math:`V_j`, the Mutual Information
    between clusterings :math:`U` and :math:`V` is given as:
    .. math::
        MI(U,V)=\\sum_{i=1}^{|U|} \\sum_{j=1}^{|V|} \\frac{|U_i\\cap V_j|}{N}
        \\log\\frac{N|U_i \\cap V_j|}{|U_i||V_j|}
    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.
    This metric is furthermore symmetric: switching :math:`U` (i.e
    ``label_true``) with :math:`V` (i.e. ``label_pred``) will return the
    same score value. This can be useful to measure the agreement of two
    independent label assignments strategies on the same dataset when the
    real ground truth is not known.
    Read more in the :ref:`User Guide <mutual_info_score>`.
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        A clustering of the data into disjoint subsets, called :math:`U` in
        the above formula.
    labels_pred : int array-like of shape (n_samples,)
        A clustering of the data into disjoint subsets, called :math:`V` in
        the above formula.
    contingency : {ndarray, sparse matrix} of shape \
            (n_classes_true, n_classes_pred), default=None
        A contingency matrix given by the :func:`contingency_matrix` function.
        If value is ``None``, it will be computed, otherwise the given value is
        used, with ``labels_true`` and ``labels_pred`` ignored.
    Returns
    -------
    mi : float
       Mutual information, a non-negative value, measured in nats using the
       natural logarithm.
    Notes
    -----
    The logarithm used is the natural logarithm (base-e).
    See Also
    --------
    adjusted_mutual_info_score : Adjusted against chance Mutual Information.
    normalized_mutual_info_score : Normalized Mutual Information.
    """
    if contingency is None:
        # labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
        contingency = nb_contingency_matrix(labels_true, labels_pred)
    # else:
    #     contingency = check_array(
    #         contingency,
    #         accept_sparse=["csr", "csc", "coo"],
    #         dtype=[int, np.int32, np.int64],
    #     )

    if isinstance(contingency, np.ndarray):
        # For an array
        nzx, nzy = np.nonzero(contingency)
        nnzs = len(nzx)
        nz_val = np.zeros(nnzs)
        for i in range(nnzs):
            nz_val[i] = contingency[nzx[i], nzy[i]]
    # elif sp.issparse(contingency):
    #     # For a sparse matrix
    #     nzx, nzy, nz_val = sp.find(contingency)
    # else:
    #     raise ValueError("Unsupported type for 'contingency': %s" % type(contingency))

    contingency_sum = contingency.sum()
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))
    log_contingency_nm = np.log(nz_val)
    contingency_nm = nz_val / contingency_sum
    # Don't need to calculate the full outer product, just for non-zeroes
    outer = pi.take(nzx).astype(np.int64, copy=False) * pj.take(nzy).astype(
        np.int64, copy=False
    )
    log_outer = -np.log(outer) + np.log(pi.sum()) + np.log(pj.sum())
    mi = (
        contingency_nm * (log_contingency_nm - np.log(contingency_sum))
        + contingency_nm * log_outer
    )
    mi = np.where(np.abs(mi) < np.finfo(mi.dtype).eps, 0.0, mi)
    # h_x =
    # h_y =
    return np.clip(mi.sum(), 0.0, None)  # , h_x, h_y


@njit
def nb_nmi(labels_true, labels_pred):
    mi, h_x, h_y = nb_mi(labels_true, labels_pred)
    nmi = mi / (h_x + h_y)
    print("Success nmi")
    return nmi


@njit
def nb_nmi_local(labels_true, labels_pred):
    """Assume given temporal partitions in shape N x T

    Args:
        pred (_type_): _description_
        true (_type_): _description_
    """
    assert len(labels_true.shape) == 2
    assert labels_true.shape == labels_pred.shape
    n_nan = np.isnan(labels_true).sum()
    assert n_nan == 0
    T = labels_true.shape[1]
    nmis = np.array([nb_nmi(labels_true[:, t], labels_pred[:, t]) for t in range(T)])
    return nmis


@njit
def nb_pair_confusion_mat(labels_true, labels_pred):
    """Using sklearn implementation suitably mod"""
    n_samples = np.int64(labels_true.shape[0])

    # Computation using the contingency data
    contingency = nb_contingency_matrix(labels_true, labels_pred)
    n_c = np.ravel(contingency.sum(axis=1))
    n_k = np.ravel(contingency.sum(axis=0))
    sum_squares = (contingency**2).sum()
    C = np.empty((2, 2), dtype=np.int64)
    C[1, 1] = sum_squares - n_samples
    C[0, 1] = contingency.dot(n_k).sum() - sum_squares
    C[1, 0] = contingency.transpose().dot(n_c).sum() - sum_squares
    C[0, 0] = n_samples**2 - C[0, 1] - C[1, 0] - sum_squares
    return C


@njit
def nb_ari(labels_true, labels_pred):
    """Using sklearn implementation suitably mod"""
    (tn, fp), (fn, tp) = nb_pair_confusion_mat(labels_true, labels_pred)
    # convert to Python integer types, to avoid overflow or underflow
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    # Special cases: empty data or full agreement
    if fn == 0 and fp == 0:
        return 1.0

    return 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))


@njit
def nb_ari_local(labels_true, labels_pred):
    """Calculate ARI between true and inferred partition at each timestep"""
    assert len(labels_true.shape) == 2
    assert labels_true.shape == labels_pred.shape
    n_nan = np.isnan(labels_true).sum()
    assert n_nan == 0
    T = labels_true.shape[1]
    aris = np.array([nb_ari(labels_true[:, t], labels_pred[:, t]) for t in range(T)])
    return aris


@njit
def nb_cc_merge(parent: np.ndarray, x: int):
    if parent[x] == x:
        return x
    return nb_cc_merge(parent, parent[x])


@njit
def nb_connected_components(N: int, edges: np.ndarray):
    parent = np.arange(N)
    for source, target in edges:
        parent[nb_cc_merge(parent, source)] = nb_cc_merge(parent, target)

    n_cc = 0
    for i in range(N):
        n_cc += parent[i] == i

    for i in range(N):
        parent[i] = nb_cc_merge(parent, parent[i])

    comps = List()
    nodes = np.arange(N)
    for val in np.unique(parent):
        comp = nodes[parent == val]
        comps.append(comp)
    return n_cc, comps


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


def max_overlap_over_perms(true_Z, pred_Z):
    """Calculate normalised maximum overlap between true and predicted Z

    :param true_Z: _description_
    :type true_Z: _type_
    :param pred_Z: _description_
    :type pred_Z: _type_
    :return: _description_
    :rtype: _type_
    """
    Q = len(np.unique(true_Z))
    perms = permutations(range(Q))
    max_ol = 0.0
    for perm in perms:
        tmp_Z = np.zeros_like(pred_Z)
        for q in range(Q):
            tmp_Z[pred_Z == q] = perm[q]
        tmp_ol = ((tmp_Z == true_Z).mean() - 1 / Q) / (1 - 1 / Q)
        if tmp_ol > max_ol:
            max_ol = tmp_ol
    return max_ol


def construct_hier_trans(hier_pis_run, pred_ZL, h_min_N):
    """Construct transition matrix given set of pi inferred
    for each level of a hierarchy, and number of descendant groups
    at each level

    :param hier_pis_run: set of transition matrices -- length L list, with each element
                either a (Q,Q) trans mat (first element), or a dict with keys
                the group at level above to which corresponding groups belong,
                and values the (Q,Q) trans mats inferred.
                Assumed to descend hierarchy (smaller Q to larger Q)
    :type hier_pis_run: List[np.ndarray,dict[int,np.ndarray]]
    :param pred_ZL: predicted group labels at each level of hierarchy --
                    np.ndarray shape (L,N,T) of ints
    :type pred_ZL: np.ndarray
    :param h_min_N: minimum number of nodes in a group at each level of hierarchy
    :type h_min_N: int
    """

    L = len(hier_pis_run)
    topdown_hier, small_qs = get_hier(pred_ZL, h_min_N)
    # print(topdown_hier, small_qs)
    all_q = np.unique(pred_ZL[-1, ...][pred_ZL[-1, ...] != -1])
    all_q = np.concatenate([all_q, list(small_qs)]).astype(int)
    all_q.sort()
    h_Q = len(topdown_hier[0].keys())
    # print(topdown_hier)
    all_q_at_l = [set(topdown_hier[0].keys())] + [
        set(chain.from_iterable(topdown_hier[l].values())) for l in range(L - 1)
    ]
    bottomup_hier = []
    lq_idxs = {q: q_idx for q_idx, q in enumerate(sorted(topdown_hier[0].keys()))}
    for hier in topdown_hier[::-1]:
        rev_hier = {}
        for up_q, down_qs in hier.items():
            for lq_idx, q in enumerate(sorted(down_qs)):
                rev_hier[q] = up_q
                lq_idxs[q] = lq_idx
        bottomup_hier.append(rev_hier)

    total_Q = len(all_q)
    n_descs = {q: len(sub_qs) for q, sub_qs in topdown_hier[0].items()}
    if L == 2:
        n_descs.update({q: 1 for q in chain.from_iterable(topdown_hier[0].values())})
    else:
        for l in range(1, L - 1):
            # l_qs = list(topdown_hier[l].keys())
            n_descs.update({q: len(sub_qs) for q, sub_qs in topdown_hier[l].items()})
            for l_upper in range(l - 1, -1, -1):
                for upper_q, l_qs in topdown_hier[l_upper].items():
                    for l_q in l_qs:
                        # n_descs should be total number of descs
                        # at lowest level, i.e. total number of leaf
                        # nodes branching from q
                        if l_q in n_descs.keys():
                            # previously counted as leaf, so now
                            # need to add children - 1
                            n_descs[upper_q] += n_descs[l_q] - 1

            if l == L - 2:  # last level
                n_descs.update(
                    {q: 1 for q in chain.from_iterable(topdown_hier[l].values())}
                )
    n_descs = {q: (float(v) if v > 0 else 1.0) for q, v in n_descs.items()}
    # add in small groups, who have no descs
    n_descs.update({q: 1.0 for q in small_qs})
    # REMOVE:
    # try:
    #     assert set(list(n_descs.keys())) == set(all_q)
    # except AssertionError:
    #     print(set(all_q) - set(list(n_descs.keys())))
    #     print(small_qs)
    #     print(topdown_hier)
    #     raise ValueError("n_descs keys don't match all_q")

    all_hier_qs = np.unique(pred_ZL[pred_ZL != -1])
    q_depth = {}
    for q in all_hier_qs:
        for l, qs in enumerate(all_q_at_l):
            if q in qs:
                q_depth[q] = l
    # print(q_depth)
    # print(total_Q)
    # print(all_hier_qs)
    # print(all_q_at_l)
    # for each (q,r) comb, collect (\ell,q^\ell,r^\ell,lq_idx,lr_idx) tuples,
    # where lq_idx, rq_idx are the idxs of q^\ell, r^\ell resp. in the
    # inferred model (so from 0 to h_Q-1)
    anc_pairs = {}
    for q_idx, r_idx in combinations_with_replacement(range(total_Q), 2):
        q, r = all_q[q_idx], all_q[r_idx]
        if q_idx == r_idx:
            anc_pairs[(q_idx, r_idx)] = (
                q_depth[q],
                q,
                r,
                lq_idxs[q],
                lq_idxs[r],
            )
        else:
            if (q_depth[q] == 0) and (q_depth[r] == 0):
                anc_pairs[(q_idx, r_idx)] = (
                    0,
                    q,
                    r,
                    lq_idxs[q],
                    lq_idxs[r],
                )
            elif (q_depth[q] == 0) and (q_depth[r] != 0):
                for l in range(L - 1):
                    r = bottomup_hier[l].get(r, r)
                anc_pairs[(q_idx, r_idx)] = (
                    0,
                    q,
                    r,
                    lq_idxs[q],
                    lq_idxs[r],
                )
            elif (q_depth[q] != 0) and (q_depth[r] == 0):
                for l in range(L - 1):
                    q = bottomup_hier[l].get(q, q)
                anc_pairs[(q_idx, r_idx)] = (
                    0,
                    q,
                    r,
                    lq_idxs[q],
                    lq_idxs[r],
                )
            else:
                l_ctr = 0
                for l in range(L - 1):
                    u_q = bottomup_hier[l].get(q, q)
                    u_r = bottomup_hier[l].get(r, r)
                    if u_q == u_r:
                        anc_pairs[(q_idx, r_idx)] = (
                            L - 1 - l,
                            q,
                            r,
                            lq_idxs[q],
                            lq_idxs[r],
                        )
                        break
                    elif l == L - 2:
                        # if we're at the last level and still haven't
                        # found a common ancestor, must have split
                        # immediately
                        try:
                            assert u_q < h_Q
                            assert u_r < h_Q
                        except AssertionError:
                            print(q, r)
                            print(u_q, u_r)
                            print(bottomup_hier)
                            print(l)
                            if l_ctr > 0:
                                for l in range(l_ctr):
                                    print(bottomup_hier[l].keys())
                                    print(bottomup_hier[l][r])
                                    print(bottomup_hier[l].get(r, r))
                                    try:
                                        print(bottomup_hier[l][r])
                                    except:
                                        print(type(r))
                                        print(
                                            *[type(k) for k in bottomup_hier[l].keys()]
                                        )
                                        raise ValueError("something up with keys")
                            raise ValueError("Problem w anc pairs")
                        anc_pairs[(q_idx, r_idx)] = (
                            0,
                            u_q,
                            u_r,
                            lq_idxs[u_q],
                            lq_idxs[u_r],
                        )
                    else:
                        q = u_q
                        r = u_r
                    l_ctr += 1

    pi = np.zeros((total_Q, total_Q))
    for q_idx, r_idx in product(range(total_Q), repeat=2):
        if (q_idx, r_idx) in anc_pairs.keys():
            ell, q, r, lq_idx, lr_idx = anc_pairs[(q_idx, r_idx)]
        elif (r_idx, q_idx) in anc_pairs.keys():
            ell, r, q, lr_idx, lq_idx = anc_pairs[(r_idx, q_idx)]
        else:
            raise ValueError(
                "No ancestor relationship found for (%d,%d)" % (q_idx, r_idx)
            )
        if ell == 0:
            pi[q_idx, r_idx] = hier_pis_run[0][lq_idx, lr_idx] / n_descs[r]
        else:
            prefac = 1.0
            # 0 is L-1, so ell is L-1-ell
            u_q = bottomup_hier[L - 1 - ell][q]
            if ell == 1:
                prefac *= hier_pis_run[0][lq_idxs[u_q], lq_idxs[u_q]]
            else:
                for m in range(ell):
                    if ell - m - 1 == 0:
                        prefac *= hier_pis_run[0][lq_idxs[uu_q], lq_idxs[u_q]]
                    else:
                        try:
                            uu_q = bottomup_hier[L - 1 - (ell - m - 1)][u_q]
                            prefac *= hier_pis_run[ell - m - 1][uu_q][
                                lq_idxs[u_q], lq_idxs[u_q]
                            ]
                            u_q = uu_q
                        except KeyError:
                            print(u_q)
                            print(ell, m)
                            print(L - 1 - (ell - m))
                            print(hier_pis_run[ell - m - 1].keys())
                            print(bottomup_hier)
                            print(ell - m)
                            raise ValueError("key problem")
                        except TypeError:
                            print(u_q, uu_q)
                            print(ell, m)
                            print(L - 1 - (ell - m))
                            print(bottomup_hier)
                            print(ell - m)
                            print(lq_idxs[u_q])
                            raise ValueError("type problem")
            u_q = bottomup_hier[L - 1 - ell][q]
            if prefac > 1.0:
                print(prefac)
                raise ValueError("prefac > 1")
            elif n_descs[r] < 1:
                print(n_descs[r])
                raise ValueError("n_descs[r] < 1")
            try:
                pi[q_idx, r_idx] = (
                    prefac * hier_pis_run[ell][u_q][lq_idx, lr_idx]
                ) / n_descs[r]
            except KeyError:
                print(ell, u_q)
                print(hier_pis_run[ell][u_q])
                print(lq_idxs[q])
                print(lq_idxs[r])
                print(topdown_hier)
                print(n_descs[r])
                raise ValueError("Key error")
            except TypeError:
                print(q_idx, r_idx)
                print(L)
                print(ell)
                print(len(hier_pis_run))
                print(u_q)
                print(hier_pis_run[ell])
                print(hier_pis_run[ell][u_q])
                print(lq_idxs[u_q])
                print(n_descs)
                raise ValueError("type problem")
    # print(n_descs)
    # print(topdown_hier)
    # print(anc_pairs)
    # print(all_q)
    return pi, all_q


def get_hier(pred_ZL, h_min_N):
    """Given predicted group labels at each level of hierarchy, return
    hierarchy as list of dicts length L-1, where for each element, keys
    are labels at that level of hierarchy, value are array containing
    labels at next level that are contained in that group

    :param pred_ZL: predicted group labels at each level of hierarchy --
                    np.ndarray shape (L,N,T) of ints
    :type pred_ZL: np.ndarray
    """
    Z_1 = pred_ZL[0, ...]
    L = pred_ZL.shape[0]
    h_Q = len(np.unique(Z_1[Z_1 != -1]))
    # NB use
    # q_shift = h_Q*(n_suff_large + no_q) + n_small
    # at each level, and n_small will belong to whatever level they
    # first appeared, so can proceed recursively
    l = 0
    old_Z = pred_ZL[l, ...]
    qs = np.unique(old_Z[old_Z != -1])
    node_group_cnts = np.stack([(old_Z == q).sum(axis=1) for q in qs], axis=0)
    old_node_labels = np.argmax(
        node_group_cnts, axis=0
    )  # assigns each node its most common label
    q_idxs, group_cnts = np.unique(old_node_labels, return_counts=True)
    suff_large_q_idxs = q_idxs[group_cnts > h_min_N]
    suff_large_q = qs[suff_large_q_idxs]
    small_q_idxs = q_idxs[group_cnts <= h_min_N]
    small_qs = set(qs[small_q_idxs])
    # n_suff_large = len(suff_large_q_idxs)
    # n_small = (group_cnts <= h_min_N).sum()

    topdown_hier = [{q: [] for q in range(h_Q)}]
    for l in range(1, L):
        Z_l = pred_ZL[l, ...]
        # l_qs = np.unique(Z_l[Z_l!=-1])
        # print(suff_large_q)
        for q_idx in suff_large_q_idxs:
            q = qs[q_idx]
            sub_Zq = Z_l[old_node_labels == q_idx, :]
            sub_q = np.unique(sub_Zq[sub_Zq != -1])
            # print(sub_q)
            topdown_hier[l - 1][q] = sub_q

        if l < L - 1:
            old_Z = pred_ZL[l, ...]
            qs = np.unique(old_Z[old_Z != -1])
            node_group_cnts = np.stack([(old_Z == q).sum(axis=1) for q in qs], axis=0)
            old_node_labels = np.argmax(
                node_group_cnts, axis=0
            )  # assigns each node its most common label
            q_idxs, group_cnts = np.unique(old_node_labels, return_counts=True)
            suff_large_q_idxs = q_idxs[group_cnts > h_min_N]
            suff_large_q = qs[suff_large_q_idxs]
            # n_suff_large = len(suff_large_q_idxs)
            small_q_idxs = q_idxs[group_cnts <= h_min_N]
            small_qs |= set(qs[small_q_idxs])
            # n_small = (group_cnts <= h_min_N).sum()
            topdown_hier.append({q: [] for q in suff_large_q})

    return topdown_hier, small_qs


def sparse_isnan(A: sparse.csr_array, take_not=False):
    """Return boolean array of same shape as A indicating whether each
    element is nan or not

    :param A: sparse array with nan values
    :type A: sparse.csr_array
    """
    indptr, indices, data = A.indptr, A.indices, A.data
    if take_not:
        return sparse.csr_array((~np.isnan(data), indices, indptr), shape=A.shape)
    else:
        return sparse.csr_array((np.isnan(data), indices, indptr), shape=A.shape)
