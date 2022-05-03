from numba import njit, prange
from numba.typed import List
import numpy as np


@njit(cache=True)
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


@njit(fastmath=True, error_model="numpy", parallel=True)
def gammaln_nb_p_vec(z):
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
def nb_mi(labels_true, labels_pred):
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
    sum_squares = (contingency ** 2).sum()
    C = np.empty((2, 2), dtype=np.int64)
    C[1, 1] = sum_squares - n_samples
    C[0, 1] = contingency.dot(n_k).sum() - sum_squares
    C[1, 0] = contingency.transpose().dot(n_c).sum() - sum_squares
    C[0, 0] = n_samples ** 2 - C[0, 1] - C[1, 0] - sum_squares
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
