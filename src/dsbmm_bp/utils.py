from numba import njit, prange
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


@njit(fastmath=True, error_model="numpy", parallel=True)
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
def nb_contingency_matrix(
    labels_true, labels_pred, *, eps=None, sparse=False, dtype=np.int64
):
    """Modified from sklearn impl 
    Build a contingency matrix describing the relationship between labels.
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference.
    labels_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate.
    eps : float, default=None
        If a float, that value is added to all values in the contingency
        matrix. This helps to stop NaN propagation.
        If ``None``, nothing is adjusted.
    sparse : bool, default=False
        If `True`, return a sparse CSR continency matrix. If `eps` is not
        `None` and `sparse` is `True` will raise ValueError.
        .. versionadded:: 0.18
    dtype : numeric type, default=np.int64
        Output dtype. Ignored if `eps` is not `None`.
        .. versionadded:: 0.24
    Returns
    -------
    contingency : {array-like, sparse}, shape=[n_classes_true, n_classes_pred]
        Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
        true class :math:`i` and in predicted class :math:`j`. If
        ``eps is None``, the dtype of this array will be integer unless set
        otherwise with the ``dtype`` argument. If ``eps`` is given, the dtype
        will be float.
        Will be a ``sklearn.sparse.csr_matrix`` if ``sparse=True``.
    """

    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = sp.coo_matrix(
        (np.ones(class_idx.shape[0]), (class_idx, cluster_idx)),
        shape=(n_classes, n_clusters),
        dtype=dtype,
    )
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
    else:
        contingency = contingency.toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
    return contingency


@njit
def nb_mi(pred, true):
    """Calculate mutual information between two integer vectors
    Modified from sklearn impl 

    Args:
        pred (_type_): _description_
        true (_type_): _description_
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
        labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
        contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    else:
        contingency = check_array(
            contingency,
            accept_sparse=["csr", "csc", "coo"],
            dtype=[int, np.int32, np.int64],
        )

    if isinstance(contingency, np.ndarray):
        # For an array
        nzx, nzy = np.nonzero(contingency)
        nz_val = contingency[nzx, nzy]
    elif sp.issparse(contingency):
        # For a sparse matrix
        nzx, nzy, nz_val = sp.find(contingency)
    else:
        raise ValueError("Unsupported type for 'contingency': %s" % type(contingency))

    contingency_sum = contingency.sum()
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))
    log_contingency_nm = np.log(nz_val)
    contingency_nm = nz_val / contingency_sum
    # Don't need to calculate the full outer product, just for non-zeroes
    outer = pi.take(nzx).astype(np.int64, copy=False) * pj.take(nzy).astype(
        np.int64, copy=False
    )
    log_outer = -np.log(outer) + log(pi.sum()) + log(pj.sum())
    mi = (
        contingency_nm * (log_contingency_nm - log(contingency_sum))
        + contingency_nm * log_outer
    )
    mi = np.where(np.abs(mi) < np.finfo(mi.dtype).eps, 0.0, mi)
    return np.clip(mi.sum(), 0.0, None)


@njit
def nb_nmi(pred, true):
    mi, h_x, h_y = nb_mi(pred, true)
    nmi = mi / (h_x + h_y)
    print("Success nmi")
    return nmi


@njit
def nb_nmi_local(pred, true):
    """Assume given temporal partitions in shape N x T 

    Args:
        pred (_type_): _description_
        true (_type_): _description_
    """
    assert len(pred.shape) == 2
    assert pred.shape == true.shape
    n_nan = np.isnan(pred).sum()
    assert n_nan == 0
    T = pred.shape[1]
    nmis = np.array([nb_nmi(pred[:, t], true[:, t]) for t in range(T)])

