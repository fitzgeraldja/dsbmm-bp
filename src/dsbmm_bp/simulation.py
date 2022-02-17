from numba import jit, njit, prange
import numpy as np
from scipy.stats import norm, poisson, nbinom, bernoulli

@njit
def numba_ix(arr, rows, cols):
    """
    Numba compatible implementation of arr[np.ix_(rows, cols)] for 2D arrays.
    Problem is can't use expression for assignment, so only useful for accessing
    not assigning
    :param arr: 2D array to be indexed
    :param rows: Row indices
    :param cols: Column indices
    :return: 2D array with the given rows and columns of the input array
    """
    one_d_index = np.zeros(len(rows) * len(cols), dtype=np.int32)
    for i, r in enumerate(rows):
        start = i * len(cols)
        one_d_index[start: start + len(cols)] = cols + arr.shape[1] * r

    arr_1d = arr.reshape((arr.shape[0] * arr.shape[1], 1))
    slice_1d = np.take(arr_1d, one_d_index)
    return slice_1d.reshape((len(rows), len(cols)))

def gen_trans_mat(p_stay, Q):
    """
    Generate simple transition matrix with fixed probability of group persistence, else uniform random choice of remaining groups
    Args:
        p_stay (float): Group persistence probability
        Q (int): Number of groups
    Returns:
        [type]: [description]
    """
    return np.identity(Q) * (p_stay + (p_stay - 1) / (Q - 1)) + np.ones((Q, Q)) * (
        1 - p_stay
    ) / (Q - 1)


def gen_intergroup_probs(p_in, p_out, Q):
    # actually not even necessary, would only be worth producing if intergroup probs were more different
    return np.identity(Q) * (p_in - p_out) + np.ones((Q, Q)) * p_out


def gen_beta_mat(Q, p_in, p_out):
    return (p_in - p_out) * np.identity(Q) + p_out * np.ones((Q, Q))


def gen_ppm(Z, p_in=0.4, p_out=0.1, beta_mat=None, self_loops=False):
    """
    Generate planted partition matrix given partition, and group edge probabilities
    Args:
        Z ([type]): [description]
        p_in (float, optional): [description]. Defaults to 0.4.
        p_out (float, optional): [description]. Defaults to 0.1.
        beta_mat ([type], optional): [description]. Defaults to None.
        self_loops (bool, optional): [description]. Defaults to False.
    Returns:
        [type]: [description]
    """
    Q = Z.max() + 1
    N = Z.shape[0]
    T = Z.shape[1]
    sizes = np.array(
        [[len([i for i in Z[:, t] if i == q]) for q in range(Q)] for t in range(T)]
    ).T
    # sizes[q,t] gives size of group q at time t

    idxs = [[Z[:, t] == q for q in range(Q)] for t in range(T)]
    # idxs[t][q] gives indexes of group q at time t in 1D array length N
    # now can sim uniform [0,1] rvs in shape T x N x N, and if values at index tij are
    # less than prob beta_{qi^t,qj^t}^t then give value 1 (i.e. edge) else 0
    A = np.random.rand(T, N, N)
    for t in range(T):
        for q in range(Q):
            for r in range(Q):
                if beta_mat is not None:
                    A[t][np.ix_(idxs[t][q], idxs[t][r])] = (
                        A[t][np.ix_(idxs[t][q], idxs[t][r])] <= beta_mat[t, q, r]
                    ) * 1.0
                else:
                    if q == r:
                        A[t][np.ix_(idxs[t][q], idxs[t][r])] = (
                            A[t][np.ix_(idxs[t][q], idxs[t][r])] <= p_in
                        ) * 1.0
                    else:
                        A[t][np.ix_(idxs[t][q], idxs[t][r])] = (
                            A[t][np.ix_(idxs[t][q], idxs[t][r])] <= p_out
                        ) * 1.0
                # [pois_zeros] = poisson.rvs(
                #     ZTP_params[t, q, r], size=pois_zeros.sum()
                # )

    if not self_loops:
        [np.fill_diagonal(A[t], 0) for t in range(T)]
    return A

@njit
def rand_choice_nb(arr, prob):
    """numba alternative to np random choice
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]

@njit
def pick_category(p_dist, n_samps):
    if n_samps > 1:
        print("Not implemented")
    else:
        return rand_choice_nb(np.arange(len(p_dist)),p_dist)

@njit
def evolve_Z(Z_1, trans_prob, T):
    Z = np.zeros((len(Z_1), T))
    Z[:, 0] = Z_1
    for i in range(T - 1):
        Z[:, i + 1] = np.array(
            [pick_category(trans_prob[int(zi), :], 1) for zi in Z[:, i]]
        )
    return Z.astype(np.int32)


def get_edge_prob(zi, zj, p_in, p_out):
    if zi == zj:
        return p_in
    else:
        return p_out


def sample_dynsbm_A(
    Z_1=np.zeros((10,)),
    Q=10,
    T=10,
    trans_prob=gen_trans_mat(0.8, 10),
    p_in=0.5,
    p_out=0.1,
    beta_mat=None,
    ZTP_params=None,
    self_loops=False,
):
    """Generate temporal adjacnecy given dynsbm parameters

    Args:
        Z_1 (_type_, optional): _description_. Defaults to np.zeros((10,)).
        Q (int, optional): _description_. Defaults to 10.
        T (int, optional): _description_. Defaults to 10.
        trans_prob (_type_, optional): _description_. Defaults to gen_trans_mat(0.8, 10).
        p_in (float, optional): _description_. Defaults to 0.5.
        p_out (float, optional): _description_. Defaults to 0.1.
        beta_mat (_type_, optional): _description_. Defaults to None.
        ZTP_params (_type_, optional): _description_. Defaults to None.
        self_loops (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    N = len(Z_1)

    # generate Z
    Z = evolve_Z(Z_1, trans_prob, T)

    

    # generate networks
    # inefficient, could sample total number of edges between each pair of groups, then randomly distribute these to each pair of nodes as no degree correction, or sample blocks according to size then put back together (probably best)
    # yes - obviously best thing to do is create matrix of all edge probs (easy as can construct blockwise), sample uniform randomly full matrix, then allow edge if sample <= edge prob - fine as only binary here anyway
    if beta_mat is None:
        A = gen_ppm(Z, p_in=p_in, p_out=p_out, beta_mat=beta_mat, self_loops=self_loops)
    else:
        A = gen_ppm(Z, p_in=p_in, p_out=p_out, beta_mat=beta_mat, self_loops=self_loops)
    if ZTP_params is not None:
        # Assume passing a T x Q x Q array of Poisson means
        idxs = [[Z[:, t] == q for q in range(Q)] for t in range(T)]
        A_zeros = A == 0
        for t in range(T):
            for q in range(Q):
                for r in range(Q):
                    pois_tmp = poisson.rvs(
                        ZTP_params[t, q, r], size=(idxs[t][q].sum(), idxs[t][r].sum())
                    )
                    pois_zeros = pois_tmp == 0
                    # ensure all values >0
                    while pois_zeros.sum() > 0:
                        pois_tmp[pois_zeros] = poisson.rvs(
                            ZTP_params[t, q, r], size=pois_zeros.sum()
                        )
                        pois_zeros = pois_tmp == 0
                    A[t][np.ix_(idxs[t][q], idxs[t][r])] = pois_tmp
        if A_zeros.sum() == 0:
            print("Problems")
        A[A_zeros] = 0.0

    return A, Z

def sample_dynsbm_meta(
    Z_1=np.zeros((10,)),
    Q=10,
    T=10,
    trans_prob=gen_trans_mat(0.8, 10),
    p_in=0.5,
    p_out=0.1,
    beta_mat=None,
    ZTP_params=None,
    self_loops=False,
    meta_types=["normal", "poisson", "nbinom", "indep bernoulli", "categorical"],
    meta_dims=None,
    meta_params=None,
    meta_part=None,):
    """Sample from the dynamic SBM with metadata as proposed in paper

    Args:
        Z_1 (_type_, optional): _description_. Defaults to np.zeros((10,)).
        Q (int, optional): _description_. Defaults to 10.
        T (int, optional): _description_. Defaults to 10.
        trans_prob (_type_, optional): _description_. Defaults to gen_trans_mat(0.8, 10).
        p_in (float, optional): _description_. Defaults to 0.5.
        p_out (float, optional): _description_. Defaults to 0.1.
        beta_mat (_type_, optional): _description_. Defaults to None.
        ZTP_params (_type_, optional): _description_. Defaults to None.
        self_loops (bool, optional): _description_. Defaults to False.
        meta_types (list, optional): _description_. Defaults to ["normal", "poisson", "nbinom", "indep bernoulli", "categorical"].
        meta_dims (_type_, optional): _description_. Defaults to None.
        meta_params (_type_, optional): _description_. Defaults to None.
        meta_part (_type_, optional): _description_. Defaults to None.
    """
    A,Z = sample_dynsbm_A(
    Z_1=Z_1,
    Q=Q,
    T=T,
    trans_prob=trans_prob,
    p_in=p_in,
    p_out=p_out,
    beta_mat=beta_mat,
    ZTP_params=ZTP_params,
    self_loops=self_loops,
    )
    N = len(Z_1)
    # do metadata - reliance on scipy stops using numba easily, though
    # could reimplement - see 
    # https://numba.pydata.org/numba-doc/dev/reference/pysupported.html#random
    sizes = np.array(
        [[len([i for i in Z[:, t] if i == q]) for q in range(Q)] for t in range(T)]
    ).T
    if meta_part is None:
        meta_sizes = sizes
    else:
        meta_sizes = np.array(
            [
                [len([i for i in meta_part[:, t] if i == q]) for q in range(Q)]
                for t in range(T)
            ]
        ).T

    # generate metadata
    Xt = {
        metatype: np.zeros((meta_dims[i], N, T))
        for i, metatype in enumerate(meta_types)
    }
    # params in Ds x Q x T shape - require 3d array even if Ds==1
    for i, meta_type in enumerate(meta_types):
        params = meta_params[i]
        # print(params)

        if meta_type == "normal":
            # passing mean and sd
            # initially assuming just 1d normal, generalise later
            X = [
                [
                    norm.rvs(
                        loc=params[0, q, t],
                        scale=params[1, q, t],
                        size=(meta_sizes[q, t],),
                    )
                    for q in range(Q)
                ]
                for t in range(T)
            ]

        elif meta_type == "poisson":
            # passing lambda (mean)
            X = [
                [
                    poisson.rvs(params[0, q, t], size=(meta_sizes[q, t],))
                    for q in range(Q)
                ]
                for t in range(T)
            ]
            # print('Poisson: ',len(X))

        elif meta_type == "nbinom":
            # passing r and p
            X = [
                [
                    nbinom.rvs(
                        params[0, q, t], params[1, q, t], size=(meta_sizes[q, t],)
                    )
                    for q in range(Q)
                ]
                for t in range(T)
            ]

        elif meta_type == "indep bernoulli":
            # passing independent probabilities of each category
            # means generating L x |Zq| x Q x T array - check
            L = len(params[:, 0, 0])
            X = [
                [
                    np.array(
                        [
                            bernoulli.rvs(params[l, q, t], size=(meta_sizes[q, t],))
                            for l in range(L)
                        ]
                    )
                    for q in range(Q)
                ]
                for t in range(T)
            ]
            # print('Bernoulli: ',X.shape)

        elif meta_type == "categorical":
            # passing distribution over categories
            X = [
                [pick_category(params[:, q, t], meta_sizes[q, t]) for q in range(Q)]
                for t in range(T)
            ]

        else:
            # can't raise error in numba setting
            # raise ValueError(f"Unrecognised metadata distribution: {meta_type}")
            print("Warning unknown metadata distribution used - metadata not generated")
        idxs = {}
        for q in range(Q):
            if meta_part is None:
                idxs[q] = Z == q
            else:
                idxs[q] = meta_part == q
            for t in range(T):
                if meta_dims[i] == 1:
                    Xt[meta_type][0, idxs[q][:, t], t] = X[t][q]
                else:
                    Xt[meta_type][:, idxs[q][:, t], t] = X[t][q]
                    
    return {"A":A,"Z":Z,"sizes":sizes,"X":X}
                    
    



