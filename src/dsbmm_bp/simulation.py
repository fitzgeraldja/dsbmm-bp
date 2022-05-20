import pickle

import numpy as np
from numba import njit
from scipy.stats import bernoulli
from scipy.stats import nbinom
from scipy.stats import norm
from scipy.stats import poisson
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi

# from scipy import sparse


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
        one_d_index[start : start + len(cols)] = cols + arr.shape[1] * r

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


def gen_ppm(
    Z,
    p_in=0.4,
    p_out=0.1,
    beta_mat=None,
    self_loops=False,
    directed=False,
    sparse=False,
):
    """
    Generate planted partition matrix given partition, and group edge probabilities
    Args:
        Z ([type]): [description]
        p_in (float, optional): [description]. Defaults to 0.4.
        p_out (float, optional): [description]. Defaults to 0.1.
        beta_mat ([type], optional): [description]. Defaults to None.
        self_loops (bool, optional): [description]. Defaults to False.
        directed (bool, optional): [description]. Defaults to False.
        sparse (bool, optional): [description]. Defaults to False.
    Returns:
        [type]: [description]
    """
    Q = Z.max() + 1
    N = Z.shape[0]
    T = Z.shape[1]

    idxs = [[Z[:, t] == q for q in range(Q)] for t in range(T)]
    # idxs[t][q] gives indexes of group q at time t in 1D array length N
    # now can sim uniform [0,1] rvs in shape T x N x N, and if values at index tij are
    # less than prob beta_{qi^t,qj^t}^t then give value 1 (i.e. edge) else 0
    if not sparse:
        A = np.random.rand(T, N, N)
        for t in range(T):
            for q in range(Q):
                for r in range(q, Q):
                    if beta_mat is not None:
                        if len(beta_mat.shape) == 3:
                            A[t][np.ix_(idxs[t][q], idxs[t][r])] = (
                                A[t][np.ix_(idxs[t][q], idxs[t][r])]
                                <= beta_mat[t, q, r]
                            ) * 1.0
                        elif len(beta_mat.shape) == 2:
                            A[t][np.ix_(idxs[t][q], idxs[t][r])] = (
                                A[t][np.ix_(idxs[t][q], idxs[t][r])] <= beta_mat[q, r]
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
                    if not directed and r != q:
                        A[t][np.ix_(idxs[t][r], idxs[t][q])] = A[t][
                            np.ix_(idxs[t][q], idxs[t][r])
                        ].T
                    elif directed and r != q:
                        if beta_mat is not None:
                            if len(beta_mat.shape) == 3:
                                A[t][np.ix_(idxs[t][r], idxs[t][q])] = (
                                    A[t][np.ix_(idxs[t][r], idxs[t][q])]
                                    <= beta_mat[t, q, r]
                                ) * 1.0
                            elif len(beta_mat.shape) == 2:
                                A[t][np.ix_(idxs[t][r], idxs[t][q])] = (
                                    A[t][np.ix_(idxs[t][r], idxs[t][q])]
                                    <= beta_mat[q, r]
                                ) * 1.0
                        else:
                            A[t][np.ix_(idxs[t][r], idxs[t][q])] = (
                                A[t][np.ix_(idxs[t][r], idxs[t][q])] <= p_out
                            ) * 1.0
                    # [pois_zeros] = poisson.rvs(
                    #     ZTP_params[t, q, r], size=pois_zeros.sum()
                    # )
    else:
        # sparse construction - only implemented currently for p_in / p_out formulation
        try:
            assert beta_mat is None
        except Exception:  # AssertionError:
            raise ValueError(
                "Sparse construction only implemented for p_in / p_out formulation"
            )
        # sizes = np.array(
        #     [[len([i for i in Z[:, t] if i == q]) for q in range(Q)] for t in range(T)]
        # ).T
        # sizes[q,t] gives size of group q at time t
        # sparse_blocks = [
        #     [
        #         [
        #             sparse.random(
        #                 sizes[q, t], sizes[r, t], density=p_in if r == q else p_out,
        #             )
        #             for r in range(Q)
        #         ]
        #         for q in range(Q)
        #     ]
        #     for t in range(T)
        # ]
        # A = [sparse.to_csr(A_t) for A_t in A]
        # TODO: finish (above will likely need to be tweaked - possibly want to draw nnzs for each block pair (from binomial) and/or directly
        # sample edge pairs according to idxs followed by sparse construction according to COO format)

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
        return rand_choice_nb(np.arange(len(p_dist)), p_dist)


@njit
def evolve_Z(Z_1, trans_prob, T):
    Z = np.zeros((len(Z_1), T))
    Z[:, 0] = Z_1
    for i in range(T - 1):
        Z[:, i + 1] = np.array(
            [pick_category(trans_prob[int(zi), :], 1) for zi in Z[:, i]]
        )
    return Z.astype(np.int32)


def ari_meta_aligned(Z, target_ari, mask_prop=0.01, max_trials=100):
    """
    Generate new series of partitions, where at each timestep (so locally), the
    ARI between the new partition and the old partition is roughly equal to the
    target ARI.

    Args:
        Z (_type_): Base (temporal) partition, shape N x T
        target_ari (float): target ARI
        mask_prop (float, optional): Proportion of partition at each timestep to mask each loop - smaller
                                     gives finer control (so closer to target ARI), but is slower.
                                     Defaults to 0.01.
        max_trials (int, optional): maximum number of trials to get roughly correct
    """
    Zalt = Z.copy()
    for t in range(Z.shape[1]):
        for trial_no in range(max_trials):
            mask = np.random.rand(Z.shape[0]) < mask_prop
            Zalt[mask, t] = np.random.randint(
                0, high=Z[:, t].max() + 1, size=mask.sum()
            )
            if ari(Z[:, t], Zalt[:, t]) < target_ari:
                break

    return Zalt


def nmi_meta_aligned(Z, target_nmi, mask_prop=0.01, max_trials=1000):
    """
    Generate new series of partitions, where at each timestep (so locally), the
    NMI between the new partition and the old partition is roughly equal to the
    target NMI.

    Args:
        Z (_type_): Base (temporal) partition, shape N x T
        target_nmi (float): target NMI
        mask_prop (float, optional): Proportion of partition at each timestep to mask each loop - smaller
                                     gives finer control (so closer to target NMI), but is slower.
                                     Defaults to 0.01.
        max_trials (int, optional): maximum number of trials to get roughly correct
    """
    Zalt = Z.copy()
    for t in range(Z.shape[1]):
        for trial_no in range(max_trials):
            mask = np.random.rand(Z.shape[0]) < mask_prop
            Zalt[mask, t] = np.random.randint(
                0, high=Z[:, t].max() + 1, size=mask.sum()
            )
            if nmi(Z[:, t], Zalt[:, t]) < target_nmi:
                break

    return Zalt


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
    directed=False,
    sparse=False,
):
    """Generate temporal adjacency given dynsbm parameters

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
        directed (bool, optional): _description_. Defaults to False.
        sparse (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # N = len(Z_1)

    # generate Z
    Z = evolve_Z(Z_1, trans_prob, T)

    # generate networks
    # inefficient, could sample total number of edges between each pair of groups, then randomly distribute these to each pair of nodes as no degree correction, or sample blocks according to size then put back together (probably best)
    # yes - obviously best thing to do is create matrix of all edge probs (easy as can construct blockwise), sample uniform randomly full matrix, then allow edge if sample <= edge prob - fine as only binary here anyway
    if beta_mat is None:
        A = gen_ppm(
            Z,
            p_in=p_in,
            p_out=p_out,
            beta_mat=beta_mat,
            self_loops=self_loops,
            directed=directed,
            sparse=sparse,
        )
    else:
        A = gen_ppm(
            Z,
            p_in=p_in,
            p_out=p_out,
            beta_mat=beta_mat,
            self_loops=self_loops,
            directed=directed,
            sparse=sparse,
        )
    if ZTP_params is not None:
        # Assume passing a T x Q x Q array of Poisson means
        idxs = [[Z[:, t] == q for q in range(Q)] for t in range(T)]
        A_zeros = A == 0
        for t in range(T):
            for q in range(Q):
                for r in range(Q):
                    pois_tmp = poisson.rvs(
                        ZTP_params[t, q, r],
                        size=(idxs[t][q].sum(), idxs[t][r].sum()),
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
    meta_types=[
        "normal",
        "poisson",
        "nbinom",
        "indep bernoulli",
        "categorical",
    ],
    meta_dims=None,
    meta_params=None,
    meta_part=None,
    directed=False,
    sparse=False,
):
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
        meta_part (np.array or float, optional): either specified partition for metadata, or float defining score of alignment.
                                                 Defaults to None.
    """
    A, Z = sample_dynsbm_A(
        Z_1=Z_1,
        Q=Q,
        T=T,
        trans_prob=trans_prob,
        p_in=p_in,
        p_out=p_out,
        beta_mat=beta_mat,
        ZTP_params=ZTP_params,
        self_loops=self_loops,
        directed=directed,
        sparse=sparse,
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
        if type(meta_part) != np.ndarray:
            meta_part = ari_meta_aligned(Z, meta_part)
        meta_sizes = np.array(
            [
                [len([i for i in meta_part[:, t] if i == q]) for q in range(Q)]
                for t in range(T)
            ]
        ).T

    if meta_dims is None and meta_params is not None:
        meta_dims = [mp.shape[0] for mp in meta_params]
    # generate metadata
    Xt = {
        metatype: np.zeros((meta_dims[i], N, T), order="C")
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
                        params[0, q, t],
                        params[1, q, t],
                        size=(meta_sizes[q, t],),
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
        # idxs = {}
        for q in range(Q):
            if meta_part is None:
                idxs = Z == q
            else:
                idxs = meta_part == q
            for t in range(T):
                if meta_dims[i] == 1:
                    Xt[meta_type][0, idxs[:, t], t] = X[t][q]
                else:
                    Xt[meta_type][:, idxs[:, t], t] = X[t][q]

    return {
        "A": np.ascontiguousarray(A.transpose(1, 2, 0)) if not sparse else A,
        "Z": np.ascontiguousarray(Z),
        "X": {k: np.ascontiguousarray(v) for k, v in Xt.items()},
        "meta_part": np.ascontiguousarray(meta_part),
    }


def gen_test_data(
    test_no="000",
    n_samps=20,
    N=100,
    Q=4,
    meta_Q=None,
    p_in=None,
    p_out=None,
    beta_mat=None,
    T=5,
    p_stay=None,
    trans_mat=None,
    directed=False,
    sparse=False,
    meta_types=["poisson", "indep bernoulli"],
    L=4,
    meta_dims=None,
    pois_params=None,
    base_bern_params=None,
    indep_bern_params=None,
    sample_meta_params=False,
    meta_aligned=True,
    save=False,
    export_dir=".",
):
    """Generate test data given params (multiple runs,
    optionally saved)

    Args:
        test_no (_type_): _description_
        n_samps (int, optional): _description_. Defaults to 20.
        N (int, optional): _description_. Defaults to 100.
        Q (int, optional): _description_. Defaults to 4.
        meta_Q (_type_, optional): _description_. Defaults to None.
        p_in (_type_, optional): _description_. Defaults to None.
        p_out (_type_, optional): _description_. Defaults to None.
        beta_mat (_type_, optional): _description_. Defaults to None.
        T (int, optional): _description_. Defaults to 5.
        p_stay (_type_, optional): _description_. Defaults to None.
        trans_mat (_type_, optional): _description_. Defaults to None.
        directed (bool, optional): _description_. Defaults to False.
        meta_types (list, optional): _description_. Defaults to ["poisson", "indep bernoulli"].
        L (int, optional): _description_. Defaults to 4.
        meta_dims (_type_, optional): _description_. Defaults to None.
        pois_params (_type_, optional): _description_. Defaults to None.
        base_bern_params (_type_, optional): _description_. Defaults to None.
        indep_bern_params (_type_, optional): _description_. Defaults to None.
        sample_meta_params (bool,optional): _description_. Defaults to False.
        meta_aligned (bool or float, optional): Whether metadata is perfectly aligned or not (bool),
                                                or rough ARI between net and meta partition at each
                                                step. Defaults to True.
        save (bool, optional): _description_. Defaults to False.
        export_dir (str, optional): _description_. Defaults to ".".

    Returns:
        _type_: _description_

    Example for test 7 - misaligned metadata for relatively difficult test (poor group stability)
    gen_test_data(
        7,
        n_samps=20,
        N=100,
        Q=4,
        p_in=0.4,
        p_out=0.1,
        p_stay=0.6,
        T=5,
        meta_types=["poisson", "indep bernoulli"],
        L=4,
        meta_aligned=False,
        save=True,
        export_dir='.',
    )
    """
    if "poisson" in meta_types and pois_params is None:
        pois_params = np.array(
            [[[5 * (q + 1)] for q in range(Q)] for t in range(T)]
        ).T  # + np.random.normal(loc=0.0,scale=0.5,size=(1,Q,T))
    if "indep bernoulli" in meta_types and indep_bern_params is None:
        if base_bern_params is None:
            base_bern_params = 0.1 * (
                np.ones((L, 1, T))
            )  # +np.random.normal(loc=0,scale=0.1))
        indep_bern_params = np.concatenate(
            [base_bern_params * ib_fac for ib_fac in np.linspace(1, 9, Q)],
            axis=1,
        )
    if sample_meta_params:
        # TODO: allow generalisation of params to sample around
        meta_params = [
            [
                np.array(
                    [
                        [[np.random.randint(3, high=12)] for q in range(Q)]
                        for t in range(T)
                    ]
                ).T,
                np.random.rand(L, Q, T),
            ]
            for _ in range(n_samps)
        ]
    else:
        meta_params = [pois_params, indep_bern_params]

    if meta_dims is None and meta_types == ["poisson", "indep bernoulli"]:
        meta_dims = [1, L]

    params = {
        "N": N,
        "Q": Q,
        "meta_Q": meta_Q,
        "block_params": {"p_in": p_in, "p_out": p_out, "beta_mat": beta_mat},
        "p_stay": p_stay,
        "trans_mat": trans_mat,
        "T": T,
        "meta_types": meta_types,
        "L": L,
        "meta_dims": meta_dims,
        "meta_params": meta_params,
    }

    tests = []
    # meta_parts = []
    if trans_mat is None:
        trans_mat = gen_trans_mat(p_stay, Q)

    for samp_no in range(n_samps):
        Z_1 = np.random.randint(0, high=Q, size=(N,))
        sizes = np.array([len([i for i in Z_1 if i == q]) for q in range(Q)])
        cum_size = np.cumsum(sizes)
        Z_1 = np.zeros((N,))
        Z_1[: cum_size[0]] = 0
        for q in range(1, Q):
            Z_1[cum_size[q - 1] : cum_size[q]] = q

        if meta_aligned is True:
            meta_part = None
        else:
            if meta_Q is None:
                meta_Q = Q
            if meta_aligned is False:
                meta_part = evolve_Z(
                    np.random.randint(0, high=meta_Q, size=(N,)),
                    gen_trans_mat(p_stay, meta_Q),
                    T,
                )  # TODO: allow to pass more general transitions for metadata
                # meta_parts.append(meta_part)
            else:
                # have explicitly passed degree of alignment as float
                meta_part = meta_aligned

        if not sample_meta_params:
            test = sample_dynsbm_meta(
                Z_1,
                Q=Q,
                T=T,
                meta_types=meta_types,
                meta_dims=meta_dims,
                meta_params=meta_params,
                p_in=p_in,
                p_out=p_out,
                beta_mat=beta_mat,
                trans_prob=trans_mat,
                meta_part=meta_part,
                directed=directed,
                sparse=sparse,
            )
        else:
            test = sample_dynsbm_meta(
                Z_1,
                Q=Q,
                T=T,
                meta_types=meta_types,
                meta_dims=meta_dims,
                meta_params=meta_params[samp_no],
                p_in=p_in,
                p_out=p_out,
                beta_mat=beta_mat,
                trans_prob=trans_mat,
                meta_part=meta_part,
                directed=directed,
                sparse=sparse,
            )
        tests.append(test)

    if save:
        with open(export_dir + f"/test{test_no}_params.pkl", "wb") as f:
            pickle.dump(params, f)
        with open(export_dir + f"/test{test_no}_data.pkl", "wb") as f:
            pickle.dump(tests, f)

    return tests


################################################################
##################   SPECIFY TESTSET PARAMS   ##################
################################################################

default_test_params = {}

n_samps = 20
default_test_params["n_samps"] = n_samps
test_no = [1, 2, 3, 4, 5, 6, 7, 8]
default_test_params["test_no"] = test_no
N = [100, 100, 100, 100, 100, 100, 100, 100]
default_test_params["N"] = N
Q = [4, 4, 4, 4, 4, 4, 4, 4]
default_test_params["Q"] = Q
c_in = [
    10,
    10,
    6,
    6,
    10,
    6,
    10,
    6,
]
p_in = [ci / ni for ci, ni in zip(c_in, N)]
default_test_params["p_in"] = p_in
c_out = 2
p_out = [c_out / ni for ni in N]
default_test_params["p_out"] = p_out
p_stay = [
    0.8,
    0.6,
    0.8,
    0.6,
    0.6,
    0.8,
    0.6,
    0.6,
]
default_test_params["p_stay"] = p_stay
T = 5
default_test_params["T"] = T
meta_types = ["poisson", "indep bernoulli"]
default_test_params["meta_types"] = meta_types
L = 4
default_test_params["L"] = L
meta_dims = [1, L]
default_test_params["meta_dims"] = meta_dims
pois_params = [
    np.array(
        [[[5 * (q + 1)] for q in range(Q_i)] for t in range(T)]
    ).T  # + np.random.normal(loc=0.0,scale=0.5,size=(1,Q,T))
    for Q_i in Q
]
base_bern_params = 0.1 * (np.ones((L, 1, T)))  # +np.random.normal(loc=0,scale=0.1))
indep_bern_params = [
    np.concatenate(
        [base_bern_params * ib_fac for ib_fac in np.linspace(1, 9, Q_i)],
        axis=1,
    )
    for Q_i in Q
]
meta_params = list(zip(pois_params, indep_bern_params))
default_test_params["meta_params"] = meta_params
meta_align = [
    True,
    True,
    True,
    True,
    False,
    False,
    False,
    False,
]
default_test_params["meta_align"] = meta_align

# OG paper tests
og_test_params = {}
og_test_params["test_no"] = []
og_test_params["n_samps"] = n_samps
Q = 2
og_test_params["Q"] = Q
N = 100
og_test_params["N"] = N
Ts = [5, 10]
og_test_params["T"] = []
beta_11 = np.array([0.2, 0.25, 0.3, 0.4, 0.3])
beta_12 = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
beta_22 = np.array([0.15, 0.2, 0.2, 0.2, 0.3])
beta_mats = [
    np.array([[beta_11[i], beta_12[i]], [beta_12[i], beta_22[i]]])
    for i in range(len(beta_11))
]
og_test_params["beta_mat"] = []

# pi_low, pi_medium, pi_high from paper
trans_mats = [gen_trans_mat(p, Q) for p in [0.6, 0.75, 0.9]]
og_test_params["trans_mat"] = []

meta_types = ["poisson", "indep bernoulli"]
og_test_params["meta_types"] = meta_types
L = 4
og_test_params["L"] = L
meta_dims = [1, L]
og_test_params["meta_dims"] = meta_dims

# meta_params_og = []
og_test_params["sample_meta_params"] = True

for bn, beta_mat in enumerate(beta_mats):
    for trn, trans_mat in enumerate(trans_mats):
        for tn, T in enumerate(Ts):
            testno = 13 + bn * len(trans_mats) * len(Ts) + trn * len(Ts) + tn
            # print(testno)
            og_test_params["test_no"].append(testno)
            og_test_params["T"].append(T)
            og_test_params["trans_mat"].append(trans_mat)
            og_test_params["beta_mat"].append(beta_mat)


# scaling tests
def calc_det_limit(p_stay, c_in, c_out, Q, tol=1e-1):
    c = (c_in + (Q - 1) * c_out) / Q
    lam = (c_in - c_out) / (Q * c)
    bdd_up = c * lam * lam
    bdd_down = (1 - p_stay * p_stay) / (1 + p_stay * p_stay)
    if np.any(bdd_up - tol < bdd_down):
        print(
            "Warning: Q for scaling test given params is close to theoretical det. limit (in no meta case)"
        )
        print(f"For test idxs {np.where(bdd_up - tol < bdd_down)[0]}")
    # note that GCC exists in undirected graph iff E[k^2] - 2E[k] > 0, so this is also important


scaling_test_params = {}
n_tests = 20
n_samps = 3
scaling_test_params["n_samps"] = n_samps
test_no = np.arange(400, 400 + n_tests)
scaling_test_params["test_no"] = test_no
N = np.floor(np.exp(np.linspace(5, 8, n_tests))).astype(int)
scaling_test_params["N"] = N
Q = np.floor(np.log(N)).astype(int)
scaling_test_params["Q"] = Q
c_in = 20
p_in = [c_in / ni for ni in N]
scaling_test_params["p_in"] = p_in
c_out = 5
p_out = [c_out / ni for ni in N]
scaling_test_params["p_out"] = p_out
p_stay = 0.8 * np.ones_like(N)
scaling_test_params["p_stay"] = p_stay

calc_det_limit(p_stay, c_in, c_out, Q)

T = 5
scaling_test_params["T"] = T
meta_types = ["poisson", "indep bernoulli"]
scaling_test_params["meta_types"] = meta_types
L = 4
scaling_test_params["L"] = L
meta_dims = [1, L]
scaling_test_params["meta_dims"] = meta_dims
pois_params = [
    np.array(
        [[[5 * (q + 1)] for q in range(Q_i)] for t in range(T)]
    ).T  # + np.random.normal(loc=0.0,scale=0.5,size=(1,Q,T))
    for Q_i in Q
]
base_bern_params = 0.1 * (np.ones((L, 1, T)))  # +np.random.normal(loc=0,scale=0.1))
indep_bern_params = [
    np.concatenate(
        [base_bern_params * ib_fac for ib_fac in np.linspace(1, 9, Q_i)],
        axis=1,
    )
    for Q_i in Q
]
meta_params = list(zip(pois_params, indep_bern_params))
scaling_test_params["meta_params"] = meta_params
meta_align = np.ones_like(N, dtype=bool)
scaling_test_params["meta_align"] = meta_align

# alignment tests
align_test_params = {}

n_samps = 20
n_tests = 40
align_test_params["n_samps"] = n_samps
test_no = np.arange(500, 500 + n_tests)
align_test_params["test_no"] = test_no
N = 100 * np.ones_like(test_no)
align_test_params["N"] = N
Q = 4 * np.ones_like(test_no)
align_test_params["Q"] = Q
c_in = np.concatenate([10 * np.ones(n_tests // 2), 6 * np.ones(n_tests - n_tests // 2)])
p_in = [ci / ni for ci, ni in zip(c_in, N)]
align_test_params["p_in"] = p_in
c_out = 2
p_out = [c_out / ni for ni in N]
align_test_params["p_out"] = p_out
p_stay = 0.8 * np.ones_like(test_no)
p_stay[::2] = 0.6
align_test_params["p_stay"] = p_stay
T = 5
align_test_params["T"] = T
meta_types = ["poisson", "indep bernoulli"]
align_test_params["meta_types"] = meta_types
L = 4
align_test_params["L"] = L
meta_dims = [1, L]
align_test_params["meta_dims"] = meta_dims
pois_params = [
    np.array(
        [[[5 * (q + 1)] for q in range(Q_i)] for t in range(T)]
    ).T  # + np.random.normal(loc=0.0,scale=0.5,size=(1,Q,T))
    for Q_i in Q
]
base_bern_params = 0.1 * (np.ones((L, 1, T)))  # +np.random.normal(loc=0,scale=0.1))
indep_bern_params = [
    np.concatenate(
        [base_bern_params * ib_fac for ib_fac in np.linspace(1, 9, Q_i)],
        axis=1,
    )
    for Q_i in Q
]
meta_params = list(zip(pois_params, indep_bern_params))
align_test_params["meta_params"] = meta_params
tmp = np.linspace(0.1, 1.0, n_tests // 2)
meta_align = np.zeros(n_tests)
meta_align[::2] = tmp
meta_align[1::2] = tmp
align_test_params["meta_align"] = meta_align
