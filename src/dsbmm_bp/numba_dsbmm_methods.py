# numba reimplementation of all methods for DSBMM class that reasonably stand to gain from doing so
# - simply prepend method name with nb_
import numpy as np
import yaml  # type: ignore
from numba import njit, prange
from utils import nb_ib_lkl, nb_poisson_lkl_int

with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
TOL = config["tol"]  # min value permitted for msgs etc (for numerical stability)
NON_INFORMATIVE_INIT = config[
    "non_informative_init"
]  # initialise alpha, pi as uniform (True), or according to init part passed (False)
USE_FASTMATH = config["use_fastmath"]

# TODO: annotate all fns


@njit(parallel=True, fastmath=USE_FASTMATH)
def nb_calc_meta_lkl(
    N, T, Q, meta_types, _meta_params, X, tuning_param, _pres_nodes, verbose
):
    meta_lkl = np.ones((N, T, Q))
    for s, mt in enumerate(meta_types):
        # print(f"Updating params for {mt} dist")
        if mt == "poisson":
            # print("In Poisson")
            pois_params = _meta_params[s]  # shape (Q x T x 1)
            # recall X[s] has shape (N x T x Ds), w Ds = 1 here
            for t in range(T):
                for i in range(N):
                    if _pres_nodes[i, t]:
                        for q in range(Q):
                            # potentially could speed up further but
                            # loops still v efficient in numba
                            meta_lkl[i, t, q] *= nb_poisson_lkl_int(
                                X[s][i, t, 0], pois_params[q, t, 0]
                            )
            if verbose:
                print("\tUpdated Poisson lkl contribution")
        elif mt == "indep bernoulli":
            # print("In IB")
            ib_params = _meta_params[s]  # shape (Q x T x L)
            # recall X[s] has shape (N x T x Ds), w Ds = L here
            for t in range(T):
                for i in range(N):
                    if _pres_nodes[i, t]:
                        for q in range(Q):
                            meta_lkl[i, t, q] *= nb_ib_lkl(
                                X[s][i, t, :], ib_params[q, t, :]
                            )
            if verbose:
                print("\tUpdated IB lkl contribution")
        elif mt == "categorical":
            cat_params = _meta_params[
                s
            ]  # shape (Q x T x L), where now sum_l = 1 for all q,t
            for t in range(T):
                for i in range(N):
                    if _pres_nodes[i, t]:
                        for q in range(Q):
                            meta_lkl[i, t, q] *= cat_params[
                                q, t, np.flatnonzero(X[s][i, t, :])
                            ]
            if verbose:
                print("\tUpdated categorical lkl contribution")
        else:
            raise NotImplementedError(
                "Yet to implement metadata distribution of given type \nOptions are 'poisson', 'indep bernoulli', or 'categorical'"
            )  # NB can't use string formatting for print in numba
    for i in prange(N):
        for t in range(T):
            if _pres_nodes[i, t]:
                for q in range(Q):
                    if meta_lkl[i, t, q] < TOL:
                        meta_lkl[i, t, q] = TOL
                    meta_lkl[i, t, q] = meta_lkl[i, t, q] ** tuning_param
                    if meta_lkl[i, t, q] < TOL:
                        meta_lkl[i, t, q] = TOL
                    elif meta_lkl[i, t, q] > 1 - TOL:
                        meta_lkl[i, t, q] = 1 - TOL
    return meta_lkl


@njit(parallel=True, fastmath=USE_FASTMATH)
def nb_update_alpha(
    init,
    learning_rate,
    N,
    T,
    Q,
    Z,
    _tot_N_pres,
    _alpha,
    node_marg,
    diff,
    verbose,
):
    if init:
        if NON_INFORMATIVE_INIT:
            _alpha = np.ones(Q) / Q
        else:
            # case of no marginals / true partition provided to calculate most likely params
            _alpha = np.array([(Z == q).sum() / _tot_N_pres for q in range(Q)])
            if _alpha.sum() > 0:
                _alpha /= _alpha.sum()
            _alpha[_alpha < TOL] = TOL
            _alpha /= _alpha.sum()
            # _alpha[_alpha > 1 - TOL] = 1 - TOL
    else:
        # if DC, seems like should multiply marg by degree prior to sum - unseen for directed case but can calculate
        tmp = np.zeros(Q)
        # print("Updating alpha")
        #
        for q in range(Q):
            for t in range(T):
                for i in prange(N):
                    tmp[q] += node_marg[i, t, q]
            tmp[q] /= _tot_N_pres
        if tmp.sum() > 0:
            tmp /= tmp.sum()
        tmp[tmp < TOL] = TOL
        tmp /= tmp.sum()
        # tmp[tmp > 1 - TOL] = 1 - TOL

        tmp = learning_rate * tmp + (1 - learning_rate) * _alpha
        tmp_diff = np.abs(tmp - _alpha).mean()
        if np.isnan(tmp_diff):
            raise RuntimeError("Problem updating alpha")
        if verbose:
            print("Alpha diff:", np.round_(tmp_diff, 3))
        diff += tmp_diff
        _alpha = tmp
    return _alpha, diff


@njit(parallel=True, fastmath=USE_FASTMATH)
def nb_update_pi(
    init,
    learning_rate,
    N,
    T,
    Q,
    Z,
    _pres_trans,
    _pi,
    twopoint_time_marg,
    diff,
    verbose,
):
    qqprime_trans = np.zeros((Q, Q))
    if init:
        if NON_INFORMATIVE_INIT:
            qqprime_trans = np.ones((Q, Q))
        else:
            for i in prange(N):
                for t in range(T - 1):
                    if _pres_trans[i, t]:
                        q1 = Z[i, t]
                        q2 = Z[i, t + 1]
                        qqprime_trans[q1, q2] += 1

            # TODO: provide different cases for non-uniform init clustering
            # NB for uniform init clustering this provides too homogeneous pi, and is more important now
            trans_sums = np.zeros(Q)
            for q in range(Q):
                trans_sums[q] = qqprime_trans[q, :].sum()
            for q in range(Q):
                if trans_sums[q] > 0:
                    qqprime_trans[q, :] /= trans_sums[q]
            p = 0.8
            qqprime_trans = p * qqprime_trans + (1 - p) * np.random.rand(
                *qqprime_trans.shape
            )
    else:
        for q in range(Q):
            for qprime in range(Q):
                for i in prange(N):
                    for t in range(T - 1):
                        if _pres_trans[i, t]:
                            qqprime_trans[q, qprime] += twopoint_time_marg[
                                i, t, q, qprime
                            ]
        # qqprime_trans /= np.expand_dims(
        #     node_marg[:, :-1, :].sum(axis=0).sum(axis=0), 1
        # )  # need to do sums twice as numba axis argument
        # # only takes integers (rather than axis=(0,1) as
        # # we would want) - can't use this as node_marg sums could
        # # be tiny / zero
        # below is unnecessary as can enforce normalisation directly
        # - just introduces instability
        # tot_marg = node_marg[:, :-1, :].sum(axis=0).sum(axis=0)
        # print("tot marg sums:", tot_marg)
        # for q in range(Q):
        #     if tot_marg[q] > TOL:
        #         qqprime_trans[q, :] = qqprime_trans[q, :] / tot_marg[q]
        #     else:
        #         raise RuntimeError("Problem with node marginals")
        # qqprime_trans[q, :] = TOL
        # correct_pi()
    trans_sums = np.zeros(Q)
    for q in range(Q):
        trans_sums[q] = qqprime_trans[q, :].sum()
    for q in range(Q):
        if trans_sums[q] > 0:
            qqprime_trans[q, :] /= trans_sums[q]
    trans_sums = np.zeros(Q)
    for q in range(Q):
        for qprime in range(Q):
            if qqprime_trans[q, qprime] < TOL:
                qqprime_trans[q, qprime] = TOL
        trans_sums[q] = qqprime_trans[q, :].sum()
    for q in range(Q):
        # normalise rows last time
        qqprime_trans[q, :] /= trans_sums[q]
    if not init:
        tmp = learning_rate * qqprime_trans + (1 - learning_rate) * _pi
        tmp_diff = np.abs(tmp - _pi).mean()
        if np.isnan(tmp_diff):
            raise RuntimeError("Problem updating pi")
        if verbose:
            print("Pi diff:", np.round_(tmp_diff, 3))
        diff += tmp_diff
        _pi = tmp
    else:
        _pi = qqprime_trans
    return _pi, diff


@njit(parallel=True, fastmath=USE_FASTMATH)
def nb_update_lambda(
    init,
    learning_rate,
    directed,
    _edge_vals,
    nbrs,
    degs,
    kappa,
    T,
    Q,
    Z,
    _lam,
    node_marg,
    twopoint_edge_marg,
    diff,
    verbose,
):
    # TODO: fix on basis of beta below
    lam_num = np.zeros((Q, Q, T))
    lam_den = np.zeros((Q, Q, T))
    if init:
        for e_idx in prange(_edge_vals.shape[0]):
            i, j, t, val = _edge_vals[e_idx, :]
            i, j, t = int(i), int(j), int(t)
            lam_num[Z[i, t], Z[j, t], t] += val

        # np.array(
        #     [
        #         [
        #             [
        #                 A[Z[:, t] == q, Z[:, t] == r, t].sum()
        #                 for r in range(Q)
        #             ]
        #             for q in range(Q)
        #         ]
        #         for t in range(T)
        #     ]
        # )
        # lam_den = np.array(
        #     [
        #         [degs[Z[:, t] == q].sum() for q in range(Q)]
        #         for t in range(T)
        #     ]
        # )
        # lam_den = np.einsum("tq,tr->tqr", lam_den, lam_den)
        for q in range(Q):
            for t in range(T):
                # TODO: check if this is correct, as makes vanishing
                lam_den[q, q, t] = (
                    kappa[q, t, 1] * kappa[q, t, 0]
                )  # TODO: check right for r==q
                if lam_num[q, q, t] < TOL:
                    lam_num[q, q, t] = TOL
                if lam_den[q, q, t] < TOL:
                    lam_den[q, q, t] = 1.0
                for r in range(q + 1, Q):
                    lam_den[q, r, t] = (
                        kappa[q, t, 1] * kappa[r, t, 0]
                    )  # TODO: check right in directed case
                    if lam_num[q, r, t] < TOL:
                        lam_num[q, r, t] = TOL
                    if lam_den[q, r, t] < TOL:
                        lam_den[q, r, t] = 1.0
                    if not directed:
                        lam_num[r, q, t] = lam_num[q, r, t]
                        lam_den[r, q, t] = lam_den[q, r, t]
                if directed:
                    for r in range(q):
                        lam_den[q, r, t] = (
                            kappa[q, t, 1] * kappa[r, t, 0]
                        )  # TODO: check right in directed case
                        if lam_num[q, r, t] < TOL:
                            lam_num[q, r, t] = TOL
                        if lam_den[q, r, t] < TOL:
                            lam_den[q, r, t] = 1.0

        # lam_den = np.array(
        #     [
        #         [
        #             [
        #                 kappa[q, t, 0] * kappa[r, t, 0]
        #                 for t in range(T)
        #             ]
        #             for r in range(Q)
        #         ]
        #         for q in range(Q)
        #     ]
        # )
    else:
        # lam_num = np.einsum("ijtqr,ijt->qrt", twopoint_edge_marg, A)
        for q in range(Q):
            for r in range(q, Q):
                for e_idx in prange(_edge_vals.shape[0]):
                    i, j, t, val = _edge_vals[e_idx, :]
                    i, j, t = int(i), int(j), int(t)
                    j_idx = nbrs[t][i] == j
                    # if r==q: # TODO: special treatment?
                    lam_num[q, r, t] += (twopoint_edge_marg[t][i][j_idx, q, r] * val)[0]
                for t in range(T):
                    if lam_num[q, r, t] < TOL:
                        lam_num[q, r, t] = TOL
                    if not directed:
                        lam_num[r, q, t] = lam_num[q, r, t]
            if directed:
                for r in range(q):
                    for e_idx in prange(_edge_vals.shape[0]):
                        i, j, t, val = _edge_vals[e_idx, :]
                        i, j, t = int(i), int(j), int(t)
                        j_idx = nbrs[t][i] == j
                        # if r==q: # TODO: special treatment?
                        lam_num[q, r, t] += (
                            twopoint_edge_marg[t][i][j_idx, q, r] * val
                        )[0]

        # lam_den = np.einsum("itq,it->qt", node_marg, degs)
        # lam_den = np.einsum("qt,rt->qrt", lam_den, lam_den)
        marg_kappa_out = np.zeros((Q, T))
        marg_kappa_in = np.zeros((Q, T))
        for q in range(Q):
            for t in range(T):
                marg_kappa_out[q, t] = (
                    node_marg[:, t, q] * degs[:, t, 1]
                ).sum()  # TODO: again check this uses right deg if directed
                marg_kappa_in[q, t] = (node_marg[:, t, q] * degs[:, t, 0]).sum()
        for q in range(Q):
            for t in range(T):
                for r in range(q, Q):
                    lam_den[q, r, t] = marg_kappa_out[q, t] * marg_kappa_in[r, t]
                    if lam_den[q, r, t] < TOL:
                        lam_den[q, r, t] = 1.0
                    if not directed:
                        lam_den[r, q, t] = lam_den[q, r, t]
                if directed:
                    for r in range(q):
                        lam_den[q, r, t] = marg_kappa_out[q, t] * marg_kappa_in[r, t]
                        if lam_den[q, r, t] < TOL:
                            lam_den[q, r, t] = 1.0
    # NB use relative rather than absolute difference here as lam could be large
    tmp = lam_num / lam_den
    if not init:
        tmp = learning_rate * tmp + (1 - learning_rate) * _lam
        tmp_diff = np.abs((tmp - _lam) / _lam).mean()
        if np.isnan(tmp_diff):
            raise RuntimeError("Problem updating lambda")
        if verbose:
            print("Lambda diff:", np.round_(tmp_diff, 3))
        diff += tmp_diff
        _lam = tmp
    else:
        _lam = tmp
    return _lam, diff


@njit(parallel=True, fastmath=USE_FASTMATH)
def nb_update_beta(
    init: bool,
    learning_rate: float,
    directed: bool,
    _edge_vals: np.ndarray,
    _pres_nodes: np.ndarray,
    nbrs,
    degs: np.ndarray,
    _n_qt: np.ndarray,
    T: int,
    Q: int,
    Z: np.ndarray,
    _beta: np.ndarray,
    node_marg: np.ndarray,
    twopoint_edge_marg,
    diff: float,
    verbose: bool,
) -> tuple[np.ndarray, float]:
    beta_num = np.zeros((Q, Q, T))
    beta_den = np.ones((Q, Q, T))
    if init:
        # TODO: consider alt init as random / uniform (both options considered in OG BP SBM code, random seems used in practice)
        if NON_INFORMATIVE_INIT:
            # assign as near uniform - just assume edges twice as likely in comms as out,
            # and that all groups have same average out-degree at each timestep
            Ns = _pres_nodes.sum(axis=0)
            av_degs = degs.sum(axis=0)[:, 1] / Ns
            # beta_in = 2*beta_out
            # N*(beta_in + (Q - 1)*beta_out) = av_degs
            # = (Q + 1)*beta_out*N
            beta_out = av_degs / (Ns * (Q + 1))
            beta_in = 2 * beta_out
            for t in range(T):
                for q in range(Q):
                    for r in range(Q):
                        if r == q:
                            beta_num[q, r, t] = beta_in[t]
                        else:
                            beta_num[q, r, t] = beta_out[t]
        # beta_num = np.array(
        #     [
        #         [
        #             [
        #                 (A[Z[:, t] == q, Z[:, t] == r, t] > 0).sum()
        #                 for r in range(Q)
        #             ]
        #             for q in range(Q)
        #         ]
        #         for t in range(T)
        #     ]
        # )
        else:
            for e_idx in prange(_edge_vals.shape[0]):
                i, j, t, _ = _edge_vals[e_idx, :]
                i, j, t = int(i), int(j), int(t)
                beta_num[Z[i, t], Z[j, t], t] += 1
            for q in range(Q):
                # enforce uniformity for identifiability
                tmp = 0.0
                for t in range(T):
                    tmp += beta_num[q, q, t]
                for t in range(T):
                    beta_num[q, q, t] = tmp
            # beta_den = np.array(
            #     [
            #         [degs[Z[:, t] == q].sum() for q in range(Q)]
            #         for t in range(T)
            #     ]
            # )
            # beta_den = np.einsum("tq,tr->tqr", beta_den, beta_den)
            # print("beta_num:", beta_num.transpose(2, 0, 1))
            # print("kappa:", kappa)
            for q in range(Q):
                for t in range(T):
                    for r in range(q, Q):
                        beta_den[q, r, t] = (
                            _n_qt[q, t] * _n_qt[r, t]
                        )  # TODO: check right in directed case, and if r==q
                        # if beta_num[q, r, t] < TOL:
                        #     beta_num[q, r, t] = TOL
                        if beta_den[q, r, t] < TOL:
                            beta_den[
                                q, r, t
                            ] = 1.0  # this is same as how prev people have handled (effectively just don't do division if will cause problems, as num will
                            # be v. small anyway)
                        if not directed and r != q:
                            beta_den[r, q, t] = beta_den[q, r, t]
                            beta_num[
                                q, r, t
                            ] /= 2.0  # done elsewhere, TODO: check basis
                            beta_num[r, q, t] = beta_num[q, r, t]
                    if directed:
                        for r in range(q):
                            beta_den[q, r, t] = _n_qt[q, t] * _n_qt[r, t]
                            if beta_den[q, r, t] < TOL:
                                beta_den[q, r, t] = 1.0
            for q in range(Q):
                # enforce uniformity for identifiability
                tmp = 0.0
                for t in range(T):
                    tmp += beta_den[q, q, t]
                for t in range(T):
                    beta_den[q, q, t] = tmp
            # print("beta_den:", beta_den)
    else:
        # beta_num = np.einsum(
        #     "ijtqr,ijt->qrt", twopoint_edge_marg, (A > 0)
        # )
        for q in range(Q):
            for r in range(q, Q):
                if r != q:
                    for e_idx in prange(_edge_vals.shape[0]):
                        i, j, t, _ = _edge_vals[e_idx, :]
                        i, j, t = int(i), int(j), int(t)
                        j_idx = np.flatnonzero(nbrs[t][i] == j)
                        # print(twopoint_edge_marg[t][i][j_idx, q, r])
                        # assert j_idx.sum() == 1
                        val = twopoint_edge_marg[t][i][j_idx, q, r][0]
                        # try:
                        #     assert not np.isnan(val)
                        # except:
                        #     print("(i,j,t):", i, j, t)
                        #     print("A[i,j,t] = ", a_ijt)
                        #     print("twopoint marg: ", val)
                        #     raise RuntimeError("Problem updating beta")
                        beta_num[q, r, t] += val
                    if not directed:
                        for t in range(T):
                            beta_num[q, r, t] /= 2.0
                            beta_num[r, q, t] = beta_num[q, r, t]
                else:
                    # enforce uniformity across t for identifiability
                    for e_idx in prange(_edge_vals.shape[0]):
                        i, j, t, _ = _edge_vals[e_idx, :]
                        i, j, t = int(i), int(j), int(t)
                        j_idx = np.flatnonzero(nbrs[t][i] == j)
                        # print(twopoint_edge_marg[t][i][j_idx, q, r])
                        # assert j_idx.sum() == 1
                        val = twopoint_edge_marg[t][i][j_idx, q, r][0]
                        # try:
                        #     assert not np.isnan(val)
                        # except:
                        #     print("(i,j,t):", i, j, t)
                        #     print("A[i,j,t] = ", a_ijt)
                        #     print("twopoint marg: ", val)
                        #     raise RuntimeError("Problem updating beta")
                        for tprime in range(T):
                            beta_num[q, r, tprime] += (
                                val / 2.0
                            )  # TODO: check if this should also be here

            if directed:
                for r in range(q):
                    for e_idx in prange(_edge_vals.shape[0]):
                        i, j, t, _ = _edge_vals[e_idx, :]
                        i, j, t = int(i), int(j), int(t)
                        j_idx = np.flatnonzero(nbrs[t][i] == j)
                        val = twopoint_edge_marg[t][i][j_idx, q, r][0]
                        beta_num[q, r, t] += val

        # beta_den = np.einsum("itq,it->qt", node_marg, degs)
        # beta_den = np.einsum("qt,rt->qrt", beta_den, beta_den)
        group_marg = np.zeros((Q, T))
        for q in range(Q):
            for t in range(T):
                group_marg[q, t] = node_marg[
                    :, t, q
                ].sum()  # TODO: again check this is right if directed
        for q in range(Q):
            for t in range(T):
                for r in range(q, Q):
                    if r != q:
                        beta_den[q, r, t] = group_marg[q, t] * group_marg[r, t]

                        if beta_den[q, r, t] < TOL:
                            beta_den[q, r, t] = 1.0
                        # if not directed: same either way (should this be?)
                        beta_den[r, q, t] = beta_den[q, r, t]
                    else:
                        for tprime in range(T):
                            # again enforce uniformity for identifiability
                            beta_den[q, r, t] += (
                                group_marg[q, tprime] * group_marg[r, tprime]
                            )

    # TODO: fix for case where beta_den very small (just consider using logs)
    # correct for numerical stability
    tmp = beta_num / beta_den
    for q in range(Q):
        for r in range(Q):
            for t in range(T):
                if tmp[q, r, t] < TOL:  # type: ignore
                    tmp[q, r, t] = TOL  # type: ignore
                elif tmp[q, r, t] > 1 - TOL:  # type: ignore
                    tmp[q, r, t] = 1 - TOL  # type: ignore
    if not init:
        tmp = learning_rate * tmp + (1 - learning_rate) * _beta
        tmp_diff = np.abs(tmp - _beta).mean()
        if np.isnan(tmp_diff):
            raise RuntimeError("Problem updating beta")
        if verbose:
            print("Beta diff:", np.round_(tmp_diff, 3))
        diff += tmp_diff
        _beta = tmp
    else:
        _beta = tmp
    return _beta, diff


@njit
def nb_update_meta_params(
    init,
    learning_rate,
    meta_types,
    N,
    T,
    Q,
    Z,
    X,
    _meta_params,
    _pres_nodes,
    node_marg,
    diff,
    verbose,
):
    # NB can't internally parallelise as need to aggregate
    # on diff, but would need to write entirely within this
    # fn to do so (less clear) - marginal benefit anyway
    # as typically S << N, T
    for s in range(len(meta_types)):
        # print(f"Updating params for {mt} dist")
        if meta_types[s] == "poisson":
            # print("In Poisson")
            _meta_params[s], diff = nb_update_poisson_meta(
                init,
                learning_rate,
                N,
                T,
                Q,
                Z,
                X[s],
                _meta_params[s],
                _pres_nodes,
                node_marg,
                diff,
                verbose,
            )
            if verbose:
                print("\tUpdated Poisson")
        elif meta_types[s] == "indep bernoulli":
            # print("In IB")
            _meta_params[s], diff = nb_update_indep_bern_meta(
                init,
                learning_rate,
                N,
                T,
                Q,
                Z,
                X[s],
                _meta_params[s],
                _pres_nodes,
                node_marg,
                diff,
                verbose,
            )
            if verbose:
                print("\tUpdated IB")
        elif meta_types[s] == "categorical":
            _meta_params[s], diff = nb_update_cat_meta(
                init,
                learning_rate,
                N,
                T,
                Q,
                Z,
                X[s],
                _meta_params[s],
                _pres_nodes,
                node_marg,
                diff,
                verbose,
            )
            if verbose:
                print("\tUpdated categorical")
        else:
            raise NotImplementedError(
                "Yet to implement metadata distribution of given type \nOptions are 'poisson', 'indep bernoulli', or 'categorical'"
            )  # NB can't use string formatting for print in numba
    return _meta_params, diff


@njit(parallel=True, fastmath=USE_FASTMATH)
def nb_update_poisson_meta(
    init,
    learning_rate,
    N,
    T,
    Q,
    Z,
    X_s,
    _mt_params,
    _pres_nodes,
    node_marg,
    diff,
    verbose,
):
    xi = np.ones((Q, T, 1))
    zeta = np.zeros((Q, T, 1))
    if init:
        # xi = np.array(
        #     [
        #         [(Z[:, t] == q).sum() for t in range(T)]
        #         for q in range(Q)
        #     ]
        # )
        for t in range(T):
            for i in prange(N):
                if _pres_nodes[i, t]:
                    xi[Z[i, t], t, 0] += 1
                    zeta[Z[i, t], t, 0] += X_s[i, t, 0]
            for q in range(Q):
                if xi[q, t, 0] < TOL:
                    xi[q, t, 0] = 1.0
                if zeta[q, t, 0] < TOL:
                    zeta[q, t, 0] = TOL
        # zeta = np.array(
        #     [
        #         [X[s][Z[:, t] == q, t, 0].sum() for t in range(T)]
        #         for q in range(Q)
        #     ]
        # )
        # gdb()
    else:
        for q in range(Q):
            for t in range(T):
                for i in prange(N):
                    if _pres_nodes[i, t]:
                        xi[q, t, 0] += node_marg[i, t, q]
        # zeta = np.einsum("itq,itd->qt", node_marg, X[s])
        for t in range(T):
            for i in prange(N):
                if _pres_nodes[i, t]:
                    for q in range(Q):
                        zeta[q, t, 0] += node_marg[i, t, q] * X_s[i, t, 0]
            for q in range(Q):
                if xi[q, t, 0] < TOL:
                    xi[q, t, 0] = 1.0
                if zeta[q, t, 0] < TOL:
                    zeta[q, t, 0] = TOL
    # NB again use relative error here as could be large
    # TODO: fix for case where xi small - rather than just setting as 1 when less than
    # 1e-50 (just consider using logs)
    tmp = zeta / xi
    for t in range(T):
        for q in range(Q):
            if tmp[q, t, 0] < TOL:
                tmp[q, t, 0] = TOL

    if not init:
        tmp = learning_rate * tmp + (1 - learning_rate) * _mt_params
        tmp_diff = np.abs((tmp - _mt_params) / _mt_params).mean()
        if np.isnan(tmp_diff):
            raise RuntimeError("Problem updating poisson params")
        if verbose:
            print("Poisson diff: ", np.round_(tmp_diff, 3))
        diff += tmp_diff
        _mt_params = tmp
    else:
        _mt_params = tmp
    return _mt_params, diff


@njit(parallel=True, fastmath=USE_FASTMATH)
def nb_update_indep_bern_meta(
    init,
    learning_rate,
    N,
    T,
    Q,
    Z,
    X_s,
    _mt_params,
    _pres_nodes,
    node_marg,
    diff,
    verbose,
):
    xi = np.ones((Q, T, 1))
    L = X_s.shape[-1]
    rho = np.zeros((Q, T, L))
    if init:
        # xi = np.array(
        #     [
        #         [(Z[:, t] == q).sum() for t in range(T)]
        #         for q in range(Q)
        #     ]
        # )
        for t in range(T):
            for i in prange(N):
                if _pres_nodes[i, t]:
                    xi[Z[i, t], t, 0] += 1
                    rho[Z[i, t], t, :] += X_s[i, t, :]
            for q in range(Q):
                if xi[q, t, 0] < TOL:
                    xi[q, t, 0] = 1.0
                for l_idx in range(L):
                    if rho[q, t, l_idx] < TOL:
                        rho[q, t, l_idx] = TOL

        # rho = np.array(
        #     [
        #         [
        #             X[s][Z[:, t] == q, t, :].sum(axis=0)
        #             for t in range(T)
        #         ]
        #         for q in range(Q)
        #     ]
        # )
    else:
        for t in range(T):
            for i in prange(N):
                if _pres_nodes[i, t]:
                    for q in range(Q):
                        xi[q, t, 0] += node_marg[i, t, q]
        # rho = np.einsum("itq,itl->qtl", node_marg, X[s])
        for t in range(T):
            for i in prange(N):
                if _pres_nodes[i, t]:
                    for q in range(Q):
                        rho[q, t, :] += node_marg[i, t, q] * X_s[i, t, :]
            for q in range(Q):
                if xi[q, t, 0] < TOL:
                    xi[q, t, 0] = 1.0
                for l_idx in range(L):
                    if rho[q, t, l_idx] < TOL:
                        rho[q, t, l_idx] = TOL
    # TODO: fix for xi very small (just use logs)
    tmp = rho / xi
    for q in range(Q):
        for t in range(T):
            for l_idx in range(L):
                if tmp[q, t, l_idx] < TOL:
                    tmp[q, t, l_idx] = TOL
                elif tmp[q, t, l_idx] > 1 - TOL:
                    tmp[q, t, l_idx] = 1 - TOL
    if not init:
        tmp = learning_rate * tmp + (1 - learning_rate) * _mt_params
        tmp_diff = np.abs(tmp - _mt_params).mean()
        if np.isnan(tmp_diff):
            raise RuntimeError("Problem updating IB params")
        if verbose:
            print("IB diff: ", np.round_(tmp_diff, 3))
        diff += tmp_diff
        _mt_params = tmp
    else:
        _mt_params = tmp
    return _mt_params, diff


@njit(parallel=True, fastmath=USE_FASTMATH)
def nb_update_cat_meta(
    init,
    learning_rate,
    N,
    T,
    Q,
    Z,
    X_s,
    _mt_params,
    _pres_nodes,
    node_marg,
    diff,
    verbose,
):
    L = X_s.shape[-1]
    rho = np.zeros((Q, T, L))
    if init:
        # xi = np.array(
        #     [
        #         [(Z[:, t] == q).sum() for t in range(T)]
        #         for q in range(Q)
        #     ]
        # )
        for t in range(T):
            for i in prange(N):
                if _pres_nodes[i, t]:
                    rho[Z[i, t], t, np.flatnonzero(X_s[i, t, :])] += 1
            for q in range(Q):
                for l_idx in range(L):
                    if rho[q, t, l_idx] < TOL:
                        rho[q, t, l_idx] = TOL

        # rho = np.array(
        #     [
        #         [
        #             X[s][Z[:, t] == q, t, :].sum(axis=0)
        #             for t in range(T)
        #         ]
        #         for q in range(Q)
        #     ]
        # )
    else:
        # rho = np.einsum("itq,itl->qtl", node_marg, X[s])
        for t in range(T):
            for i in prange(N):
                if _pres_nodes[i, t]:
                    for q in range(Q):
                        rho[q, t, np.flatnonzero(X_s[i, t, :])] += node_marg[i, t, q]
            for q in range(Q):
                for l_idx in range(L):
                    if rho[q, t, l_idx] < TOL:
                        rho[q, t, l_idx] = TOL
    for q in range(Q):
        for t in range(T):
            norm_rho = rho[q, t, :].sum()
            for l_idx in range(L):
                rho[q, t, l_idx] /= norm_rho
    for q in range(Q):
        for t in range(T):
            for l_idx in range(L):
                if rho[q, t, l_idx] < TOL:
                    rho[q, t, l_idx] = TOL
            norm_rho = rho[q, t, :].sum()
            for l_idx in range(L):
                if rho[q, t, l_idx] < TOL:
                    rho[q, t, l_idx] = TOL
    if not init:
        tmp = learning_rate * rho + (1 - learning_rate) * _mt_params
        tmp_diff = np.abs(tmp - _mt_params).mean()
        if np.isnan(tmp_diff):
            raise RuntimeError("Problem updating categorical params")
        if verbose:
            print("Categorical diff: ", np.round_(tmp_diff, 3))
        diff += tmp_diff
        _mt_params = tmp
    else:
        _mt_params = tmp
    return _mt_params, diff


@njit(parallel=True, fastmath=USE_FASTMATH)
def nb_compute_DC_lkl(_edge_vals, Q, degs, _lam):
    # Sort computation for all pres edges simultaneously (in DSBMM),
    # then just pass as matrix rather than computing on fly
    dc_lkl = np.zeros((_edge_vals.shape[0], Q, Q))
    for q in range(Q):
        for r in range(Q):
            for e_idx in prange(_edge_vals.shape[0]):
                i, j, t, a_ijt = _edge_vals[e_idx, :]
                i, j, t = int(i), int(j), int(t)
                dc_lkl[e_idx, q, r] = nb_poisson_lkl_int(
                    a_ijt, degs[i, t, 1] * degs[j, t, 0] * _lam[q, r, t]
                )
    return dc_lkl
