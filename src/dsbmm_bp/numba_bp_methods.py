# numba reimplementation of all methods for BP class that reasonably stand to gain from doing so (would allow parallelisation + GPU usage)
# - simply prepend method name with nb_
import numpy as np
import yaml  # type: ignore
from numba import njit, prange
from numba.typed import List

with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
TOL = config["tol"]  # min value permitted for msgs etc
# (for numerical stability)
LARGE_DEG_THR = config[
    "large_deg_thr"
]  # threshold of node degree above which log msgs calculated
# (for numerical stability)
RANDOM_ONLINE_UPDATE_MSG = config[
    "random_online_update_msg"
]  # if true then update messages online (always using most recent vals),
# otherwise update all messages simultaneously
USE_FASTMATH = config["use_fastmath"]
PLANTED_P = config["planted_p"]


@njit(parallel=True, fastmath=USE_FASTMATH)
def nb_init_msgs(
    mode,
    N,
    T,
    Q,
    nbrs,
    Z,
):
    if mode == "random":
        # initialise by random messages and marginals
        # _psi_e = [
        #     [
        #         np.random.rand(len(nbrs[t][i]), Q)
        #         if len(nbrs[t][i]) > 0
        #         else None
        #         for i in range(N)
        #     ]
        #     for t in range(T)
        # ]
        # _psi_e = [
        #     [
        #         msg / msg.sum(axis=1) if msg is not None else None
        #         for msg in _psi_e[t]
        #     ]
        #     for t in range(T)
        # ]
        ## INIT MARGINALS ##
        node_marg = np.random.rand(N, T, Q)
        marg_sums = node_marg.sum(axis=2)
        for q in range(Q):
            node_marg[:, :, q] /= marg_sums
        ## INIT MESSAGES ##
        tmp = List()
        for t in range(T):
            tmp2 = List()
            for i in range(N):
                n_nbrs = len(nbrs[t][i])
                if n_nbrs > 0:
                    msg = np.random.rand(n_nbrs, Q)
                    # msg /= msg.sum(axis=1)[:, np.newaxis]
                    msg_sums = msg.sum(axis=1)
                    for nbr_idx in range(n_nbrs):
                        msg[nbr_idx, :] /= msg_sums[nbr_idx]
                else:
                    msg = np.empty((1, Q), dtype=np.float64)
                # assert np.isnan(msg).sum() == 0
                tmp2.append(msg)
            # print("Trying to update psi_e")
            tmp.append(tmp2)
        _psi_e = tmp  # msgs from [t][i] to nbr j about i being in q (so 4d)
        # print("psi_e updated")
        # _psi_t = [
        #     [np.random.rand(2, Q) for i in range(N)]
        #     for t in range(T - 1)
        # ]
        # _psi_t = [
        #     [msg / msg.sum(axis=1) for msg in _psi_t[t]]
        #     for t in range(T - 1)
        # ]
        _psi_t = np.random.rand(N, T - 1, Q, 2)
        # _psi_t /= _psi_t.sum(axis=3)[:, :, :, np.newaxis, :]
        t_msg_sums = _psi_t.sum(axis=2)
        for q in range(Q):
            # msgs from i at t forwards/backwards
            _psi_t[:, :, q, :] /= t_msg_sums
        # assert np.isnan(_psi_t).sum() == 0
        # about being in group q,
        # so again 4d
        # assert np.all((_psi_t.sum(axis=3) - 1) ** 2 < 1e-14)
    elif mode == "partial":
        # initialise by partly planted partition plus some noise - others left random
        # see planted below for info on how planting considered
        ## INIT MARGINALS ##
        pass
        ## INIT MESSAGES ##
        pass
    elif mode == "planted":
        # initialise by given partition plus some random noise, with strength of info used
        # specified by plant_strength (shortened to ps below)
        # i.e. if z_0(i,t) = r,
        # \psi^{it}_q = \delta_{qr}(ps + (1 - ps)*rand) + (1 - \delta_{qr})*(1 - ps)*rand
        p = PLANTED_P
        ## INIT MARGINALS ##
        one_hot_Z = np.zeros((N, T, Q))
        for i in range(N):
            for t in range(T):
                one_hot_Z[i, t, Z[i, t]] = 1.0

        node_marg = p * one_hot_Z + (1 - p) * np.random.rand(N, T, Q)
        norm_sums = node_marg.sum(axis=2)
        for q in range(Q):
            node_marg[:, :, q] /= norm_sums

        ## INIT MESSAGES ##
        tmp = List()
        for t in range(T):
            tmp2 = List()
            for i in range(N):
                n_nbrs = len(nbrs[t][i])
                if n_nbrs > 0:
                    msg = (1 - p) * np.random.rand(n_nbrs, Q)
                    for nbr_idx in range(msg.shape[0]):
                        msg[nbr_idx, :] += p * one_hot_Z[i, t, :]
                        msg[nbr_idx, :] /= msg[nbr_idx, :].sum()
                else:
                    # print("WARNING: empty nodes not properly handled yet")
                    msg = np.empty((1, Q), dtype=np.float64)
                # assert np.isnan(msg).sum() == 0
                tmp2.append(msg)
            # print("Trying to update psi_e")
            tmp.append(tmp2)
        _psi_e = tmp

        _psi_t = (1 - p) * np.random.rand(N, T - 1, Q, 2)
        for t in range(T - 1):
            _psi_t[:, t, :, 0] += p * one_hot_Z[:, t + 1, :]
            _psi_t[:, t, :, 1] += p * one_hot_Z[:, t, :]
        t_msg_sums = _psi_t.sum(axis=2)
        try:
            for q in range(Q):
                _psi_t[:, :, q, :] /= t_msg_sums
        except Exception:  # ValueError:
            print(_psi_t[:, :, 0, :].shape, t_msg_sums.shape)
    return _psi_e, _psi_t, node_marg


@njit(fastmath=USE_FASTMATH)
def nb_forward_temp_msg_term(Q, trans_prob, i, t, _psi_t):
    # sum_qprime(trans_prob(qprime,q)*_psi_t[i,t-1,qprime,1])
    # from t-1 to t
    out = np.zeros(Q)
    try:
        # print(trans_prob)
        # print(trans_prob.T)
        for q in range(Q):
            for qprime in range(Q):
                out[q] += trans_prob[qprime, q] * _psi_t[i, t - 1, qprime, 1]
    except Exception:  # IndexError:
        # must have t=0 so t-1 outside of range, no forward message, but do have alpha instead - stopped now as need for backward term
        assert t == 0
        # out = model._alpha
    for q in range(Q):
        if out[q] < TOL:
            out[q] = TOL
    return out


@njit(fastmath=USE_FASTMATH)
def nb_backward_temp_msg_term(Q, T, trans_prob, i, t, _psi_t):
    """Backwards temporal message term for marginal of i at t, coming from i at t + 1
    Much as for spatial messages, by definition
        \\psi^{i(t)\to i(t+1)} \\propto \\psi^{it} / \\psi^{i(t+1)\to i(t)}
    so we can use this term to update forward temporal messages to t + 1 if t < T

    Args:
        i (_type_): _description_
        t (_type_): _description_

    Raises:
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """
    # sum_qprime(trans_prob(q,qprime)*_psi_t[i,t,qprime,0])
    # from t+1 to t
    try:
        out = np.zeros(Q)
        for q in range(Q):
            for qprime in range(Q):
                out[q] += trans_prob[q, qprime] * _psi_t[i, t, qprime, 0]
        for q in range(Q):
            if out[q] < TOL:
                out[q] = TOL
    except Exception:  # IndexError:
        # t=T outside of range, so no backward message
        assert t == T - 1
        out = np.ones(Q)
    return np.ascontiguousarray(out)


@njit(fastmath=USE_FASTMATH)
def nb_spatial_msg_term_small_deg(
    Q,
    nbrs,
    e_nbrs_inv,
    nbrs_inv,
    deg_corr,
    degs,
    i,
    t,
    dc_lkl,
    _h,
    meta_prob,
    block_edge_prob,
    _psi_e,
):
    """For node i with degree smaller than threshold within timestep t, and
    neighbours nbrs, calculate the spatial message term for the marginal distribution of i,
    updating spatial messages \\psi^{i\to j}(t) and external field h(t) in the process.

    Updating together rational as by definition,
        \\psi^{i\to j}(t) \\propto \\psi^{it}/\\psi^{j\to i}(t),
    hence we can
        (i) Calculate all terms involved in unnormalised node marginal / node messages
        (ii) Calculate product of these (for unnorm marginals), divide by term involving j
                for \\psi^{i\to j}(t) for unnorm messages
        (iii) Calculate normalisation for node marginals, update marginals
        (iv) Calculate normalisation for node messages, update messages

    As there are \\Oh(Q d_i) terms only involved for each q (so \\Oh(Q^2 d_i) total),
    for sparse networks where the average degree d ~ \\Oh(1), updating all messages
    and marginals together is an \\Oh(Q^2 N T d) process. As typically Q ~ log(N),
    this means approximate complexity of \\Oh(2 N T d log(N)), so roughly linear in the
    number of nodes - a significant improvement to quadratic complexity of mean-field VI!

    Args:
        i (_type_): node to update
        t (_type_): timestep
        nbrs (_type_): neighbours of i in network at time t

    Raises:
        RuntimeError: _description_

    Returns:
        msg (_type_): spatial message term of total node marginal for i at t
        field_iter (_type_): len(nbrs) x Q array containing value of term corresponding to
                                each nbr so can update corresponding message
    """
    beta = block_edge_prob[:, :, t]

    # print("Beta:", beta)
    # sum_terms = np.array(
    #     [
    #         block_edge_prob[:, :, t].T
    #         @ _psi_e[t][j][nbrs[t][j] == i]
    #         for j in nbrs
    #     ]
    # )  # |N_i| x Q
    # np.prod(sum_terms, axis=0) # can't use axis kwarg for prod in numba
    # for sum_term in sum_terms:
    #     msg *= sum_term

    msg = np.ones((Q,))
    field_iter = np.zeros((len(nbrs), Q))
    for nbr_idx in range(nbrs.shape[0]):
        j = nbrs[nbr_idx]
        idx = nbrs_inv[nbr_idx]
        jtoi_msgs = _psi_e[t][j][idx, :].reshape(
            -1
        )  # for whatever reason this stays 2d, so need to flatten first
        tmp = np.zeros(Q)
        if deg_corr:
            for q in range(Q):
                for r in range(Q):
                    tmp[q] += dc_lkl[e_nbrs_inv[nbr_idx], r, q] * jtoi_msgs[r]
        else:
            for q in range(Q):
                for r in range(Q):
                    tmp[q] += beta[r, q] * jtoi_msgs[r]
        # for q in range(Q):
        #     if tmp[q] < TOL:
        #         tmp[q] = TOL
        # try:
        #     assert not np.isnan(tmp).sum() > 0
        #     assert not np.isinf(tmp).sum() > 0
        # except:
        #     # print("A[t]:", A[t])
        #     print("(i,j,t):", i, j, t)
        #     print("deg[i,t]:", len(nbrs))
        #     print("jtoi:", jtoi_msgs)
        #     print("full j msgs:", _psi_e[t][j])
        #     print("Beta:", beta)
        #     print("tmp:", tmp)
        #     print("spatial msg term:", msg)
        #     raise RuntimeError("Problem with field iter term")
        field_iter[nbr_idx, :] = tmp
        # print("summed:", tmp.shape)
        for q in range(Q):
            msg[q] *= tmp[q]
        # try:
        #     assert not np.all(msg < TOL)
        # except:
        #     print("(i,j,t):", i, j, t)
        #     print("deg[i,t]:", len(nbrs))
        #     print("jtoi:", jtoi_msgs)
        #     print("Beta:", beta)
        #     print("tmp:", tmp)
        #     print("spatial msg term:", msg)
        #     raise RuntimeError("Problem vanishing spatial msg term")
    if deg_corr:
        msg *= np.exp(-degs[i, t, 1] * _h[:, t])
    else:
        msg *= np.exp(-1.0 * _h[:, t])
    msg *= meta_prob[i, t, :]
    # try:
    #     assert not np.isinf(msg).sum() > 0
    # except:
    #     print("(i,t):", i, t)
    #     print("deg[i,t]:", len(nbrs))
    #     print("beta:", beta)
    #     print("meta:", meta_prob(i, t))
    #     print("exp(-h):", np.exp(-1.0 * _h[:, t]))
    #     print("spatial msg term:", msg)
    #     raise RuntimeError("Problem with either meta or external field terms")
    # msg[msg < TOL] = TOL
    return msg, field_iter


@njit(fastmath=USE_FASTMATH)
def nb_spatial_msg_term_large_deg(
    Q,
    nbrs,
    e_nbrs_inv,
    nbrs_inv,
    deg_corr,
    degs,
    i,
    t,
    dc_lkl,
    _h,
    meta_prob,
    block_edge_prob,
    _psi_e,
):
    """Same as spatial_msg_term_small_deg but for node i that has degree within timestep t
    larger than specified threshold - basically just handle logs + subtract max value
    before exponentiating and normalising instead for numerical stability

    Args:
        i (_type_): _description_
        t (_type_): _description_
        nbrs (_type_): _description_

    Returns:
        _type_: _description_
    """
    beta = block_edge_prob[:, :, t]
    # deg_i = len(nbrs)
    msg = np.zeros((Q,))
    log_field_iter = np.zeros((len(nbrs), Q))
    # debug option: make logger that tracks this
    # print("Large deg version used")
    max_log_msg = -1000000.0
    for nbr_idx in range(nbrs.shape[0]):
        j = nbrs[nbr_idx]
        e_idx = e_nbrs_inv[nbr_idx]
        idx = nbrs_inv[nbr_idx]
        jtoi_msgs = _psi_e[t][j][idx, :].reshape(
            -1
        )  # for whatever reason this stays 2d, so need to flatten first
        # print("jtoi_msg:", jtoi_msgs.shape)
        # tmp = np.log(
        #     np.ascontiguousarray(beta.T) @ np.ascontiguousarray(jtoi_msgs)
        # )
        tmp = np.zeros(Q)
        if deg_corr:
            for q in range(Q):
                for r in range(Q):
                    tmp[q] += dc_lkl[e_idx, r, q] * jtoi_msgs[r]
        else:
            for q in range(Q):
                for r in range(Q):
                    tmp[q] += beta[r, q] * jtoi_msgs[r]
                if tmp[q] < TOL:
                    tmp[q] = TOL

        tmp = np.log(tmp)

        # try:
        #     assert np.isinf(tmp).sum() == 0
        # except:
        #     print("jtoi msgs:", jtoi_msgs)
        #     print("i,j,t:", i, j, t)
        #     print("beta:", beta)
        #     raise RuntimeError("Problem w large deg spatial msg")
        log_field_iter[nbr_idx, :] = tmp
        # print("summed:", tmp.shape)
        msg += tmp
        max_msg_log = msg.max()
        if max_msg_log > max_log_msg:
            max_log_msg = max_msg_log
    if deg_corr:
        msg -= _h[:, t] * degs[i, t, 1]
    else:
        msg -= _h[:, t]  # NB don't need / N as using p_ab to calc, not c_ab
    msg += np.log(meta_prob[i, t, :])
    return msg, max_log_msg, log_field_iter


@njit(parallel=True, fastmath=USE_FASTMATH)
def nb_init_h(
    N,
    T,
    Q,
    degs,
    deg_corr,
    block_edge_prob,
    node_marg,
):
    # update within each timestep is unchanged from static case,
    # i.e. = \sum_r \sum_i \psi_r^{it} p_{rq}^t
    # _h = np.einsum("itr,rqt->qt", node_marg, block_edge_prob)
    # _h = (
    #     block_edge_prob.transpose(1, 0, 2) * node_marg.sum(axis=0).T
    # ).sum(axis=1)
    _h = np.zeros((Q, T))
    if deg_corr:
        for q in prange(Q):
            for t in range(T):
                for i in range(N):
                    for r in range(Q):
                        _h[q, t] += (
                            block_edge_prob[r, q, t]
                            * node_marg[i, t, r]
                            * degs[i, t, 1]
                        )
    else:
        for q in prange(Q):
            for t in range(T):
                for i in range(N):
                    for r in range(Q):
                        _h[q, t] += block_edge_prob[r, q, t] * node_marg[i, t, r]
    # print("h after init:", _h)
    return _h


@njit(parallel=True, fastmath=USE_FASTMATH)
def nb_update_h(Q, sign, i, t, degs, deg_corr, block_edge_prob, _h, node_marg):
    # _h[:, t] += (
    #     sign
    #     * np.ascontiguousarray(block_edge_prob[:, :, t].T)
    #     @ np.ascontiguousarray(node_marg[i, t, :])
    # )
    if deg_corr:
        for q in range(Q):
            for r in range(Q):
                _h[q, t] += (
                    sign * block_edge_prob[r, q, t] * node_marg[i, t, r] * degs[i, t, 1]
                )
    else:
        # try:
        #     assert np.isnan(node_marg[i, t, :]).sum() == 0
        # except:
        #     print("i, t:", i, t)
        #     print("node_marg:", node_marg[i, t, :])
        #     raise ValueError("Problem with node marg")
        for q in range(Q):
            for r in range(Q):
                _h[q, t] += sign * block_edge_prob[r, q, t] * node_marg[i, t, r]


@njit(parallel=True, fastmath=USE_FASTMATH)
def nb_update_node_marg(
    N,
    T,
    Q,
    deg_corr,
    degs,
    _pres_nodes,
    _pres_trans,
    all_nbrs,
    nbrs_inv,
    e_nbrs_inv,
    n_msgs,
    block_edge_prob,
    trans_prob,
    dc_lkl,
    _h,
    meta_prob,
    _alpha,
    node_marg,
    _psi_e,
    _psi_t,
    msg_diff,
):
    """Update all node marginals (now synchronously to
    avoid race condition), simultaneously updating
    messages and external fields h(t) - process is
    as follows:
        (i) Determine whether large or small degree version of spatial message updates
            should be used
        (ii) Subtract old marginals for i from external field h(t)
        (iii) Update spatial messages while calculating spatial term
        (iv) Update forwards temporal messages from i at t while calculating backward temp
                term from i at t + 1
        (v) Update backwards temporal messages from i at t while calculating forward temp term
            from i at t - 1
        (vi) Update marginals for i at t
        (vii) Add new marginals for i to external field h(t)
    """
    node_update_order = np.arange(N)
    time_update_order = np.arange(T)
    tmp = List()
    for t in range(T):
        tmp2 = List()
        for i in range(N):
            n_nbrs = len(all_nbrs[t][i])
            if n_nbrs > 0:
                msg = np.empty((n_nbrs, Q), dtype=np.float64)
                # msg /= msg.sum(axis=1)[:, np.newaxis]
                # msg_sums = msg.sum(axis=1)
                # for nbr_idx in range(n_nbrs):
                #     msg[nbr_idx, :] /= msg_sums[nbr_idx]
            else:
                msg = np.empty((1, Q), dtype=np.float64)
            # assert np.isnan(msg).sum() == 0
            tmp2.append(msg)
        # print("Trying to update psi_e")
        tmp.append(tmp2)
    _new_psi_e = tmp
    _new_psi_t = np.empty((N, T - 1, Q, 2))
    new_node_marg = np.empty((N, T, Q))
    # if RANDOM_ONLINE_UPDATE_MSG:
    #     np.random.shuffle(node_update_order)
    #     np.random.shuffle(time_update_order)
    # with np.errstate(all="ignore"): # not usable in numba
    # necessary as LLVM optimisation seems to result in spurious DividebyZero errors which aren't possible given checks
    h_diff = np.zeros((Q, T))
    msg_diff = 0.0
    for iup_idx in prange(N):
        i = node_update_order[iup_idx]
        for tup_idx in range(T):
            t = time_update_order[tup_idx]
            # TODO: Rather than looping over all N for every T then checking for presence, setup in same way as for edges
            # where directly loop over present nodes only - just need to change how shuffling is done
            if _pres_nodes[i, t]:
                nbrs = all_nbrs[t][i]
                deg_i = len(nbrs)
                if deg_i > 0:
                    spatial_msg_term = np.empty(Q)
                    if deg_i < LARGE_DEG_THR:
                        field_iter = np.empty((deg_i, Q))
                        (spatial_msg_term, field_iter,) = nb_spatial_msg_term_small_deg(
                            Q,
                            nbrs,
                            e_nbrs_inv[t][i],
                            nbrs_inv[t][i],
                            deg_corr,
                            degs,
                            i,
                            t,
                            dc_lkl,
                            _h,
                            meta_prob,
                            block_edge_prob,
                            _psi_e,
                        )
                        # now terms using current h have been calculated, subtract old marg vals
                        # nb_update_h(
                        #    ...
                        # ) # Possible problem using when parallel, just write out instead
                        if deg_corr:
                            for q in range(Q):
                                for r in range(Q):
                                    h_diff[q, t] -= (
                                        block_edge_prob[r, q, t]
                                        * node_marg[i, t, r]
                                        * degs[i, t, 1]
                                    )
                        else:
                            for q in range(Q):
                                for r in range(Q):
                                    h_diff[q, t] -= (
                                        block_edge_prob[r, q, t] * node_marg[i, t, r]
                                    )
                        tmp = spatial_msg_term.copy()
                        if t == 0:
                            for q in range(Q):
                                tmp[q] *= _alpha[q]
                        back_term = np.ones(Q)
                        if t < T - 1:
                            if _pres_trans[i, t]:
                                back_term = nb_backward_temp_msg_term(
                                    Q, T, trans_prob, i, t, _psi_t
                                )
                                # REMOVE CHECK AFTER FIX
                                if np.isnan(back_term).sum() > 0:
                                    print("i,t:", i, t)
                                    print("back:", back_term)
                                    raise RuntimeError("Problem w back term")
                                for q in range(Q):
                                    tmp[q] *= back_term[q]
                        ## UPDATE BACKWARDS MESSAGES FROM i AT t ##
                        forward_term = np.ones(Q)
                        if t > 0:
                            if _pres_trans[i, t - 1]:
                                tmp_backwards_msg = tmp.copy()
                                if tmp_backwards_msg.sum() > 0:
                                    tmp_backwards_msg /= tmp_backwards_msg.sum()
                                for q in range(Q):
                                    if tmp_backwards_msg[q] < TOL:
                                        tmp_backwards_msg[q] = TOL
                                if tmp_backwards_msg.sum() > 0:
                                    tmp_backwards_msg /= tmp_backwards_msg.sum()
                                msg_diff += (
                                    np.abs(
                                        tmp_backwards_msg - _psi_t[i, t - 1, :, 0]
                                    ).mean()
                                    / n_msgs
                                )
                                _new_psi_t[i, t - 1, :, 0] = tmp_backwards_msg
                                forward_term = nb_forward_temp_msg_term(
                                    Q, trans_prob, i, t, _psi_t
                                )
                            else:
                                # node present at t but not t-1, use alpha instead
                                forward_term = _alpha.copy()
                            for q in range(Q):
                                tmp[q] *= forward_term[q]
                        ## UPDATE SPATIAL MESSAGES FROM i AT t ##
                        # tmp_spatial_msg = (
                        #     np.expand_dims(tmp, 0) / field_iter
                        # )  # can't use as problem if field_iter << 1
                        tmp_spatial_msg = np.ones((deg_i, Q))
                        for nbr_idx in range(deg_i):
                            for q in range(Q):
                                if field_iter[nbr_idx, q] > TOL:
                                    tmp_spatial_msg[nbr_idx, q] = (
                                        tmp[q] / field_iter[nbr_idx, q]
                                    )
                                else:
                                    # too small for stable div, construct
                                    # directly instead
                                    tmp_loc = back_term[q] * forward_term[q]
                                    for k in range(deg_i):
                                        if k != nbr_idx:
                                            tmp_loc *= field_iter[k, q]
                                    tmp_spatial_msg[nbr_idx, q] = tmp_loc
                        tmp_spat_sums = np.empty(deg_i)
                        for nbr_idx in range(deg_i):
                            tmp_spat_sums[nbr_idx] = tmp_spatial_msg[nbr_idx, :].sum()
                        for nbr_idx in range(deg_i):
                            if tmp_spat_sums[nbr_idx] > 0:
                                for q in range(Q):
                                    tmp_spatial_msg[nbr_idx, q] /= tmp_spat_sums[
                                        nbr_idx
                                    ]
                        for nbr_idx in range(deg_i):
                            for q in range(Q):
                                if tmp_spatial_msg[nbr_idx, q] < TOL:
                                    tmp_spatial_msg[nbr_idx, q] = TOL
                        for nbr_idx in range(deg_i):
                            tmp_spat_sums[nbr_idx] = tmp_spatial_msg[nbr_idx, :].sum()
                        for nbr_idx in range(deg_i):
                            if tmp_spat_sums[nbr_idx] > 0:
                                for q in range(Q):
                                    tmp_spatial_msg[nbr_idx, q] /= tmp_spat_sums[
                                        nbr_idx
                                    ]
                        for nbr_idx in range(deg_i):
                            msg_diff += (
                                np.abs(
                                    tmp_spatial_msg[nbr_idx, :]
                                    - _psi_e[t][i][nbr_idx, :]
                                ).mean()
                                * deg_i
                                / n_msgs
                            )  # NB need to mult by deg_i so weighted correctly
                        _new_psi_e[t][i] = tmp_spatial_msg
                        ## UPDATE FORWARDS MESSAGES FROM i AT t ##
                        if t < T - 1 and _pres_trans[i, t]:
                            tmp_forwards_msg = spatial_msg_term
                            if t > 0:
                                for q in range(Q):
                                    tmp_forwards_msg[q] *= forward_term[q]
                                if tmp_forwards_msg.sum() > 0:
                                    tmp_forwards_msg /= tmp_forwards_msg.sum()
                                for q in range(Q):
                                    if tmp_forwards_msg[q] < TOL:
                                        tmp_forwards_msg[q] = TOL
                                if tmp_forwards_msg.sum() > 0:
                                    tmp_forwards_msg /= tmp_forwards_msg.sum()
                                msg_diff += (
                                    np.abs(tmp_forwards_msg - _psi_t[i, t, :, 1]).mean()
                                    / n_msgs
                                )
                                _new_psi_t[i, t, :, 1] = tmp_forwards_msg
                            ## UPDATE MARGINAL OF i AT t ##
                            tmp_marg = tmp
                            if tmp_marg.sum() > 0:
                                tmp_marg /= tmp_marg.sum()
                            for q in range(Q):
                                if tmp_marg[q] < TOL:
                                    tmp_marg[q] = TOL
                            if tmp_marg.sum() > 0:
                                tmp_marg /= tmp_marg.sum()
                            new_node_marg[i, t, :] = tmp_marg

                        else:
                            log_field_iter = np.empty((deg_i, Q))
                            (
                                spatial_msg_term,
                                max_log_spatial_msg_term,
                                log_field_iter,
                            ) = nb_spatial_msg_term_large_deg(
                                Q,
                                nbrs,
                                e_nbrs_inv[t][i],
                                nbrs_inv[t][i],
                                deg_corr,
                                degs,
                                i,
                                t,
                                dc_lkl,
                                _h,
                                meta_prob,
                                block_edge_prob,
                                _psi_e,
                            )
                            if deg_corr:
                                for q in range(Q):
                                    for r in range(Q):
                                        h_diff[q, t] -= (
                                            block_edge_prob[r, q, t]
                                            * node_marg[i, t, r]
                                            * degs[i, t, 1]
                                        )
                            else:
                                for q in range(Q):
                                    for r in range(Q):
                                        h_diff[q, t] -= (
                                            block_edge_prob[r, q, t]
                                            * node_marg[i, t, r]
                                        )
                            tmp = spatial_msg_term.copy()
                            if t == 0:
                                for q in range(Q):
                                    tmp[q] += np.log(_alpha[q])
                            back_term = np.zeros(Q)
                            if t < T - 1:
                                if _pres_trans[i, t]:
                                    back_term = np.log(
                                        nb_backward_temp_msg_term(
                                            Q, T, trans_prob, i, t, _psi_t
                                        )
                                    )
                                for q in range(Q):
                                    tmp[q] += back_term[q]
                            ## UPDATE BACKWARDS MESSAGES FROM i AT t ##
                            forward_term = np.zeros(Q)
                            if t > 0:
                                if _pres_trans[i, t - 1]:
                                    tmp_backwards_msg = np.exp(
                                        tmp - max_log_spatial_msg_term
                                    )
                                    if tmp_backwards_msg.sum() > 0:
                                        tmp_backwards_msg /= tmp_backwards_msg.sum()
                                    for q in range(Q):
                                        if tmp_backwards_msg[q] < TOL:
                                            tmp_backwards_msg[q] = TOL
                                    # tmp_backwards_msg[tmp_backwards_msg > 1 - TOL] = TOL
                                    if tmp_backwards_msg.sum() > 0:
                                        tmp_backwards_msg /= tmp_backwards_msg.sum()
                                    msg_diff += (
                                        np.abs(
                                            tmp_backwards_msg - _psi_t[i, t - 1, :, 0]
                                        ).mean()
                                        / n_msgs
                                    )
                                    _new_psi_t[i, t - 1, :, 0] = tmp_backwards_msg
                                    forward_term = np.log(
                                        nb_forward_temp_msg_term(
                                            Q, trans_prob, i, t, _psi_t
                                        )
                                    )
                                else:
                                    # node present at t but not t-1, so use alpha instead
                                    forward_term = np.log(_alpha)
                                for q in range(Q):
                                    tmp[q] += forward_term[q]
                            ## UPDATE SPATIAL MESSAGES FROM i AT t ##
                            tmp_spatial_msg = -1.0 * log_field_iter.copy()
                            for nbr_idx in range(deg_i):
                                for q in range(Q):
                                    tmp_spatial_msg[nbr_idx, q] += tmp[q]
                            log_field_iter_max = np.empty(deg_i)
                            for nbr_idx in range(deg_i):
                                log_field_iter_max[nbr_idx] = tmp_spatial_msg[
                                    nbr_idx, :
                                ].max()
                            for nbr_idx in range(deg_i):
                                for q in range(Q):
                                    tmp_spatial_msg[nbr_idx, q] = np.exp(
                                        tmp_spatial_msg[nbr_idx, q]
                                        - log_field_iter_max[nbr_idx]
                                    )
                            tmp_spat_sums = np.empty(deg_i)
                            for nbr_idx in range(deg_i):
                                tmp_spat_sums[nbr_idx] = tmp_spatial_msg[
                                    nbr_idx, :
                                ].sum()
                            for nbr_idx in range(deg_i):
                                if tmp_spat_sums[nbr_idx] > 0:
                                    for q in range(Q):
                                        tmp_spatial_msg[nbr_idx, q] /= tmp_spat_sums[
                                            nbr_idx
                                        ]
                            for nbr_idx in range(deg_i):
                                for q in range(Q):
                                    if tmp_spatial_msg[nbr_idx, q] < TOL:
                                        tmp_spatial_msg[nbr_idx, q] = TOL
                            for nbr_idx in range(deg_i):
                                tmp_spat_sums[nbr_idx] = tmp_spatial_msg[
                                    nbr_idx, :
                                ].sum()
                            for nbr_idx in range(deg_i):
                                if tmp_spat_sums[nbr_idx] > 0:
                                    for q in range(Q):
                                        tmp_spatial_msg[nbr_idx, q] /= tmp_spat_sums[
                                            nbr_idx
                                        ]
                            for nbr_idx in range(deg_i):
                                msg_diff += (
                                    np.abs(
                                        tmp_spatial_msg[nbr_idx, :]
                                        - _psi_e[t][i][nbr_idx, :]
                                    ).mean()
                                    * deg_i
                                    / n_msgs
                                )  # NB need to mult by deg_i so weighted correctly
                            _new_psi_e[t][i] = tmp_spatial_msg
                            # ## UPDATE FORWARDS MESSAGES FROM i AT t ##
                            if t < T - 1:
                                if _pres_trans[i, t]:
                                    tmp_forwards_msg = np.empty(Q)
                                    for q in range(Q):
                                        tmp_forwards_msg[q] = np.exp(
                                            tmp[q]
                                            - back_term[q]
                                            - max_log_spatial_msg_term
                                        )
                                    if tmp_forwards_msg.sum() > 0:
                                        tmp_forwards_msg /= tmp_forwards_msg.sum()
                                    for q in range(Q):
                                        if tmp_forwards_msg[q] < TOL:
                                            tmp_forwards_msg[q] = TOL
                                    if tmp_forwards_msg.sum() > 0:
                                        tmp_forwards_msg /= tmp_forwards_msg.sum()
                                    msg_diff += (
                                        np.abs(
                                            tmp_forwards_msg - _psi_t[i, t, :, 1]
                                        ).mean()
                                        / n_msgs
                                    )
                                    _new_psi_t[i, t, :, 1] = tmp_forwards_msg
                            # ## UPDATE MARGINAL OF i AT t ##
                            tmp_marg = np.empty(Q)
                            for q in range(Q):
                                tmp_marg[q] = np.exp(tmp[q] - max_log_spatial_msg_term)
                            if tmp_marg.sum() > 0:
                                tmp_marg /= tmp_marg.sum()
                            for q in range(Q):
                                if tmp_marg[q] < TOL:
                                    tmp_marg[q] = TOL
                            if tmp_marg.sum() > 0:
                                tmp_marg /= tmp_marg.sum()
                            new_node_marg[i, t, :] = tmp_marg
                            # GOOD TO HERE
                    # update h with new values
                    if deg_corr:
                        for q in range(Q):
                            for r in range(Q):
                                h_diff[q, t] += (
                                    block_edge_prob[r, q, t]
                                    * node_marg[i, t, r]
                                    * degs[i, t, 1]
                                )
                    else:
                        for q in range(Q):
                            for r in range(Q):
                                h_diff[q, t] += (
                                    block_edge_prob[r, q, t] * node_marg[i, t, r]
                                )
            else:
                new_node_marg[i, t, :] = 0.0

    if np.isnan(msg_diff):
        for i in range(N):
            for t in range(T - 1):
                if np.isnan(new_node_marg[i, t, :]).sum() > 0:
                    print("nans for node marg @ (i,t)=", i, t)
                if np.isnan(_new_psi_e[t][i][:, :]).sum() > 0:
                    print("nans for spatial msgs @ (i,t)=", i, t)
                if np.isnan(_new_psi_t[i, t, :, :]).sum() > 0:
                    print("nans for temp marg @ (i,t)=", i, t)
            if np.isnan(new_node_marg[i, T - 1, :]).sum() > 0:
                print("nans for node marg @ (i,t)=", i, T - 1)
            if np.isnan(_new_psi_e[T - 1][i][:, :]).sum() > 0:
                print("nans for spatial msgs @ (i,t)=", i, T - 1)
        raise RuntimeError("Problem updating messages")
    _h += h_diff
    return new_node_marg, _new_psi_e, _new_psi_t, msg_diff


@njit(parallel=True, fastmath=USE_FASTMATH, error_model="numpy")
def nb_compute_free_energy(
    N,
    T,
    Q,
    deg_corr,
    degs,
    _pres_nodes,
    _pres_trans,
    all_nbrs,
    nbrs_inv,
    e_nbrs_inv,
    block_edge_prob,
    trans_prob,
    dc_lkl,
    _h,
    meta_prob,
    _alpha,
    _beta,
    _pi,
    _psi_e,
    _psi_t,
):
    f_site = 0.0
    # TODO: fix along numpy lines
    f_link = 0.0
    last_term = 0.0
    for i in prange(N):
        for t in range(T):
            if _pres_nodes[i, t]:
                nbrs = all_nbrs[t][i]
                deg_i = len(nbrs)
                if deg_i > 0:
                    if deg_i < LARGE_DEG_THR:
                        (spatial_msg_term, field_iter,) = nb_spatial_msg_term_small_deg(
                            Q,
                            nbrs,
                            e_nbrs_inv[t][i],
                            nbrs_inv[t][i],
                            deg_corr,
                            degs,
                            i,
                            t,
                            dc_lkl,
                            _h,
                            meta_prob,
                            block_edge_prob,
                            _psi_e,
                        )
                        tmp = spatial_msg_term.copy()
                        if t == 0:
                            tmp *= _alpha
                        back_term = np.ones(Q)
                        if t < T - 1:
                            if _pres_trans[i, t]:
                                back_term = nb_backward_temp_msg_term(
                                    Q, T, trans_prob, i, t, _psi_t
                                )
                                # REMOVE CHECK AFTER FIX
                                if np.isnan(back_term).sum() > 0:
                                    print("i,t:", i, t)
                                    print("back:", back_term)
                                    raise RuntimeError("Problem w back term")
                                tmp *= back_term
                        forward_term = np.ones(Q)
                        if t > 0:
                            if _pres_trans[i, t - 1]:
                                # add back message to f_link
                                f_link += np.log(tmp.sum())
                                forward_term = nb_forward_temp_msg_term(
                                    Q, trans_prob, i, t, _psi_t
                                )
                            else:
                                # node present at t but not t-1, use alpha instead
                                forward_term = _alpha.copy()
                            tmp *= forward_term
                        tmp_spatial_msg = np.ones((deg_i, Q))
                        for nbr_idx in range(deg_i):
                            for q in range(Q):
                                if field_iter[nbr_idx, q] > TOL:
                                    tmp_spatial_msg[nbr_idx, q] = (
                                        tmp[q] / field_iter[nbr_idx, q]
                                    )
                                else:
                                    # too small for stable div, construct
                                    # directly instead
                                    tmp_loc = back_term[q] * forward_term[q]
                                    for k in range(deg_i):
                                        if k != nbr_idx:
                                            tmp_loc *= field_iter[k, q]
                                    tmp_spatial_msg[nbr_idx, q] = tmp_loc
                        tmp_spat_sums = np.empty(deg_i)
                        for nbr_idx in range(deg_i):
                            tmp_spat_sums[nbr_idx] = tmp_spatial_msg[nbr_idx, :].sum()
                        for nbr_idx in range(deg_i):
                            # add spatial messages to f_link
                            f_link += np.log(tmp_spat_sums[nbr_idx])
                        if t < T - 1 and _pres_trans[i, t]:
                            # add forwards messages to f_link
                            tmp_forwards_msg = spatial_msg_term
                            if t > 0:
                                tmp_forwards_msg *= forward_term
                            f_link += np.log(tmp_forwards_msg.sum())
                        # add marg to f_site
                        f_site += np.log(tmp.sum())
                    else:
                        (
                            spatial_msg_term,
                            max_log_spatial_msg_term,
                            log_field_iter,
                        ) = nb_spatial_msg_term_large_deg(
                            Q,
                            nbrs,
                            e_nbrs_inv[t][i],
                            nbrs_inv[t][i],
                            deg_corr,
                            degs,
                            i,
                            t,
                            dc_lkl,
                            _h,
                            meta_prob,
                            block_edge_prob,
                            _psi_e,
                        )
                        tmp = spatial_msg_term
                        if t == 0:
                            tmp += np.log(_alpha)
                        back_term = np.zeros(Q)
                        if t < T - 1:
                            if _pres_trans[i, t]:
                                back_term = np.log(
                                    nb_backward_temp_msg_term(
                                        Q, T, trans_prob, i, t, _psi_t
                                    )
                                )
                            tmp += back_term
                        forward_term = np.zeros(Q)
                        if t > 0:
                            if _pres_trans[i, t - 1]:
                                tmp_backwards_msg = np.exp(
                                    tmp - max_log_spatial_msg_term
                                )
                                # add backwards msg to f_link
                                f_link += np.log(tmp_backwards_msg.sum())

                                forward_term = np.log(
                                    nb_forward_temp_msg_term(
                                        Q, trans_prob, i, t, _psi_t
                                    )
                                )
                            else:
                                # node present at t but not t-1, so use alpha instead
                                forward_term = np.log(_alpha)
                            tmp += forward_term
                        tmp_spatial_msg = -1.0 * log_field_iter.copy()
                        for nbr_idx in range(deg_i):
                            for q in range(Q):
                                tmp_spatial_msg[nbr_idx, q] += tmp[q]
                        log_field_iter_max = np.empty(deg_i)
                        for nbr_idx in range(deg_i):
                            log_field_iter_max[nbr_idx] = tmp_spatial_msg[
                                nbr_idx, :
                            ].max()
                        for nbr_idx in range(deg_i):
                            tmp_spatial_msg[nbr_idx, :] = np.exp(
                                tmp_spatial_msg[nbr_idx, :]
                                - log_field_iter_max[nbr_idx]
                            )
                        tmp_spat_sums = np.empty(deg_i)
                        for nbr_idx in range(deg_i):
                            tmp_spat_sums[nbr_idx] = tmp_spatial_msg[nbr_idx, :].sum()
                        # add spatial msgs to f_link
                        for nbr_idx in range(deg_i):
                            f_link += np.log(tmp_spat_sums[nbr_idx])

                        # add forwards msg to f_link
                        if t < T - 1:
                            if _pres_trans[i, t]:
                                tmp_forwards_msg = np.exp(
                                    tmp - max_log_spatial_msg_term - back_term
                                )
                                f_link += np.log(tmp_forwards_msg.sum())
                        # add marg to f_site
                        tmp_marg = np.exp(tmp - max_log_spatial_msg_term)
                        f_site += np.log(tmp_marg.sum())
                else:
                    # print("WARNING: disconnected nodes not yet handled properly")
                    # print("i,t:", i, t)
                    raise RuntimeError(
                        "Problem with measuring presence - deg = 0 but saying present"
                    )

    # ... basically in static case, f_site = 1/N \sum_i log(Z_i) for Z_i the norm factors of marginals
    # similarly then calc f_link = 1/2N \sum_{ij} log(Z_{ij}) for the twopoint marg norms (for us these will include time margs)
    # then final term is something like 1/2 \sum_{qr} p_{qr} * alpha_q * alpha_r  (average degree)
    # and free energy approx F_{bethe} = f_link - f_site - last_term, and lower better

    # Assuming now instead f_site = 1/NT \sum_{it} log(Z_i), f_link = 1/NT \sum_{ijt} log(Z_{ij}), and still want av deg
    # TODO: derive rigorously - presumably last term may well include something from metadata also
    # TODO: modify for DC case also
    f_site /= N * T
    f_link /= N * T
    tmp_alpha = np.ones(Q)
    for q in range(Q):
        tmp_alpha[q] = _alpha[q]
        for r in range(Q):
            last_term += tmp_alpha[q] * tmp_alpha[r] * _beta[q, r, 0]
    for t in range(T - 1):
        tmp_alphaqm1 = tmp_alpha.copy()
        tmp_alpha = np.zeros(Q)
        for q in range(Q):
            for qprime in range(Q):
                tmp_alpha[q] += _pi[qprime, q] * tmp_alphaqm1[qprime]
        for q in range(Q):
            for r in range(Q):
                last_term += tmp_alpha[q] * tmp_alpha[r] * _beta[q, r, t + 1]
    last_term /= 2 * T
    return f_link - f_site - last_term


@njit
def nb_update_twopoint_marginals(
    N,
    T,
    Q,
    _edge_vals,
    _pres_trans,
    nbrs,
    nbrs_inv,
    directed,
    deg_corr,
    dc_lkl,
    block_edge_prob,
    trans_prob,
    _psi_e,
    _psi_t,
    twopoint_e_marg,
    twopoint_t_marg,
    verbose,
):
    # node_marg = None
    twopoint_e_marg = nb_update_twopoint_spatial_marg(
        Q,
        _edge_vals,
        nbrs,
        nbrs_inv,
        directed,
        deg_corr,
        dc_lkl,
        block_edge_prob,
        _psi_e,
        twopoint_e_marg,
    )
    if verbose:
        print("\tUpdated twopoint spatial marg")
    twopoint_t_marg = nb_update_twopoint_temp_marg(
        N, T, Q, _pres_trans, trans_prob, _psi_t, twopoint_t_marg
    )
    if verbose:
        print("\tUpdated twopoint temp marg")
    # twopoint_marginals = [twopoint_e_marg, twopoint_t_marg]
    return (twopoint_e_marg, twopoint_t_marg)


@njit
def _nb_zero_twopoint_e_marg(N, T, Q, nbrs):
    # instantiate as list construction more expensive
    # duplicate spatial msg idea for twopoint marg, so have \psi^{ijt}_{qr} = twopoint_e_marg[t][i][j_idx in nbrs[t][i],qr], then minimal memory required
    # twopoint_e_marg = List.empty_list(
    #     ListType(float64[:, :, :])
    # )  # np.zeros((N,N,T,Q,Q))
    tmp = List()
    for t in range(T):
        t_tmp = List()
        for i in range(N):
            i_tmp = np.zeros((len(nbrs[t][i]), Q, Q))
            t_tmp.append(i_tmp)
        tmp.append(t_tmp)
    twopoint_e_marg = tmp
    return twopoint_e_marg


@njit(parallel=True, fastmath=USE_FASTMATH)
def nb_update_twopoint_spatial_marg(
    Q,
    _edge_vals,
    nbrs,
    nbrs_inv,
    directed,
    deg_corr,
    dc_lkl,
    block_edge_prob,
    _psi_e,
    twopoint_e_marg,
):
    # p_qrt = block_edge_prob
    # psi_e in shape [t][i][j_idx in nbrs[t][i],q] (list(list(2d array)))
    for e_idx in prange(_edge_vals.shape[0]):
        i, j, t, a_ijt = _edge_vals[e_idx, :]
        i, j, t, a_ijt = int(i), int(j), int(t), float(a_ijt)
        # print(i, j, t)
        j_idx = nbrs[t][i] == j
        i_idx = nbrs_inv[t][i][j_idx]
        # tmp = np.outer(_psi_e[t][i][j_idx, :], _psi_e[t][j][i_idx, :])
        # if not directed:
        #     tmp += np.outer(
        #         _psi_e[t][j][i_idx, :], _psi_e[t][i][j_idx, :]
        #     )
        tmp = np.zeros((Q, Q))
        for q in range(Q):
            tmp[q, q] += (_psi_e[t][i][j_idx, q] * _psi_e[t][j][i_idx, q])[0]
            for r in range(q + 1, Q):
                tmp[q, r] += (_psi_e[t][i][j_idx, q] * _psi_e[t][j][i_idx, r])[0]
                if not directed:
                    tmp[q, r] += (_psi_e[t][j][i_idx, q] * _psi_e[t][i][j_idx, r])[0]
                    tmp[r, q] = tmp[q, r]
            if directed:
                for r in range(q):
                    tmp[q, r] += (_psi_e[t][i][j_idx, q] * _psi_e[t][j][i_idx, r])[0]
        if deg_corr:
            for q in range(Q):
                tmp[q, q] *= dc_lkl[e_idx, q, q]
                for r in range(q + 1, Q):
                    tmp[q, r] *= dc_lkl[e_idx, q, r]
                    if not directed:
                        tmp[r, q] = tmp[q, r]
                if directed:
                    for r in range(q):
                        tmp[q, r] *= dc_lkl[e_idx, q, r]
        else:
            for q in range(Q):
                for r in range(Q):
                    tmp[q, r] *= block_edge_prob[q, r, t]
        if tmp.sum() > 0:
            tmp /= tmp.sum()
        for q in range(Q):
            for r in range(Q):
                if tmp[q, r] < TOL:
                    tmp[q, r] = TOL
        if tmp.sum() > 0:
            tmp /= tmp.sum()
        twopoint_e_marg[t][i][j_idx, :, :] = tmp
    return twopoint_e_marg


@njit(parallel=True, fastmath=USE_FASTMATH)
def nb_update_twopoint_temp_marg(
    N, T, Q, _pres_trans, trans_prob, _psi_t, twopoint_t_marg
):
    # recall t msgs in shape (i,t,q,2), w t from 0 to T-2, and final dim (backwards from t+1, forwards from t)
    for i in prange(N):
        for t in range(T - 1):
            if _pres_trans[i, t]:
                tmp = np.zeros((Q, Q))
                for q in range(Q):
                    for qprime in range(Q):
                        tmp[q, qprime] += trans_prob[q, qprime] * (
                            _psi_t[i, t, q, 1] * _psi_t[i, t, qprime, 0]
                        )
                if tmp.sum() > 0:
                    tmp /= tmp.sum()
                for q in range(Q):
                    for qprime in range(Q):
                        if tmp[q, qprime] < TOL:
                            tmp[q, qprime] = TOL
                if tmp.sum() > 0:
                    tmp /= tmp.sum()
                twopoint_t_marg[i, t, :, :] = tmp
    return twopoint_t_marg


# if __name__ == "__main__":
#     N = 1000
#     T = 5
#     Q = 10
#     deg_corr = False
#     degs = np.random.randint(1, 10, (N, T, 2))
#     _pres_nodes = np.random.rand(N, T) < 0.95
#     _pres_trans = np.random.rand(N, T - 1) < 0.9
#     tmp = List()
#     for t in range(T):
#         tmp2 = List()
#         for i in range(N):
#             tmp2.append(np.random.randint(0, N, degs[i, t, 0]))
#         tmp.append(tmp2)
#     all_nbrs = tmp
#     nbrs_inv = tmp
#     e_nbrs_inv = tmp
#     n_msgs = 1000
#     block_edge_prob = np.random.rand(Q, Q, T)
#     trans_prob = np.random.rand(Q, Q)
#     dc_lkl = np.random.rand(N, T, Q)
#     _h = np.random.rand(Q, T)
#     meta_prob = np.random.rand(N, T, Q)
#     _alpha = np.random.rand(Q)
#     node_marg = np.random.rand(N, T)
#     tmp = List()
#     for t in range(T):
#         tmp2 = List()
#         for i in range(N):
#             tmp2.append(np.random.rand(degs[i, t, 0], Q))
#         tmp.append(tmp2)
#     _psi_e = tmp
#     _psi_t = np.random.rand(N, T, Q, 2)
#     msg_diff = 0.5
#     _edge_vals = np.random.randint(0, N, (N * T, 4))
#     directed = False
#     twopoint_e_marg = tmp
#     twopoint_t_marg = np.random.rand(N, T, Q, Q)

#     nb_update_node_marg(
#         N,
#         T,
#         Q,
#         deg_corr,
#         degs,
#         _pres_nodes,
#         _pres_trans,
#         all_nbrs,
#         nbrs_inv,
#         e_nbrs_inv,
#         n_msgs,
#         block_edge_prob,
#         trans_prob,
#         dc_lkl,
#         _h,
#         meta_prob,
#         _alpha,
#         node_marg,
#         _psi_e,
#         _psi_t,
#         msg_diff,
#     )
#     nb_update_node_marg.parallel_diagnostics(level=4)
#     nb_update_twopoint_spatial_marg(
#         Q,
#         _edge_vals,
#         all_nbrs[0][0],
#         nbrs_inv,
#         directed,
#         deg_corr,
#         dc_lkl,
#         block_edge_prob,
#         _psi_e,
#         twopoint_e_marg,
#     )
#     nb_update_twopoint_temp_marg(
#         N, T, Q, _pres_trans, trans_prob, _psi_t, twopoint_t_marg
#     )
