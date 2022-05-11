# numba reimplementation of all methods for BP class that reasonably stand to gain from doing so (would allow parallelisation + GPU usage)
# - simply prepend method name with nb_
from numba import njit, prange
from numba.typed import List
import numpy as np

from utils import nb_poisson_lkl_int

import yaml

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


@njit(parallel=True, fastmath=USE_FASTMATH)
def nb_init_msgs(
    mode, N, T, Q, nbrs, Z,
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
        node_marg /= np.expand_dims(node_marg.sum(axis=2), 2)
        ## INIT MESSAGES ##
        tmp = List()
        for t in range(T):
            tmp2 = List()
            for i in range(N):
                n_nbrs = len(nbrs[t][i])
                if n_nbrs > 0:
                    msg = np.random.rand(n_nbrs, Q)
                    # msg /= msg.sum(axis=1)[:, np.newaxis]
                    msg /= np.expand_dims(msg.sum(axis=1), 1)
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
        _psi_t /= np.expand_dims(
            _psi_t.sum(axis=2), 2
        )  # msgs from i at t forwards/backwards
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
        # TODO: don't hardcode
        p = 0.8
        ## INIT MARGINALS ##
        one_hot_Z = np.zeros((N, T, Q))
        for i in range(N):
            for t in range(T):
                one_hot_Z[i, t, Z[i, t]] = 1.0

        node_marg = p * one_hot_Z + (1 - p) * np.random.rand(N, T, Q)
        node_marg /= np.expand_dims(node_marg.sum(axis=2), 2)

        ## INIT MESSAGES ##
        tmp = List()
        for t in range(T):
            tmp2 = List()
            for i in range(N):
                n_nbrs = len(nbrs[t][i])
                if n_nbrs > 0:
                    msg = p * np.expand_dims(one_hot_Z[i, t, :], 0) + (
                        1 - p
                    ) * np.random.rand(n_nbrs, Q)
                    msg /= np.expand_dims(msg.sum(axis=1), 1)
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
        _psi_t /= np.expand_dims(_psi_t.sum(axis=2), 2)
    return _psi_e, _psi_t, node_marg


@njit(parallel=True, fastmath=USE_FASTMATH)
def nb_forward_temp_msg_term(Q, trans_prob, i, t, _psi_t):
    # sum_qprime(trans_prob(qprime,q)*_psi_t[i,t-1,qprime,1])
    # from t-1 to t
    try:
        # print(trans_prob)
        # print(trans_prob.T)
        out = np.zeros(Q)
        for q in prange(Q):
            for qprime in prange(Q):
                out[q] += trans_prob[qprime, q] * _psi_t[i, t - 1, qprime, 1]
        for q in range(Q):
            if out[q] < TOL:
                out[q] = TOL
    except:
        # must have t=0 so t-1 outside of range, no forward message, but do have alpha instead - stopped now as need for backward term
        assert t == 0
        # out = model._alpha
    return out


@njit(parallel=True, fastmath=USE_FASTMATH)
def nb_backward_temp_msg_term(Q, T, trans_prob, i, t, _psi_t):
    """Backwards temporal message term for marginal of i at t, coming from i at t + 1
    Much as for spatial messages, by definition 
        \psi^{i(t)\to i(t+1)} \propto \psi^{it} / \psi^{i(t+1)\to i(t)} 
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
        for q in prange(Q):
            for qprime in prange(Q):
                out[q] += trans_prob[q, qprime] * _psi_t[i, t, qprime, 0]
        for q in range(Q):
            if out[q] < TOL:
                out[q] = TOL
    except:
        # t=T outside of range, so no backward message
        assert t == T - 1
        out = np.ones(Q)
    return np.ascontiguousarray(out)


@njit(parallel=True, fastmath=USE_FASTMATH)
def nb_spatial_msg_term_small_deg(
    Q,
    nbrs,
    e_nbrs_inv,
    nbrs_inv,
    deg_corr,
    degs,
    e_idx,
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
    updating spatial messages \psi^{i\to j}(t) and external field h(t) in the process.
    
    Updating together rational as by definition, 
        \psi^{i\to j}(t) \propto \psi^{it}/\psi^{j\to i}(t),
    hence we can 
        (i) Calculate all terms involved in unnormalised node marginal / node messages 
        (ii) Calculate product of these (for unnorm marginals), divide by term involving j 
                for \psi^{i\to j}(t) for unnorm messages
        (iii) Calculate normalisation for node marginals, update marginals 
        (iv) Calculate normalisation for node messages, update messages 
        
    As there are \Oh(Q d_i) terms only involved for each q (so \Oh(Q^2 d_i) total),
    for sparse networks where the average degree d ~ \Oh(1), updating all messages 
    and marginals together is an \Oh(Q^2 N T d) process. As typically Q ~ log(N), 
    this means approximate complexity of \Oh(2 N T d log(N)), so roughly linear in the 
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
    for nbr_idx, j in enumerate(nbrs):
        idx = nbrs_inv[nbr_idx]
        jtoi_msgs = _psi_e[t][j][idx, :].reshape(
            -1
        )  # for whatever reason this stays 2d, so need to flatten first
        tmp = np.zeros(Q)
        if deg_corr:
            for q in prange(Q):
                for r in prange(Q):
                    tmp[q] += dc_lkl[e_nbrs_inv[nbr_idx], r, q] * jtoi_msgs[r]
        else:
            for q in prange(Q):
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
        msg *= tmp
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
    msg *= meta_prob[i, t]
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


@njit(parallel=True, fastmath=USE_FASTMATH)
def nb_spatial_msg_term_large_deg(
    Q,
    nbrs,
    nbrs_inv,
    deg_corr,
    degs,
    e_idx,
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
    # TODO: create edge idx lookup table along w nbrs s.t.
    # _edge_vals[e_idx[t][i][j],-1]=A[i,j,t]
    beta = block_edge_prob[:, :, t]
    # deg_i = len(nbrs)
    msg = np.zeros((Q,))
    log_field_iter = np.zeros((len(nbrs), Q))
    # TODO: make logger that tracks this
    # print("Large deg version used")
    max_log_msg = -1000000.0
    for nbr_idx, j in enumerate(nbrs):
        if len(nbrs[t][j] > 0):
            idx = nbrs[t][j] == i
            if idx.sum() == 0:
                print("Fault:", j, t)
        else:
            print("Fault:", j, t)
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
    msg += np.log(meta_prob(i, t))
    return msg, max_log_msg, log_field_iter


@njit(parallel=True, fastmath=USE_FASTMATH)
def nb_init_h(
    N, T, Q, degs, deg_corr, block_edge_prob, node_marg,
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
        for q in range(Q):
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
    e_idx,
    i,
    t,
    _pres_nodes,
    _pres_trans,
    all_nbrs,
    nbrs_inv,
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
    """Update all node marginals (in random order), simultaneously updating messages and 
    external fields h(t) - process is as follows:
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
    if RANDOM_ONLINE_UPDATE_MSG:
        np.random.shuffle(node_update_order)
        np.random.shuffle(time_update_order)
    for i in node_update_order:
        for t in time_update_order:
            if _pres_nodes[i, t]:
                nbrs = all_nbrs[t][i]
                deg_i = len(nbrs)
                if deg_i > 0:
                    if deg_i < LARGE_DEG_THR:
                        (spatial_msg_term, field_iter,) = nb_spatial_msg_term_small_deg(
                            Q,
                            nbrs,
                            nbrs_inv,
                            deg_corr,
                            degs,
                            e_idx,
                            i,
                            t,
                            dc_lkl,
                            _h,
                            meta_prob,
                            block_edge_prob,
                            _psi_e,
                        )
                        nb_update_h(
                            Q,
                            -1.0,
                            i,
                            t,
                            degs,
                            deg_corr,
                            block_edge_prob,
                            _h,
                            node_marg,
                        )
                        # REMOVE CHECK AFTER FIX
                        if np.isnan(spatial_msg_term).sum() > 0:
                            print("i,t:", i, t)
                            print("spatial:", spatial_msg_term)
                            print("deg[i,t]", degs[i, t])
                            print("beta:", block_edge_prob[:, :, t])
                            raise RuntimeError("Problem w spatial term")
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
                                    # if tmp_backwards_msg[q] > 1 - TOL:
                                    #     tmp_backwards_msg[q] = 1 - TOL
                                # tmp_backwards_msg[tmp_backwards_msg > 1 - TOL] = 1 - TOL
                                tmp_backwards_msg /= tmp_backwards_msg.sum()
                                msg_diff += (
                                    np.abs(
                                        tmp_backwards_msg - _psi_t[i, t - 1, :, 0]
                                    ).mean()
                                    / n_msgs
                                )
                                _psi_t[i, t - 1, :, 0] = tmp_backwards_msg
                                forward_term = nb_forward_temp_msg_term(
                                    Q, trans_prob, i, t, _psi_t
                                )
                            else:
                                # node present at t but not t-1, use alpha instead
                                forward_term = _alpha
                            tmp *= forward_term
                        # REMOVE CHECK AFTER FIX
                        if np.isnan(forward_term).sum() > 0:
                            print("i,t:", i, t)
                            print("forward:", forward_term)
                            raise RuntimeError("Problem w forward term")
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
                                    alt_nbrs = np.arange(deg_i)
                                    alt_nbrs = alt_nbrs[alt_nbrs != nbr_idx]
                                    for k in alt_nbrs:
                                        tmp_loc *= field_iter[k, q]
                                    tmp_spatial_msg[nbr_idx, q] = tmp_loc
                        tmp_spat_sums = tmp_spatial_msg.sum(axis=1)
                        for nbr_idx in range(deg_i):
                            if tmp_spat_sums[nbr_idx] > 0:
                                tmp_spatial_msg[nbr_idx, :] /= tmp_spat_sums[nbr_idx]
                        for nbr_idx in range(deg_i):
                            for q in range(Q):
                                if tmp_spatial_msg[nbr_idx, q] < TOL:
                                    tmp_spatial_msg[nbr_idx, q] = TOL
                                # if tmp_spatial_msg[nbr_idx, q] > 1 - TOL:
                                #     tmp_spatial_msg[nbr_idx, q] = 1 - TOL
                        tmp_spatial_msg /= np.expand_dims(
                            tmp_spatial_msg.sum(axis=1), 1
                        )
                        msg_diff += (
                            np.abs(tmp_spatial_msg - _psi_e[t][i]).mean()
                            * deg_i
                            / n_msgs
                        )  # NB need to mult by deg_i so weighted correctly
                        # try:
                        #     assert np.isnan(tmp_spatial_msg).sum() == 0
                        #     assert np.isinf(tmp_spatial_msg).sum() == 0
                        # except:
                        #     print("(i,t):", i, t)
                        #     print("tmp_spatial:", tmp_spatial_msg)
                        #     print("back_term:", back_term)
                        #     print("forward_term:", forward_term)
                        #     print("unnorm spatial:", spatial_msg_term)
                        #     print("field iters:", field_iter)
                        #     raise RuntimeError("Problem with spatial msg")
                        _psi_e[t][i] = tmp_spatial_msg
                        ## UPDATE FORWARDS MESSAGES FROM i AT t ##
                        if t < T - 1 and _pres_trans[i, t]:
                            # tmp_forwards_msg = (
                            #     tmp / back_term
                            # )  # again causes problem if back_term << 1
                            # so better to just calc directly - in this
                            # case just insignificant memory overhead
                            # as calc complexity identical (if anything
                            # easier)
                            tmp_forwards_msg = spatial_msg_term
                            if t > 0:
                                tmp_forwards_msg *= forward_term
                            if tmp_forwards_msg.sum() > 0:
                                tmp_forwards_msg /= tmp_forwards_msg.sum()
                            tmp_forwards_msg[tmp_forwards_msg < TOL] = TOL
                            tmp_forwards_msg /= tmp_forwards_msg.sum()
                            # tmp_forwards_msg[tmp_forwards_msg > 1 - TOL] = 1 - TOL
                            # try:
                            #     assert np.isnan(tmp_forwards_msg).sum() == 0
                            #     assert np.isinf(tmp_forwards_msg).sum() == 0
                            # except:
                            #     print("(i,t):", i, t)
                            #     print("tmp_forwards:", tmp_forwards_msg)
                            #     print("back_term:", back_term)
                            #     print("forward_term:", forward_term)
                            #     print("unnorm spatial:", spatial_msg_term)
                            #     raise RuntimeError("Problem with forward msg")
                            msg_diff += (
                                np.abs(tmp_forwards_msg - _psi_t[i, t, :, 1]).mean()
                                / n_msgs
                            )
                            _psi_t[i, t, :, 1] = tmp_forwards_msg
                        ## UPDATE MARGINAL OF i AT t ##
                        tmp_marg = tmp
                        if tmp_marg.sum() > 0:
                            tmp_marg /= tmp_marg.sum()
                        tmp_marg[tmp_marg < TOL] = TOL
                        tmp_marg = tmp_marg / tmp_marg.sum()
                        # tmp_marg[tmp_marg > 1 - TOL] = 1 - TOL
                        node_marg[i, t, :] = tmp_marg

                    else:
                        (
                            spatial_msg_term,
                            max_log_spatial_msg_term,
                            log_field_iter,
                        ) = nb_spatial_msg_term_large_deg(
                            Q,
                            nbrs,
                            nbrs_inv,
                            deg_corr,
                            degs,
                            e_idx,
                            i,
                            t,
                            dc_lkl,
                            _h,
                            meta_prob,
                            block_edge_prob,
                            _psi_e,
                        )
                        nb_update_h(
                            Q,
                            -1.0,
                            i,
                            t,
                            degs,
                            deg_corr,
                            block_edge_prob,
                            _h,
                            node_marg,
                        )
                        if t == 0:
                            tmp += np.log(_alpha)
                        tmp = spatial_msg_term
                        back_term = np.zeros(Q)
                        if t < T - 1:
                            if _pres_trans[i, t]:
                                back_term = np.log(
                                    nb_backward_temp_msg_term(
                                        Q, T, trans_prob, i, t, _psi_t
                                    )
                                )
                            tmp += back_term
                        ## UPDATE BACKWARDS MESSAGES FROM i AT t ##
                        forward_term = np.zeros(Q)
                        if t > 0:
                            if _pres_trans[i, t - 1]:
                                tmp_backwards_msg = np.exp(
                                    tmp - max_log_spatial_msg_term
                                )
                                if tmp_backwards_msg.sum() > 0:
                                    tmp_backwards_msg /= tmp_backwards_msg.sum()
                                tmp_backwards_msg[tmp_backwards_msg < TOL] = TOL
                                # tmp_backwards_msg[tmp_backwards_msg > 1 - TOL] = TOL
                                tmp_backwards_msg /= tmp_backwards_msg.sum()
                                _psi_t[i, t - 1, :, 0] = tmp_backwards_msg
                                forward_term = np.log(
                                    nb_forward_temp_msg_term(
                                        Q, trans_prob, i, t, _psi_t
                                    )
                                )
                            else:
                                # node present at t but not t-1, so use alpha instead
                                forward_term = np.log(_alpha)
                            tmp += forward_term
                        ## UPDATE SPATIAL MESSAGES FROM i AT t ##
                        tmp_spatial_msg = np.expand_dims(tmp, 0) - log_field_iter
                        log_field_iter_max = np.array(
                            [nbr_fld.max() for nbr_fld in tmp_spatial_msg]
                        )
                        tmp_spatial_msg = np.exp(
                            tmp_spatial_msg - np.expand_dims(log_field_iter_max, 1)
                        )
                        tmp_spat_sums = tmp_spatial_msg.sum(axis=1)
                        for nbr_idx in range(deg_i):
                            if tmp_spat_sums[nbr_idx] > 0:
                                tmp_spatial_msg[nbr_idx, :] /= tmp_spat_sums[nbr_idx]
                        for nbr_idx in range(deg_i):
                            for q in range(Q):
                                if tmp_spatial_msg[nbr_idx, q] < TOL:
                                    tmp_spatial_msg[nbr_idx, q] = TOL
                                # if tmp_spatial_msg[nbr_idx, q] > 1 - TOL:
                                #     tmp_spatial_msg[nbr_idx, q] = 1 - TOL
                        tmp_spatial_msg /= np.expand_dims(
                            tmp_spatial_msg.sum(axis=1), 1
                        )
                        # try:
                        #     assert np.isnan(tmp_spatial_msg).sum() == 0
                        #     assert np.isinf(tmp_spatial_msg).sum() == 0
                        # except:
                        #     print("i,t", i, t)
                        #     print("tmp_spatial:", tmp_spatial_msg)
                        #     print("back_term:", back_term)
                        #     print("forward_term:", forward_term)
                        #     print("unnorm spatial:", spatial_msg_term)
                        #     print("field iters:", field_iter)
                        #     raise RuntimeError("Problem with spatial msg")
                        _psi_e[t][i] = tmp_spatial_msg
                        ## UPDATE FORWARDS MESSAGES FROM i AT t ##
                        if t < T - 1:
                            if _pres_trans[i, t]:
                                tmp_forwards_msg = np.exp(
                                    tmp - max_log_spatial_msg_term - back_term
                                )
                                if tmp_forwards_msg.sum() > 0:
                                    tmp_forwards_msg /= tmp_forwards_msg.sum()
                                tmp_forwards_msg[tmp_forwards_msg < TOL] = TOL
                                # tmp_forwards_msg[tmp_forwards_msg > 1 - TOL] = 1 - TOL
                                tmp_forwards_msg /= tmp_forwards_msg.sum()
                                # try:
                                #     assert np.isnan(tmp_forwards_msg).sum() == 0
                                #     assert np.isinf(tmp_forwards_msg).sum() == 0
                                # except:
                                #     print("(i,t):", i, t)
                                #     print("tmp_forwards:", tmp_forwards_msg)
                                #     print("back_term:", back_term)
                                #     print("forward_term:", forward_term)
                                #     print("unnorm spatial:", spatial_msg_term)
                                #     raise RuntimeError("Problem with forward msg")
                                _psi_t[i, t, :, 1] = tmp_forwards_msg
                        ## UPDATE MARGINAL OF i AT t ##
                        tmp_marg = np.exp(tmp - max_log_spatial_msg_term)
                        if tmp_marg.sum() > 0:
                            tmp_marg /= tmp_marg.sum()
                        tmp_marg[tmp_marg < TOL] = TOL
                        # tmp_marg[tmp_marg > 1 - TOL] = 1 - TOL
                        tmp_marg /= tmp_marg.sum()
                        node_marg[i, t, :] = tmp_marg
                    nb_update_h(
                        Q, 1.0, i, t, degs, deg_corr, block_edge_prob, _h, node_marg
                    )
                else:
                    # print("WARNING: disconnected nodes not yet handled properly")
                    # print("i,t:", i, t)
                    raise RuntimeError(
                        "Problem with measuring presence - deg = 0 but saying present"
                    )
            else:
                node_marg[i, t, :] = 0.0

    if np.isnan(msg_diff):
        for i in range(N):
            for t in range(T - 1):
                if np.isnan(node_marg[i, t]).sum() > 0:
                    print("nans for node marg @ (i,t)=", i, t)
                if np.isnan(_psi_e[t][i]).sum() > 0:
                    print("nans for spatial msgs @ (i,t)=", i, t)
                if np.isnan(_psi_t[i, t]).sum() > 0:
                    print("nans for temp marg @ (i,t)=", i, t)
            if np.isnan(node_marg[i, T - 1]).sum() > 0:
                print("nans for node marg @ (i,t)=", i, T - 1)
            if np.isnan(_psi_e[T - 1][i]).sum() > 0:
                print("nans for spatial msgs @ (i,t)=", i, T - 1)
        raise RuntimeError("Problem updating messages")
    return node_marg, _psi_e, _psi_t, msg_diff


@njit(parallel=True, fastmath=USE_FASTMATH)
def nb_compute_free_energy(
    N,
    T,
    Q,
    deg_corr,
    degs,
    e_idx,
    i,
    t,
    _pres_nodes,
    _pres_trans,
    all_nbrs,
    nbrs_inv,
    n_msgs,
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
    f_link = 0.0
    last_term = 0.0  # something like average degree, but why?
    for i in range(N):
        for t in range(T):
            if _pres_nodes[i, t]:
                nbrs = nbrs[t][i]
                deg_i = len(nbrs)
                if deg_i > 0:
                    if deg_i < LARGE_DEG_THR:
                        (spatial_msg_term, field_iter,) = nb_spatial_msg_term_small_deg(
                            Q,
                            nbrs,
                            nbrs_inv,
                            deg_corr,
                            degs,
                            e_idx,
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
                                f_link += tmp.sum()
                                forward_term = nb_forward_temp_msg_term(
                                    Q, trans_prob, i, t, _psi_t
                                )
                            else:
                                # node present at t but not t-1, use alpha instead
                                forward_term = _alpha
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
                                    alt_nbrs = np.arange(deg_i)
                                    alt_nbrs = alt_nbrs[alt_nbrs != nbr_idx]
                                    for k in alt_nbrs:
                                        tmp_loc *= field_iter[k, q]
                                    tmp_spatial_msg[nbr_idx, q] = tmp_loc
                        tmp_spat_sums = tmp_spatial_msg.sum(axis=1)
                        for nbr_idx in range(deg_i):
                            # add spatial messages to f_link
                            f_link += tmp_spat_sums[nbr_idx]
                        if t < T - 1 and _pres_trans[i, t]:
                            # add forwards messages to f_link
                            tmp_forwards_msg = spatial_msg_term
                            if t > 0:
                                tmp_forwards_msg *= forward_term
                            f_link += tmp_forwards_msg.sum()
                        # add marg to f_site
                        f_site += tmp.sum()
                    else:
                        (
                            spatial_msg_term,
                            max_log_spatial_msg_term,
                            log_field_iter,
                        ) = nb_spatial_msg_term_large_deg(
                            Q,
                            nbrs,
                            nbrs_inv,
                            deg_corr,
                            degs,
                            e_idx,
                            i,
                            t,
                            dc_lkl,
                            _h,
                            meta_prob,
                            block_edge_prob,
                            _psi_e,
                        )
                        if t == 0:
                            tmp += np.log(_alpha)
                        tmp = spatial_msg_term
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
                                f_link += tmp_backwards_msg.sum()

                                forward_term = np.log(
                                    nb_forward_temp_msg_term(
                                        Q, trans_prob, i, t, _psi_t
                                    )
                                )
                            else:
                                # node present at t but not t-1, so use alpha instead
                                forward_term = np.log(_alpha)
                            tmp += forward_term
                        tmp_spatial_msg = np.expand_dims(tmp, 0) - log_field_iter
                        log_field_iter_max = np.array(
                            [nbr_fld.max() for nbr_fld in tmp_spatial_msg]
                        )
                        tmp_spatial_msg = np.exp(
                            tmp_spatial_msg - np.expand_dims(log_field_iter_max, 1)
                        )
                        tmp_spat_sums = tmp_spatial_msg.sum(axis=1)
                        # add spatial msgs to f_link
                        for nbr_idx in range(deg_i):
                            f_link += tmp_spat_sums[nbr_idx]

                        # add forwards msg to f_link
                        if t < T - 1:
                            if _pres_trans[i, t]:
                                tmp_forwards_msg = np.exp(
                                    tmp - max_log_spatial_msg_term - back_term
                                )
                                f_link += tmp_forwards_msg.sum()
                        # add marg to f_site
                        tmp_marg = np.exp(tmp - max_log_spatial_msg_term)
                        f_site += tmp_marg.sum()
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


@njit(parallel=True, fastmath=USE_FASTMATH)
def nb_update_twopoint_marginals(verbose):
    # node_marg = None
    twopoint_e_marg = nb_update_twopoint_spatial_marg()
    if verbose:
        print("\tUpdated twopoint spatial marg")
    twopoint_t_marg = nb_update_twopoint_temp_marg()
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
    directed,
    deg_corr,
    dc_lkl,
    block_edge_prob,
    _psi_e,
    twopoint_e_marg,
):
    # p_qrt = block_edge_prob
    # psi_e in shape [t][i][j_idx in nbrs[t][i],q] (list(list(2d array)))
    for e_idx, (i, j, t, a_ijt) in enumerate(_edge_vals):
        i, j, t, a_ijt = int(i), int(j), int(t), float(a_ijt)
        # print(i, j, t)
        j_idx = nbrs[t][i] == j
        # TODO: create inverse array that holds these values
        #       in memory rather than calc on the fly each time
        i_idx = nbrs[t][j] == i
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
            tmp *= block_edge_prob[:, :, t]
        if tmp.sum() > 0:
            tmp /= tmp.sum()
        for q in range(Q):
            for r in range(Q):
                if tmp[q, r] < TOL:
                    tmp[q, r] = TOL
        tmp /= tmp.sum()
        twopoint_e_marg[t][i][j_idx, :, :] = tmp
    return twopoint_e_marg


@njit(parallel=True, fastmath=USE_FASTMATH)
def nb_update_twopoint_temp_marg(
    N, T, Q, _pres_trans, trans_prob, _psi_t, twopoint_t_marg
):
    # recall t msgs in shape (i,t,q,2), w t from 0 to T-2, and final dim (backwards from t+1, forwards from t)
    for i in range(N):
        for t in range(T - 1):
            if _pres_trans[i, t]:
                tmp = np.zeros((Q, Q))
                for q in range(Q):
                    for qprime in range(Q):
                        tmp[q, qprime] += trans_prob[q, qprime] * (
                            _psi_t[i, t, q, 1]
                            * _psi_t[i, t, qprime, 0]
                            # + _psi_t[i, t, qprime, 1] * _psi_t[i, t, q, 0]
                        )
                # tmp = np.outer(
                #     _psi_t[i, t, :, 1], _psi_t[i, t, :, 0]
                # )
                # TODO: check this
                # tmp += np.outer(_psi_t[i, t, :, 0], _psi_t[i, t, :, 1])
                # tmp *= trans_prob
                if tmp.sum() > 0:
                    tmp /= tmp.sum()
                for q in range(Q):
                    for qprime in range(Q):
                        if tmp[q, qprime] < TOL:
                            tmp[q, qprime] = TOL
                tmp /= tmp.sum()
                twopoint_t_marg[i, t, :, :] = tmp
    return twopoint_t_marg

