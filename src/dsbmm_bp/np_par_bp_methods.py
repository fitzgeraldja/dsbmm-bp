# numpy reimplementation of all methods for BP class that reasonably stand to gain from doing so
import numpy as np
import yaml  # type: ignore
from scipy import sparse

from dsbmm_bp.np_par_dsbmm_methods import NumpyDSBMM

try:
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    TOL = config["tol"]  # min value permitted for msgs etc
    # (for numerical stability)
    LARGE_DEG_THR = config[
        "large_deg_thr"
    ]  # threshold of node degree above which log msgs calculated
    # (for numerical stability)
    PLANTED_P = config["planted_p"]
except FileNotFoundError:
    TOL = 1e-50
    LARGE_DEG_THR = 20
    RANDOM_ONLINE_UPDATE_MSG = False
    PLANTED_P = 0.8


class NumpyBP:
    def __init__(self, dsbmm: NumpyDSBMM):
        self.model = dsbmm
        self.T = self.model.T
        self.N = self.model.N
        self.Q = self.model.Q
        self.deg_corr = self.model.deg_corr
        self.directed = self.model.directed
        self.use_meta = self.model.use_meta
        self.verbose = self.model.verbose
        self.frozen = self.model.frozen
        self.A = self.model.A
        self.n_msgs = self.model.E.sum() + self.N * (self.T - 1) * 2
        self.X = self.model.X
        self.Z = self.model.Z
        self._pres_nodes = self.model._pres_nodes
        self._pres_trans = self.model._pres_trans
        # start with given membership and corresponding messages, iterate until reach fixed point
        # given messages at fixed point, can update parameters to their most likely values - these emerge naturally
        # by requiring that the fixed point provides stationary free energy w.r.t. parameters
        # self.get_neighbours()
        # self._zero_twopoint_e_marg()
        self.node_marg = np.zeros((self.N, self.T, self.Q))
        self.construct_edge_idxs_and_inv()
        if not self.frozen:
            self.twopoint_e_marg = np.zeros((self.E_idxs[-1], self.Q, self.Q))
            self.twopoint_t_marg = np.zeros((self.N, self.T - 1, self.Q, self.Q))
        self.msg_diff = 0.0

    def zero_diff(self):
        self.msg_diff = 0.0

    @property
    def meta_prob(self):
        # NB DSBMM already performed
        # meta_lkl = meta_lkl**tuning_param
        # so adjusted suitably already
        return self.model.meta_lkl

    def init_messages(
        self,
        mode="planted",
    ):
        # construct msgs as T*Q*N,N sparse matrix,
        # where (Q*N*t + N*q + i, j) gives \psi_q^{i\to j}(t),
        # This must have same sparsity structure as symmetrised
        # version of A
        # NB only currently implemented for undirected
        # nets, else would need to introduce some changes
        # to message eqns below
        # - see e.g.
        # 'Detectability of macroscopic structures in directed
        # asymmetric stochastic block model' (2019)
        # Assume passing A already as T length list of
        # sparse matrices
        sym_A = [((self.A[t] + self.A[t].T) != 0) * 1.0 for t in range(self.T)]
        self._psi_e = sparse.vstack(
            [sparse.csr_matrix(sym_A[t]) for t in range(self.T) for _ in range(self.Q)]
        )
        if mode == "random":
            # initialise by random messages and marginals
            self.node_marg = np.random.rand(self.N, self.T, self.Q)
            self.node_marg /= self.node_marg.sum(axis=2, keepdims=True)
            ## INIT MESSAGES ##
            self._psi_e.data = np.random.rand(len(self._psi_e.data))
            # psi_e[np.arange(N*T*Q).reshape(T,Q,N)[0].T.flatten(),:].T # gives N,N*Q array where j, i*q + q entry is message from i to j about being in q at t
            # so can reshape to N*N,Q, take sum along last axis, then reshape to N x N to get normalising sum for each i, then tile to make in right shape overall
            sums = sparse.vstack(
                [
                    sparse.csr_matrix(
                        self._psi_e[
                            np.arange(self.N * self.T * self.Q)
                            .reshape(self.T, self.Q, self.N)[t]
                            .T.flatten(),
                            :,
                        ]
                        .T.reshape(self.N * self.N, self.Q)
                        .sum(axis=-1)
                        .reshape(self.N, self.N)
                    )
                    for t in range(self.T)
                    for _ in range(self.Q)
                ]
            )
            self._psi_e.data /= sums.data  # normalise messages

            self._psi_t = np.random.rand(self.N, self.T - 1, self.Q, 2)
            self._psi_t /= self._psi_t.sum(axis=2)[:, :, np.newaxis, :]
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
            raise NotImplementedError("partial msg init not yet implemented")
        elif mode == "planted":
            # initialise by given partition plus some random noise, with strength of info used
            # specified by plant_strength (shortened to ps below)
            # i.e. if z_0(i,t) = r,
            # \psi^{it}_q = \delta_{qr}(ps + (1 - ps)*rand) + (1 - \delta_{qr})*(1 - ps)*rand
            p = PLANTED_P
            ## INIT MARGINALS ##
            one_hot_Z = self.onehot_initialization(self.Z)  # in shape N x T x Q
            self.node_marg = p * one_hot_Z + (1 - p) * np.random.rand(
                self.N, self.T, self.Q
            )
            self.node_marg /= self.node_marg.sum(axis=-1)[:, :, np.newaxis]

            ## INIT MESSAGES ##
            tmp = sparse.csr_matrix(
                (self._psi_e.data, self._psi_e.indices, self._psi_e.indptr),
                shape=self._psi_e.shape,
            )
            Z_idxs = np.flatnonzero(one_hot_Z.transpose(1, 2, 0))
            # column indices for row i are stored in
            # indices[indptr[i]:indptr[i+1]]
            # and corresponding values are stored in
            # data[indptr[i]:indptr[i+1]]
            tmp_indptr = tmp.indptr
            for i in Z_idxs:
                tmp.data[tmp_indptr[i] : tmp_indptr[i + 1]] *= p
            other_idxs = np.arange(self.N * self.T * self.Q)
            other_idxs = np.setdiff1d(other_idxs, Z_idxs)
            for i in other_idxs:
                tmp.data[tmp_indptr[i] : tmp_indptr[i + 1]] = 0
            self._psi_e.data = np.random.rand(len(self._psi_e.data))

            sums = sparse.vstack(
                [
                    sparse.csr_matrix(
                        self._psi_e[
                            np.arange(self.N * self.T * self.Q)
                            .reshape(self.T, self.Q, self.N)[t]
                            .T.flatten(),
                            :,
                        ]
                        .T.reshape(self.N * self.N, self.Q)
                        .sum(axis=-1)
                        .reshape(self.N, self.N)
                    )
                    for t in range(self.T)
                    for _ in range(self.Q)
                ]
            )
            self._psi_e.data /= sums.data  # normalise noise
            self._psi_e.data *= 1 - p
            self._psi_e.data += tmp.data
            # renormalise just in case
            sums = sparse.vstack(
                [
                    sparse.csr_matrix(
                        self._psi_e[
                            np.arange(self.N * self.T * self.Q)
                            .reshape(self.T, self.Q, self.N)[t]
                            .T.flatten(),
                            :,
                        ]
                        .T.reshape(self.N * self.N, self.Q)
                        .sum(axis=-1)
                        .reshape(self.N, self.N)
                    )
                    for t in range(self.T)
                    for _ in range(self.Q)
                ]
            )
            self._psi_e.data /= sums.data

            self._psi_t = np.random.rand(self.N, self.T - 1, self.Q, 2)
            self._psi_t /= self._psi_t.sum(axis=2)[:, :, np.newaxis, :]
            self._psi_t *= 1 - p
            self._psi_t[..., 0] += p * one_hot_Z[:, 1:, :]
            self._psi_t[..., 1] += p * one_hot_Z[:, : self.T - 1, :]
            self._psi_t /= self._psi_t.sum(axis=2)[:, :, np.newaxis, :]

    @property
    def trans_prob(self):
        return self.model._pi

    def forward_temp_msg_term(self):
        # sum_qprime(trans_prob(qprime,q)*_psi_t[i,t-1,qprime,1])
        # from t-1 to t
        # _psi_t in shape [N, T - 1, Q, (backwards from t+1, forwards from t)]
        # so return gives msg term for i belonging to q at t to i at t+1 for t<T in i,t,q
        out = np.einsum("itr,rq->itq", self._psi_t[:, :, :, 1], self.trans_prob)
        out[out < TOL] = TOL
        return out

    def backward_temp_msg_term(self):
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
        out = np.einsum("itr,qr->itq", self._psi_t[..., 0], self.trans_prob)
        out[out < TOL] = TOL
        return out

    def construct_edge_idxs_and_inv(self):
        self.bin_degs = np.array(
            np.vstack([(self.A[t] != 0).sum(axis=0).squeeze() for t in range(self.T)]).T
        )
        self.nz_idxs = np.vstack(
            (np.zeros(self.T, dtype=int), np.cumsum(self.bin_degs, axis=0))
        )
        self.E_idxs = np.concatenate([[0], self.bin_degs.sum(axis=0).cumsum()])
        self.all_idxs = {}
        self.flat_i_idxs = {}
        self.all_inv_idxs = {}
        for t in range(self.T):
            # np.flatnonzero fails for sparse matrix,
            # but can easily get equivalent using
            row_idx, col_idx = self.A[t].T.nonzero()
            # and then np.flatnonzero == col_idx + N*row_idx
            msg_idxs = (
                np.array(
                    np.meshgrid(self.N * np.arange(self.Q), col_idx + self.N * row_idx)
                )
                .sum(axis=0)
                .flatten()
                .reshape(-1, self.Q)
            )
            msg_idxs -= np.repeat(self.N * np.arange(self.N), self.bin_degs[:, t])[
                :, np.newaxis
            ]
            i_idxs = np.repeat(
                np.arange(self.N), self.bin_degs[:, t] * self.Q
            ).flatten()
            j_idxs = msg_idxs.flatten() + self.Q * self.N * t
            self.all_idxs[t] = {"i_idxs": i_idxs, "j_idxs": j_idxs}
            # now need inverse idx to align i\to j w j\to i
            just_is = i_idxs[:: self.Q]
            self.flat_i_idxs[t] = just_is
            inv_j_idxs = np.mod(j_idxs, self.N)
            just_js = inv_j_idxs[:: self.Q]
            self.all_inv_idxs[t] = np.array(
                [
                    self.nz_idxs[j, t]
                    # only using these to index within each timestep, so don't need to place within full _psi_e
                    + np.flatnonzero(
                        just_js[self.nz_idxs[j, t] : self.nz_idxs[j + 1, t]] == i
                    )
                    # then need to find where it is specifically i sending to j
                    for i, j in zip(just_is, just_js)
                ]
            ).squeeze()

    def spatial_field_terms(
        self,
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
            ...

        Raises:
            RuntimeError: _description_

        Returns:
            field_term (_type_): ... x Q array containing value of term corresponding to
                                    each nbr so can update corresponding message
        """

        # should have nz_idxs = np.vstack((np.zeros(T,dtype=int),np.cumsum(bin_in_degs,axis=0)),
        # where bin_in_degs = bin_degs[:,:,0] = (A!=0).sum(axis=0)
        # as then _psi_e[Q*N*t+msgs.flatten()][nz_idxs[i,t]*Q:nz_idxs[i+1,t]*Q] gives all messages from i at t
        # w msgs as below
        # this works as only extracting a sum(in_degs[:,t]) size slice of _psi_e at each t, and structured msg_idxs
        # s.t. extract all msgs from i at t in sequence, for each connected j, for each q, so total of in_degs[i,t]*Q msgs for i at t
        field_terms = np.zeros((self.E_idxs[-1], self.Q))
        for t in range(self.T):
            beta = self.block_edge_prob[:, :, t]
            # msg_idxs[nz_idxs[i,t]:nz_idxs[i+1,t],:]+Q*N*t would give idxs of j in psi_e which connect to i at t, i.e. exactly what we want
            # so now just need to match multiplicities of msgs to in degree of i, multiplied by Q (as msgs for each q)

            # Want \sum_r psi_r^{j to i} beta_rqt, for each j, i, so can get from
            # as _psi_e[j_idxs,i_idxs].T.reshape(-1,Q) gives all messages from all j to i at t, in order of
            # ...,psi_1^{j\to i},...,psi_Q^{j\to i},psi_1^{k\to i},...,psi_Q^{k\to i},..., for j,k \in N_i
            # so now have spatial term for all j to all i
            i_idxs, j_idxs = self.all_idxs[t]["i_idxs"], self.all_idxs[t]["j_idxs"]
            if self.deg_corr:
                raise NotImplementedError("deg_corr not implemented")
                # for q in range(Q):
                #     for r in range(Q):
                #         tmp[q] += dc_lkl[e_nbrs_inv[nbr_idx], r, q] * jtoi_msgs[r]
            else:
                field_terms[self.E_idxs[t] : self.E_idxs[t + 1], :] = (
                    self._psi_e[j_idxs, i_idxs].T.reshape(-1, self.Q) @ beta
                )

        return field_terms

    @property
    def dc_lkl(self):
        try:
            return self._dc_lkl
        except AttributeError:
            self._dc_lkl = np.zeros((self.E_idxs[-1], self.Q, self.Q))
            for t in range(self.T):
                self._dc_lkl[
                    self.E_idxs[t] : self.E_idxs[t + 1], :, :
                ] = self.model.dc_lkl[t]
            return self._dc_lkl

    @property
    def block_edge_prob(self):
        if self.deg_corr:
            # will refer to as beta, but really omega or lambda (DC edge factor)
            return self.model._lam
        else:
            # return beta_{qr}^t as Q x Q x T array
            return self.model._beta

    # def nb_spatial_msg_term_large_deg(
    #     nz_idxs,
    #     deg_corr,
    #     bin_degs,
    #     degs,
    #     sym_A,
    #     dc_lkl,
    #     _h,
    #     meta_prob,
    #     block_edge_prob,
    #     _psi_e,
    # ):
    #     """
    #     NB again not sensible in numpy framework - compute all field terms above so can use directly
    #     Same as spatial_msg_term_small_deg but for node i that has degree within timestep t
    #     larger than specified threshold - basically just handle logs + subtract max value
    #     before exponentiating and normalising instead for numerical stability.

    #     Eqns
    #     \psi_q^{i}(t) \propto \prod_{k \in N_i} \sum_r \psi_r^{k\to i}(t)\beta_{rqt}
    #     \psi_q^{i\to j}(t) = \psi_q^{i}(t) / \psi_q^{j\to i}(t)

    #     OK instead flatten into sparse array of dims T*Q*N, N, s.t.
    #     _psi_e[t*QN + q*N + i,j] = \psi_q^{i\to j}(t)_e

    #     so _psi_e[t*QN + np.meshgrid(N*np.arange(Q), np.flatnonzero(A[i,:,t])),i] gives {\psi_r^{j\to i}(t) for r in range(Q), for j \in N_i}
    #     in shape [|N_i| x Q]

    #     So psi_e in shape [T*Q*N x N] and above eqns suggest
    #     psi_q^i(t)_e = [N x T x Q] = np.prod([i x N]*beta[r,:,t] for r in range(Q)])

    #     Args:
    #         i (_type_): _description_
    #         t (_type_): _description_
    #         nbrs (_type_): _description_

    #     Returns:
    #         _type_: _description_
    #     """
    #     N = degs.shape[0]
    #     T = degs.shape[1]
    #     Q = block_edge_prob.shape[0]
    #     log_msg = np.zeros((N, T, Q))
    #     for t in range(T):
    #         beta = block_edge_prob[:, :, t]
    #         msg_idxs = (
    #             np.array(np.meshgrid(N * np.arange(Q), np.flatnonzero(A[:, :, t].T)))
    #             .sum(axis=0)
    #             .flatten()
    #             .reshape(-1, Q)
    #         )
    #         msg_idxs -= np.repeat(N * np.arange(N), bin_degs[:, t, 0])[:, np.newaxis]
    #         i_idxs = np.repeat(np.arange(N), bin_degs[:, t, 0] * Q).flatten()
    #         j_idxs = msg_idxs.flatten() + Q * N * t

    #         max_log_msg = -1000000.0
    #         if deg_corr:
    #             raise NotImplementedError("deg_corr not implemented")
    #             # for q in range(Q):
    #             #     for r in range(Q):
    #             #         tmp[q] += dc_lkl[e_idx, r, q] * jtoi_msgs[r]
    #         else:
    #             log_field_terms = _psi_e[j_idxs, i_idxs].T.reshape(-1, Q) @ beta
    #             log_field_terms[log_field_terms < TOL] = TOL

    #         log_field_terms = np.log(log_field_terms)

    #         log_msg[:, t, :] = np.array(
    #             [
    #                 np.sum(log_field_terms[nz_idxs[i, t] : nz_idxs[i + 1, t], :], axis=0)
    #                 for i in range(N)
    #             ]
    #         ).squeeze()
    #         max_msg_log = log_msg.max(axis=-1)
    #         max_msg_log[max_msg_log < max_log_msg] = max_log_msg
    #     if deg_corr:
    #         log_msg -= np.einsum("qt,it->itq", _h, degs[:, :, 1])
    #     else:
    #         log_msg -= _h[
    #             np.newaxis, :, :
    #         ]  # NB don't need / N as using p_ab to calc, not c_ab
    #     log_msg += np.log(meta_prob)
    #     return log_msg, max_msg_log, log_field_terms

    def calc_h(
        self,
    ):
        # update within each timestep is unchanged from static case,
        # i.e. = \sum_r \sum_i \psi_r^{it} p_{rq}^t

        if self.deg_corr:
            self._h = np.einsum(
                "itr,rqt,it->qt",
                self.node_marg,
                self.block_edge_prob,
                self.degs[:, :, 1],
            )
        else:
            self._h = np.einsum("itr,rqt->qt", self.node_marg, self.block_edge_prob)
        # print("h after init:", _h)

    # def np_update_h(Q, sign, i, t, degs, deg_corr, block_edge_prob, _h, node_marg):
    # actually no need of this in numpy version, as synchronous updates mean need to completely recalculate h each time
    #     # _h[:, t] += (
    #     #     sign
    #     #     * np.ascontiguousarray(block_edge_prob[:, :, t].T)
    #     #     @ np.ascontiguousarray(node_marg[i, t, :])
    #     # )
    #     if deg_corr:

    #         for q in range(Q):
    #             for r in range(Q):
    #                 _h[q, t] += (
    #                     sign * block_edge_prob[r, q, t] * node_marg[i, t, r] * degs[i, t, 1]
    #                 )
    #     else:
    #         # try:
    #         #     assert np.isnan(node_marg[i, t, :]).sum() == 0
    #         # except:
    #         #     print("i, t:", i, t)
    #         #     print("node_marg:", node_marg[i, t, :])
    #         #     raise ValueError("Problem with node marg")
    #         for q in range(Q):
    #             for r in range(Q):
    #                 _h[q, t] += sign * block_edge_prob[r, q, t] * node_marg[i, t, r]

    def update_node_marg(
        self,
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

        self.msg_diff = 0.0
        # handles missing nodes correctly (?)
        # get spatial
        spatial_field_terms = self.spatial_field_terms()
        # just leave as doing via logs, should be fine and probably faster
        # large_degs = degs[:,:,0] > LARGE_DEG_THR

        # log_msg[large_degs, :] = np.array(
        #         [
        #             np.sum(np.log(spatial_field_terms[nz_idxs[i, t] : nz_idxs[i + 1, t], :]), axis=0)
        #             for i,t in zip(*large_degs.nonzero())
        #         ]
        #     )
        log_spatial_msg = np.stack(
            [
                np.array(
                    [
                        np.sum(
                            np.log(
                                spatial_field_terms[
                                    self.E_idxs[t]
                                    + self.nz_idxs[i, t] : self.E_idxs[t]
                                    + self.nz_idxs[i + 1, t],
                                    :,
                                ]
                            ),
                            axis=0,
                        )
                        for i in range(self.N)
                    ]
                )
                for t in range(self.T)
            ],
            axis=1,
        )
        max_log_msg = -1000000.0
        max_msg_log = log_spatial_msg.max(axis=-1)
        max_msg_log[max_msg_log < max_log_msg] = max_log_msg
        if self.deg_corr:
            log_spatial_msg -= np.einsum("qt,it->itq", self._h, self.degs[:, :, 1])
        else:
            log_spatial_msg -= self._h.T[
                np.newaxis, :, :
            ]  # NB don't need / N as using p_ab to calc, not c_ab
        log_spatial_msg += np.log(self.meta_prob)
        # if small_deg:
        #     # now as must do prods in chunks of in_degs[i,t], finally do need list comprehension over N
        #     msg[:, t, :] = np.array(
        #         [
        #             np.prod(field_terms[nz_idxs[i, t] : nz_idxs[i + 1, t], :], axis=0)
        #             for i in range(N)
        #         ]
        #     )
        #     # field_iter[nbr_idx, :] = tmp
        #     if deg_corr:
        #         msg *= np.exp(-np.einsum("it,qt->itq", degs[:, :, 1], _h))
        #     else:
        #         msg *= np.exp(-1.0 * _h.T[np.newaxis, ...])
        #     msg *= meta_prob

        tmp = log_spatial_msg.copy()
        tmp[:, 0, :] += np.log(self.model._alpha)[np.newaxis, :]
        log_back_term = np.log(self.backward_temp_msg_term())
        log_back_term[~self._pres_trans, :] = 0.0
        tmp[:, :-1, :] += log_back_term
        ## UPDATE BACKWARDS MESSAGES FROM i AT t ##
        tmp_backwards_msg = np.exp(tmp[:, 1:, :] - max_msg_log[:, 1:, np.newaxis])
        back_sums = tmp_backwards_msg.sum(axis=-1)
        tmp_backwards_msg = np.divide(
            tmp_backwards_msg,
            back_sums[:, :, np.newaxis],
            where=back_sums[:, :, np.newaxis] > 0,
        )
        tmp_backwards_msg[tmp_backwards_msg < TOL] = TOL
        tmp_backwards_msg /= tmp_backwards_msg.sum(axis=-1)[:, :, np.newaxis]
        self.msg_diff += np.abs(tmp_backwards_msg - self._psi_t[:, :, :, 0]).mean()
        self._psi_t[:, :, :, 0] = tmp_backwards_msg
        # include forward term now backwards term updated
        log_forward_term = np.log(self.forward_temp_msg_term())
        # use alpha where i not present at t-1
        log_forward_term[~self._pres_trans, :] = self.model._alpha[np.newaxis, :]
        tmp[:, 1:, :] += log_forward_term

        ## UPDATE SPATIAL MESSAGES FROM i AT t ##
        tmp_spatial_msg = -1.0 * np.log(spatial_field_terms).copy()
        for t in range(self.T):
            # need inv idxs for locs where i sends msgs to j
            # all_inv_idxs[t] gives order of field_term s.t.
            # all_inv_idxs[t][nz_idxs[i,t]:nz_idxs[i+1],t]
            # gives idxs of field terms corresponding to
            # messages FROM i to j in the correct order to
            # align with the block field_terms[E_idxs[t]:E_idxs[t+1]][nz_idxs[i, t] : nz_idxs[i + 1]]
            # which contains all terms corresponding to messages
            # TO i, from each j
            # NB all_idxs[t]["i_idxs"] designed for
            i_idxs = self.flat_i_idxs[t]
            inv_idxs = self.all_inv_idxs[t]
            tmp_spatial_msg[self.E_idxs[t] : self.E_idxs[t + 1]] = (
                tmp[i_idxs, t, :]
                + tmp_spatial_msg[self.E_idxs[t] : self.E_idxs[t + 1]][inv_idxs, :]
            )
        log_field_term_max = tmp_spatial_msg.max(axis=1)
        log_field_term_max[log_field_term_max < max_log_msg] = max_log_msg
        tmp_spatial_msg = np.exp(tmp_spatial_msg - log_field_term_max[:, np.newaxis])
        tmp_spat_sums = tmp_spatial_msg.sum(axis=1)
        tmp_spatial_msg = np.divide(
            tmp_spatial_msg,
            tmp_spat_sums[:, np.newaxis],
            where=tmp_spat_sums[:, np.newaxis] > 0,
        )
        tmp_spatial_msg[tmp_spatial_msg < TOL] = TOL
        tmp_spatial_msg /= tmp_spatial_msg.sum(axis=1)[:, np.newaxis]
        for t in range(self.T):
            i_idxs, j_idxs = self.all_idxs[t]["i_idxs"], self.all_idxs[t]["j_idxs"]
            self.msg_diff += np.abs(
                tmp_spatial_msg[self.E_idxs[t] : self.E_idxs[t + 1]].flatten()
                - self._psi_e[j_idxs, i_idxs]
            ).mean()  # NB need to mult by deg_i so weighted correctly
            self._psi_e[j_idxs, i_idxs] = tmp_spatial_msg[
                self.E_idxs[t] : self.E_idxs[t + 1]
            ].flatten()

        ## UPDATE FORWARDS MESSAGES FROM i AT t ##
        # just need to remove back term previously added
        tmp_forwards_msg = np.exp(
            tmp[:, :-1, :] - max_msg_log[:, :-1, np.newaxis] - log_back_term
        )
        forward_sums = tmp_forwards_msg.sum(axis=-1)
        tmp_forwards_msg = np.divide(
            tmp_forwards_msg,
            forward_sums[:, :, np.newaxis],
            where=forward_sums[:, :, np.newaxis] > 0,
        )
        tmp_forwards_msg[tmp_forwards_msg < TOL] = TOL
        tmp_forwards_msg /= tmp_forwards_msg.sum(axis=-1)[:, :, np.newaxis]
        self.msg_diff += np.abs(tmp_forwards_msg - self._psi_t[..., 1]).mean()
        self._psi_t[:, :, :, 1] = tmp_forwards_msg

        ## UPDATE MARGINAL OF i AT t ##
        tmp_marg = np.exp(tmp - max_msg_log[:, :, np.newaxis])
        marg_sums = tmp_marg.sum(axis=-1)
        tmp_marg = np.divide(
            tmp_marg, marg_sums[:, :, np.newaxis], where=marg_sums[:, :, np.newaxis] > 0
        )
        tmp_marg[tmp_marg < TOL] = TOL
        tmp_marg /= tmp_marg.sum(axis=-1)[:, :, np.newaxis]
        self.node_marg = tmp_marg

        if np.isnan(self.msg_diff).sum() > 0:
            if np.isnan(self.node_marg).sum() > 0:
                print("nans for node marg @ (i,t):")
                print(*np.array(np.nonzero(np.isnan(self.node_marg))), sep="\n")
            if np.isnan(self._psi_e).sum() > 0:
                print("nans for psi_e @ (i,t):")
                print(*np.array(np.nonzero(np.isnan(self._psi_e))), sep="\n")
            if np.isnan(self._psi_t).sum() > 0:
                print("nans for psi_t @ (i,t):")
                print(*np.array(np.nonzero(np.isnan(self._psi_t))), sep="\n")
            raise RuntimeError("Problem updating messages")
        self.calc_h()

    def compute_free_energy(
        self,
    ):
        # see e.g. https://arxiv.org/pdf/1109.3041.pdf
        f_site = 0.0  # = \sum_{i,t} log(Z^{i,t}) for Z normalising marginal of i at t
        f_spatlink = 0.0  # = \sum_{ijt \in \mathcal{E}} log(Z^{ij,t}) for Z normalising the twopoint marginal
        f_templink = 0.0  # = \sum_{i,t \in 0:T-1} log(Z^{i,t,t+1})  for Z normalising the twopoint spatial marginal between i at t and t+1
        last_term = 0.0  # = \sum_{ij,t \not\in \mathcal{E}} log(\tilde{Z}^{ijt}) where \tilde{Z} is the normalising constant for the twopoint marginal between i and j at t
        # where now i is not connected to j, and so single marginals are used
        # - approximating the sum to all ij, as the net is sparse
        # and using log(1-x) \approx -x for small x
        # in the static case adding this reduces to subtracting the average degree
        # (as MLE for \alpha is 1/N \sum_i \psi^{i,0}), but not for us

        # get spatial
        spatial_field_terms = self.spatial_field_terms()

        log_spatial_msg = np.stack(
            [
                np.array(
                    [
                        np.sum(
                            np.log(
                                spatial_field_terms[
                                    self.E_idxs[t]
                                    + self.nz_idxs[i, t] : self.E_idxs[t]
                                    + self.nz_idxs[i + 1, t],
                                    :,
                                ]
                            ),
                            axis=0,
                        )
                        for i in range(self.N)
                    ]
                )
                for t in range(self.T)
            ],
            axis=1,
        )

        if self.deg_corr:
            log_spatial_msg -= np.einsum("qt,it->itq", self._h, self.degs[:, :, 1])
        else:
            log_spatial_msg -= self._h.T[
                np.newaxis, :, :
            ]  # NB don't need / N as using p_ab to calc, not c_ab
        log_spatial_msg += np.log(self.meta_prob)

        tmp = log_spatial_msg
        # add alpha
        tmp[:, 0, :] += np.log(self.model._alpha)[np.newaxis, :]
        # include backward msgs
        log_back_term = np.log(self.backward_temp_msg_term())
        log_back_term[~self._pres_trans, :] = 0.0
        tmp[:, :-1, :] += log_back_term
        # include forward term
        log_forward_term = np.log(self.forward_temp_msg_term())
        # use alpha where i not present at t-1
        log_forward_term[~self._pres_trans, :] = self.model._alpha[np.newaxis, :]
        tmp[:, 1:, :] += log_forward_term
        tmp_marg = np.exp(tmp)
        f_site += np.log(tmp_marg.sum(axis=-1)).sum()
        f_site /= self.N * self.T

        # calc twopoint marg terms
        unnorm_twopoint_e_marg = np.zeros((self.E_idxs[-1], self.Q, self.Q))
        for t in range(self.T):
            i_idxs, j_idxs = self.all_idxs[t]["i_idxs"], self.all_idxs[t]["j_idxs"]
            inv_idxs = self.all_inv_idxs[t]
            jtoi_msgs = self._psi_e[j_idxs, i_idxs].reshape(-1, self.Q)
            itoj_msgs = jtoi_msgs[inv_idxs, :]
            unnorm_twopoint_e_marg[
                self.E_idxs[t] : self.E_idxs[t + 1], :, :
            ] += np.einsum("iq,ir->iqr", jtoi_msgs, itoj_msgs)
            if not self.directed:
                tmp = np.einsum("iq,ir->iqr", itoj_msgs, jtoi_msgs)
                diags = np.diag_indices(self.Q, ndim=2)
                # don't double diagonals
                tmp[:, diags] = 0
                unnorm_twopoint_e_marg[self.E_idxs[t] : self.E_idxs[t + 1]] += tmp

            if self.deg_corr:
                # NB make sure dc lkl is suitably symmetrised if undirected
                unnorm_twopoint_e_marg *= self.dc_lkl
            else:
                for t in range(self.T):
                    unnorm_twopoint_e_marg[
                        self.E_idxs[t] : self.E_idxs[t + 1]
                    ] *= self.block_edge_prob[np.newaxis, :, :, t]
        f_spatlink = (
            np.log(unnorm_twopoint_e_marg.sum(axis=(-2, -1))).sum() / self.N * self.T
        )
        unnorm_twopoint_t_marg = np.einsum(
            "itq,itr,qr->itqr",
            self._psi_t[..., 1],
            self._psi_t[..., 0],
            self.trans_prob,
        )
        f_templink = (
            np.log(unnorm_twopoint_t_marg.sum(axis=(-2, -1))).sum() / self.N * self.T
        )

        # calc last term
        marg_sums = self.node_marg.sum(axis=0)
        last_term = np.einsum("qrt,tq,tr->", self.model._beta, marg_sums, marg_sums)
        last_term /= self.N * self.T

        # if self.verbose:
        # print("Spatial link energy: ", f_spatlink)
        # print("Temporal link energy: ", f_templink)
        # print("Site energy: ", f_site)
        # print("Non-link energy: ", last_term)
        self.free_energy = f_spatlink + f_templink - f_site - last_term
        return self.free_energy

    def update_twopoint_marginals(
        self,
    ):
        # node_marg = None
        if not self.frozen:
            self.update_twopoint_spatial_marg()
            if self.verbose:
                print("\tUpdated twopoint spatial marg")
            self.update_twopoint_temp_marg()
            if self.verbose:
                print("\tUpdated twopoint temp marg")
            # twopoint_marginals = [twopoint_e_marg, twopoint_t_marg]
            return (self.twopoint_e_marg, self.twopoint_t_marg)
        else:
            if self.verbose:
                print("\tDSBMM params frozen, no need to update twopoint marginals")

    def update_twopoint_spatial_marg(self):
        self.twopoint_e_marg = np.zeros((self.E_idxs[-1], self.Q, self.Q))
        for t in range(self.T):
            i_idxs, j_idxs = self.all_idxs[t]["i_idxs"], self.all_idxs[t]["j_idxs"]
            inv_idxs = self.all_inv_idxs[t]
            jtoi_msgs = self._psi_e[j_idxs, i_idxs].reshape(-1, self.Q)
            itoj_msgs = jtoi_msgs[inv_idxs, :]
            self.twopoint_e_marg[
                self.E_idxs[t] : self.E_idxs[t + 1], :, :
            ] += np.einsum("iq,ir->iqr", jtoi_msgs, itoj_msgs)
            if not self.directed:
                tmp = np.einsum("iq,ir->iqr", itoj_msgs, jtoi_msgs)
                diags = np.diag_indices(self.Q, ndim=2)
                # don't double diagonals
                tmp[:, diags] = 0
                self.twopoint_e_marg[self.E_idxs[t] : self.E_idxs[t + 1]] += tmp

            if self.deg_corr:
                # NB make sure dc lkl is suitably symmetrised if undirected
                self.twopoint_e_marg *= self.dc_lkl
            else:
                for t in range(self.T):
                    self.twopoint_e_marg[
                        self.E_idxs[t] : self.E_idxs[t + 1]
                    ] *= self.block_edge_prob[np.newaxis, :, :, t]
            tp_e_sums = self.twopoint_e_marg.sum(axis=(-2, -1))
            self.twopoint_e_marg = np.divide(
                self.twopoint_e_marg,
                tp_e_sums[:, np.newaxis, np.newaxis],
                where=tp_e_sums[:, np.newaxis, np.newaxis] > 0,
            )
            self.twopoint_e_marg[self.twopoint_e_marg < TOL] = TOL
            self.twopoint_e_marg /= self.twopoint_e_marg.sum(axis=(-2, -1))[
                :, np.newaxis, np.newaxis
            ]
        return self.twopoint_e_marg

    def update_twopoint_temp_marg(self):
        # recall t msgs in shape (i,t,q,2), w t from 0 to T-2, and final dim (backwards from t+1, forwards from t)
        self.twopoint_t_marg = np.einsum(
            "itq,itr,qr->itqr",
            self._psi_t[..., 1],
            self._psi_t[..., 0],
            self.trans_prob,
        )
        tp_t_sums = self.twopoint_t_marg.sum(axis=(-2, -1))
        self.twopoint_t_marg = np.divide(
            self.twopoint_t_marg,
            tp_t_sums[..., np.newaxis, np.newaxis],
            where=tp_t_sums[..., np.newaxis, np.newaxis] > 0,
        )
        self.twopoint_t_marg[self.twopoint_t_marg < TOL] = TOL
        self.twopoint_t_marg /= self.twopoint_t_marg.sum(axis=(-2, -1))[
            ..., np.newaxis, np.newaxis
        ]
        return self.twopoint_t_marg

    def onehot_initialization(self, a):
        ncols = a.max() + 1
        out = np.zeros((a.size, ncols), dtype=np.uint8)
        out[np.arange(a.size), a.ravel()] = 1
        out.shape = a.shape + (ncols,)
        return out

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