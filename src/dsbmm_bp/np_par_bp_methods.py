# numpy reimplementation of all methods for BP class that reasonably stand to gain from doing so
import numpy as np
import yaml  # type: ignore
from numba import njit
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
    TOL = 5e-6
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
    def log_meta_prob(self):
        # NB DSBMM already performed
        # meta_lkl = meta_lkl**tuning_param
        # so adjusted suitably already
        return self.model.log_meta_lkl

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
        if self.N > 1000:
            print(f"{len(self._psi_e.data)} messages in full system")
        if mode == "random":
            # initialise by random messages and marginals
            self.node_marg = np.random.rand(self.N, self.T, self.Q)
            self.node_marg /= self.node_marg.sum(axis=2, keepdims=True)
            self.node_marg[~self._pres_nodes] = 0.0
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
            self._psi_t[~self._pres_trans] = 0.0
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
            one_hot_Z[~self._pres_nodes] = 0.0
            self.node_marg = p * one_hot_Z + (1 - p) * np.random.rand(
                self.N, self.T, self.Q
            )
            self.node_marg /= self.node_marg.sum(axis=-1)[:, :, np.newaxis]
            self.node_marg[~self._pres_nodes] = 0.0

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

            @njit
            def update_data_locs(tmp_data, tmp_indptr, locs, p):
                for i in locs:
                    tmp_data[tmp_indptr[i] : tmp_indptr[i + 1]] *= p
                return tmp_data

            tmp.data = update_data_locs(tmp.data, tmp_indptr, Z_idxs, p)
            other_idxs = np.arange(self.N * self.T * self.Q)
            other_idxs = np.setdiff1d(other_idxs, Z_idxs)
            tmp.data = update_data_locs(tmp.data, tmp_indptr, other_idxs, 0)

            self._psi_e.data = np.random.rand(len(self._psi_e.data))

            if self.verbose and self.N > 1000:
                print("Normalising init messages...")
            if self.N > 1000:

                @njit
                def normalise_psi_e(psi_data, psi_indptr, N, T, Q):
                    for t in range(T):
                        q_idxs = np.arange(N * T * Q).reshape(T, Q, N)[t].T
                        for i in range(N):
                            i_sums = np.zeros(
                                psi_indptr[q_idxs[i, 0] + 1] - psi_indptr[q_idxs[i, 0]]
                            )
                            for q in range(Q):
                                i_sums += psi_data[
                                    psi_indptr[q_idxs[i, q]] : psi_indptr[
                                        q_idxs[i, q] + 1
                                    ]
                                ]
                            # assert np.all(i_sums>0)
                            for q in range(Q):
                                psi_data[
                                    psi_indptr[q_idxs[i, q]] : psi_indptr[
                                        q_idxs[i, q] + 1
                                    ]
                                ] /= i_sums
                    return psi_data

                self._psi_e.data = normalise_psi_e(
                    self._psi_e.data, self._psi_e.indptr, self.N, self.T, self.Q
                )

            else:
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

            if self.verbose and self.N > 1000:
                print("Done for random init...")

            # renormalise just in case
            if self.verbose and self.N > 1000:
                print("Now added planted information, calculating renormalisation...")
            if self.N > 1000:
                self._psi_e.data = normalise_psi_e(
                    self._psi_e.data, self._psi_e.indptr, self.N, self.T, self.Q
                )
                if self.verbose:
                    print("Done!")
            else:
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

        self._psi_t[~self._pres_trans, :, :] = 0.0
        self.n_tot_msgs = self._psi_e.nnz + np.count_nonzero(self._psi_t)

    @property
    def trans_prob(self):
        return self.model._pi

    def forward_temp_msg_term(self):
        # sum_qprime(trans_prob(qprime,q)*_psi_t[i,t-1,qprime,1])
        # from t-1 to t
        # _psi_t in shape [N, T - 1, Q, (backwards from t+1, forwards from t)]
        # so return gives msg term for i belonging to q at t to i at t+1 for t<T in i,t,q
        # out = np.einsum("itr,rq->itq", self._psi_t[:, :, :, 1], self.trans_prob)
        out = np.nansum(
            self._psi_t[..., 1][..., np.newaxis]
            * self.trans_prob[np.newaxis, np.newaxis, ...],
            axis=-2,
        )
        # out[out < TOL] = TOL
        # REMOVE:
        try:
            assert np.all(
                out[self._pres_trans, :] > 0
            )  # even in case of rho=1, should have this term = pi_{x^{i(t-1)}q}
        except AssertionError:
            print(out[(out <= 0) & self._pres_trans[..., np.newaxis]])
            print(self._psi_t[out == 0, 1])
            print(np.isnan(out[self._pres_trans]).sum())
            print(self.trans_prob)
            raise RuntimeError("Problem w forward msg term")
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
        # out = np.einsum("itr,qr->itq", self._psi_t[..., 0], self.trans_prob)
        out = np.nansum(
            self._psi_t[..., 0][..., np.newaxis]
            * self.trans_prob[np.newaxis, np.newaxis, ...],
            axis=-1,
        )
        # out[out < TOL] = TOL
        # REMOVE:
        try:
            assert np.all(
                out[self._pres_trans, :] > 0
            )  # even in case of rho=1, should have this term = pi_{qx^{i(t+1)}}
        except AssertionError:
            print(out[(out <= 0) & self._pres_trans[..., np.newaxis]])
            print(self._psi_t[out == 0, 0])
            print(np.isnan(out[self._pres_trans]).sum())
            print(self.trans_prob)
            raise RuntimeError("Problem w backward msg term")
        return out

    def construct_edge_idxs_and_inv(self):
        # NB idxs here only used for _psi_e, so should operate on the symmetrised + binarised version of A,
        # as even in directed case a directed edge from i->j still means that there will be messages j->i, as they
        # emerge from considering joint factors in the model, and this term still looks like p(a_ij|z_i,z_j).
        sym_A = [((self.A[t] + self.A[t].T) != 0) * 1.0 for t in range(self.T)]
        self.bin_degs = np.array(
            np.vstack([(sym_A[t] != 0).sum(axis=0).squeeze() for t in range(self.T)]).T
        )
        self.nz_idxs = {}
        self.nz_is = {}
        cumdegs = np.cumsum(self.bin_degs, axis=0)
        for t in range(self.T):
            unq_cumdeg, nz_is = np.unique(
                cumdegs[:, t], return_index=True
            )  # unique necessary for missing nodes, with degree zero (and so no change in bin_degs.cumsum())
            # handle case of node 0 missing separately, else will count as present below
            if cumdegs[0, t] == 0:
                nz_is = nz_is[unq_cumdeg != 0]
                unq_cumdeg = unq_cumdeg[unq_cumdeg != 0]
            self.nz_idxs[t] = np.concatenate([[0], unq_cumdeg], dtype=int)
            self.nz_is[t] = np.concatenate(
                [nz_is, [self.N]], dtype=int
            )  # extend to make same length - now ns_is[t] == i gives posn of i in nz_idxs[t] that itself gives e_idx start posn of i at t
        self.E_idxs = np.concatenate([[0], self.bin_degs.sum(axis=0).cumsum()])
        self.all_idxs = {}
        self.flat_i_idxs = {}
        self.all_inv_idxs = {}
        for t in range(self.T):
            # np.flatnonzero fails for sparse matrix,
            # but can easily get equivalent using
            row_idx, col_idx = sym_A[t].T.nonzero()
            # handle integer overflow problem for v large nets
            nelems = self.N**2
            nbits = 2**32  # as scipy sparse by default uses int32 for idxs
            if nelems > nbits / 2:
                if nelems < nbits:
                    if t == 0:
                        print("Large net: converting idxs from int32 to uint32...")
                    row_idx, col_idx = row_idx.astype(np.uint32), col_idx.astype(
                        np.uint32
                    )
                else:
                    nbitsmax = 2**64
                    try:
                        assert nelems < nbitsmax
                        assert self.N < nbits
                    except AssertionError:
                        raise ValueError("Nets bigger than max size currently allowed.")
                    if t == 0:
                        print("Very large net: converting idxs from int32 to uint64...")
                    row_idx, col_idx = row_idx.astype(np.uint64), col_idx.astype(
                        np.uint64
                    )

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
            just_is = i_idxs[:: self.Q].astype(int)
            self.flat_i_idxs[t] = just_is
            just_js = np.mod(j_idxs[:: self.Q], self.N).astype(int)
            try:
                self.all_inv_idxs[t] = np.array(
                    [
                        self.nz_idxs[t][self.nz_is[t] == j][0]
                        # only using these to index within each timestep, so don't need to place within full _psi_e
                        + np.flatnonzero(
                            just_js[
                                self.nz_idxs[t][self.nz_is[t] == j][0] : self.nz_idxs[
                                    t
                                ][np.flatnonzero(self.nz_is[t] == j)[0] + 1]
                            ]
                            == i
                        )[0]
                        # then need to find where it is specifically i sending to j - take 0 for only data rather than array, as should have no multi-edges so only one such idx
                        for i, j in zip(just_is, just_js)
                    ]
                ).squeeze()
            except IndexError:
                for i, j in zip(just_is, just_js):
                    try:
                        start = self.nz_idxs[t][self.nz_is[t] == j][0]
                        stop = self.nz_idxs[t][
                            np.flatnonzero(self.nz_is[t] == j)[0] + 1
                        ]
                        test_i = self.flat_i_idxs[0][start:stop]
                        test_j = just_js[start:stop]
                        test_ij = np.flatnonzero(test_j == i)
                        test_ijval = test_ij[0]
                    except:
                        print(i, j, t)
                        print(start)
                        print(stop)
                        print(test_i)
                        print(test_j)
                        print(sym_A[t][test_i, test_j])
                        print(sym_A[t][test_j, test_i])
                        print(test_ij)
                        print(test_ijval)
                        raise RuntimeError(
                            "Problem w idxs, likely integer overflow error that should have been caught..."
                        )

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
        # REMOVE:
        try:
            assert np.all(field_terms.sum(axis=-1) > 0)
        except AssertionError:
            print(np.nonzero(field_terms <= 0.0))
            print(np.count_nonzero(field_terms == 0))
            print(np.count_nonzero(field_terms < 0))
            print(np.isnan(field_terms).sum())
            raise RuntimeError("Problem w spatial field terms")
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
            # self._h = np.einsum(
            #     "itr,rqt,it->qt",
            #     self.node_marg,
            #     self.block_edge_prob,
            #     self.degs[:, :, 1],
            # )
            self._h = np.nansum(
                self.node_marg[..., np.newaxis]
                * self.block_edge_prob.transpose(2, 0, 1)[np.newaxis, ...]
                * self.degs[:, :, 1][..., np.newaxis, np.newaxis],
                axis=(0, -2),
            ).T
        else:
            # self._h = np.einsum("itr,rqt->qt", self.node_marg, self.block_edge_prob)
            self._h = np.nansum(
                self.node_marg[..., np.newaxis]
                * self.block_edge_prob.transpose(2, 0, 1)[np.newaxis, ...],
                axis=(0, -2),
            ).T
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
                                    + self.nz_idxs[t][self.nz_is[t] == i][
                                        0
                                    ] : self.E_idxs[t]
                                    + self.nz_idxs[t][
                                        np.flatnonzero(self.nz_is[t] == i)[0] + 1
                                    ],
                                    :,
                                ]
                            ),
                            axis=0,
                        )
                        if self._pres_nodes[i, t]
                        else np.zeros(
                            self.Q
                        )  # need zeros rather than nan / empty else will be included in einsums
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
        log_spatial_msg += self.log_meta_prob
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
        back_term = self.backward_temp_msg_term()
        log_back_term = np.log(
            back_term,
            where=self._pres_trans[:, :, np.newaxis],
            out=np.zeros_like(back_term),
        )
        # log_back_term[~self._pres_trans, :] = 0.0
        tmp[:, :-1, :] += log_back_term
        ## UPDATE BACKWARDS MESSAGES FROM i AT t ##
        max_log_msg = -1000000.0
        max_back_msg_log = tmp[:, 1:, :].max(axis=-1, keepdims=True)
        max_back_msg_log[max_back_msg_log < max_log_msg] = max_log_msg
        tmp_backwards_msg = np.exp(tmp[:, 1:, :] - max_back_msg_log)
        back_sums = tmp_backwards_msg.sum(axis=-1, keepdims=True)
        # REMOVE:
        try:
            assert np.all(back_sums[self._pres_trans] > 0)
        except AssertionError:
            print(back_sums[self._pres_trans & back_sums <= 0])
            print(np.count_nonzero(np.isnan(back_sums[self._pres_trans])))
            raise RuntimeError("Problem w backwards msg")
        tmp_backwards_msg = np.divide(
            tmp_backwards_msg,
            back_sums,
            where=back_sums > 0,
            out=np.zeros_like(tmp_backwards_msg),
        )
        # tmp_backwards_msg[tmp_backwards_msg < TOL] = TOL
        # tmp_backwards_msg /= tmp_backwards_msg.sum(axis=-1)[:, :, np.newaxis]
        tmp_backwards_msg[~self._pres_trans, :] = 0.0  # set to zero if not present
        tmp_diff = np.max(np.abs(tmp_backwards_msg - self._psi_t[:, :, :, 0]))
        try:
            assert not np.isnan(tmp_diff)
        except AssertionError:
            raise RuntimeError("Problem w backwards msg")
        self.msg_diff = max(tmp_diff, self.msg_diff)
        self._psi_t[:, :, :, 0] = tmp_backwards_msg
        # include forward term now backwards term updated
        forward_term = self.forward_temp_msg_term()
        log_forward_term = np.log(
            forward_term,
            where=self._pres_trans[:, :, np.newaxis],
            # out=2 * np.log(TOL) * np.ones_like(forward_term),
            out=np.zeros_like(forward_term),
        )
        # use alpha where i not present at t-1 if i present at t
        log_forward_term[~self._pres_trans, :] = self.model._alpha[np.newaxis, :]
        log_forward_term[~self._pres_nodes[:, 1:], :] = 0.0
        tmp[:, 1:, :] += log_forward_term

        ## UPDATE SPATIAL MESSAGES FROM i AT t ##
        tmp_spatial_msg = -1.0 * np.log(spatial_field_terms).copy()
        for t in range(self.T):
            # need inv idxs for locs where i sends msgs to j
            # all_inv_idxs[t] gives order of field_term s.t.
            # all_inv_idxs[t][nz_idx[t][i]:nz_idxs[t][i+1]]
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
        log_field_term_max = tmp_spatial_msg.max(axis=1, keepdims=True)
        log_field_term_max[log_field_term_max < max_log_msg] = max_log_msg
        tmp_spatial_msg = np.exp(tmp_spatial_msg - log_field_term_max)
        tmp_spat_sums = tmp_spatial_msg.sum(axis=1, keepdims=True)
        tmp_spatial_msg = np.divide(
            tmp_spatial_msg,
            tmp_spat_sums,
            where=tmp_spat_sums > 0,
            out=np.zeros_like(tmp_spatial_msg),
        )
        # tmp_spatial_msg[tmp_spatial_msg < TOL] = TOL
        # tmp_spatial_msg /= tmp_spatial_msg.sum(axis=1)[:, np.newaxis]
        for t in range(self.T):
            i_idxs, j_idxs = self.all_idxs[t]["i_idxs"], self.all_idxs[t]["j_idxs"]
            tmp_diff = np.abs(
                tmp_spatial_msg[self.E_idxs[t] : self.E_idxs[t + 1]].flatten()
                - self._psi_e[j_idxs, i_idxs]
            ).max()
            try:
                assert not np.isnan(tmp_diff)
            except AssertionError:
                raise RuntimeError("Problem w spatial msg")
            self.msg_diff = max(
                tmp_diff,
                self.msg_diff,
            )
            self._psi_e[j_idxs, i_idxs] = tmp_spatial_msg[
                self.E_idxs[t] : self.E_idxs[t + 1]
            ].flatten()

        ## UPDATE FORWARDS MESSAGES FROM i AT t ##
        # just need to remove back term previously added
        log_forwards_msg = tmp[:, :-1, :] - log_back_term
        max_fwd_msg_log = log_forwards_msg.max(axis=-1, keepdims=True)
        max_fwd_msg_log[max_fwd_msg_log < max_log_msg] = max_log_msg
        tmp_forwards_msg = np.exp(log_forwards_msg - max_fwd_msg_log)
        forward_sums = tmp_forwards_msg.sum(axis=-1)
        tmp_forwards_msg = np.divide(
            tmp_forwards_msg,
            forward_sums[:, :, np.newaxis],
            where=forward_sums[:, :, np.newaxis] > 0,
            out=np.zeros_like(tmp_forwards_msg),
        )
        # tmp_forwards_msg[tmp_forwards_msg < TOL] = TOL
        # tmp_forwards_msg /= tmp_forwards_msg.sum(axis=-1)[:, :, np.newaxis]
        tmp_forwards_msg[~self._pres_trans, :] = 0.0  # set to zero if not present
        np.abs(tmp_forwards_msg - self._psi_t[..., 1]).max()
        try:
            assert not np.isnan(tmp_diff)
        except AssertionError:
            raise RuntimeError("Problem w forwards msg")
        self.msg_diff = max(tmp_diff, self.msg_diff)
        self._psi_t[:, :, :, 1] = tmp_forwards_msg

        ## UPDATE MARGINAL OF i AT t ##
        log_marg_max = tmp.max(axis=-1, keepdims=True)
        log_marg_max[log_marg_max < max_log_msg] = max_log_msg
        tmp_marg = np.exp(tmp - log_marg_max)
        marg_sums = tmp_marg.sum(axis=-1)
        tmp_marg = np.divide(
            tmp_marg,
            marg_sums[:, :, np.newaxis],
            where=marg_sums[:, :, np.newaxis] > 0,
            out=np.zeros_like(tmp_marg),
        )
        # tmp_marg[tmp_marg < TOL] = TOL
        tmp_marg = np.divide(
            tmp_marg,
            tmp_marg.sum(axis=-1, keepdims=True),
            where=tmp_marg.sum(axis=-1, keepdims=True) > 0.0,
            out=np.zeros_like(tmp_marg),
        )
        tmp_marg[~self._pres_nodes, :] = 0.0  # set to zero if not present
        self.node_marg = tmp_marg

        # self.msg_diff /= self.n_tot_msgs

        if np.isnan(self.msg_diff).sum() > 0:
            if np.isnan(self.node_marg).sum() > 0:
                print("nans for node marg @ (i,t):")
                print(*np.array(np.nonzero(np.isnan(self.node_marg))), sep="\n")
            if np.isnan(self._psi_e.data).sum() > 0:
                print("nans for psi_e")
                # print(*np.array(np.nonzero(np.isnan(self._psi_e))), sep="\n")
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
                                    + self.nz_idxs[t][self.nz_is[t] == i][
                                        0
                                    ] : self.E_idxs[t]
                                    + self.nz_idxs[t][
                                        np.flatnonzero(self.nz_is[t] == i)[0] + 1
                                    ],
                                    :,
                                ]
                            ),
                            axis=0,
                        )
                        if self._pres_nodes[i, t]
                        else np.zeros(
                            self.Q
                        )  # need zeros rather than nan / empty else will be included in einsums
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
        log_spatial_msg += self.log_meta_prob

        tmp = log_spatial_msg
        # add alpha
        tmp[:, 0, :] += np.log(self.model._alpha)[np.newaxis, :]
        # include backward msgs
        back_term = self.backward_temp_msg_term()
        log_back_term = np.log(
            back_term,
            where=self._pres_trans[:, :, np.newaxis],
            out=np.zeros_like(back_term),
        )
        # log_back_term[~self._pres_trans, :] = 0.0
        tmp[:, :-1, :] += log_back_term
        # include forward term
        forward_term = self.forward_temp_msg_term()
        log_forward_term = np.log(
            forward_term,
            where=self._pres_trans[:, :, np.newaxis],
            out=np.zeros_like(forward_term),
        )
        # use alpha where i not present at t-1, if i present at t
        log_forward_term[~self._pres_trans, :] = self.model._alpha[np.newaxis, :]
        log_forward_term[~self._pres_nodes[:, 1:], :] = 0.0
        tmp[:, 1:, :] += log_forward_term
        # log_marg_max = tmp.max(axis=-1, keepdims=True)
        tmp_marg = np.exp(
            tmp
        )  # don't subtract max here as otherwise norm sum incorrect for free energy
        # tmp_marg[tmp_marg < TOL] = TOL
        tmp_marg_sums = tmp_marg.sum(axis=-1)
        f_site += np.log(
            tmp_marg_sums,
            where=(tmp_marg_sums > 0),
            out=np.zeros_like(tmp_marg_sums),
            # out=10 * np.log(TOL) * np.ones_like(tmp_marg_sums),
        ).sum()
        f_site /= self.model._tot_N_pres

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
            np.log(unnorm_twopoint_e_marg.sum(axis=(-2, -1))).sum()
            / self.model._tot_N_pres
        )
        unnorm_twopoint_t_marg = np.einsum(
            "itq,itr,qr->itqr",
            self._psi_t[..., 1],
            self._psi_t[..., 0],
            self.trans_prob,
        )
        twopoint_t_norms = unnorm_twopoint_t_marg.sum(axis=(-2, -1))
        f_templink = (
            np.log(
                twopoint_t_norms,
                where=twopoint_t_norms > 0.0,
                out=np.zeros_like(twopoint_t_norms),
            ).sum()
            / self.model._tot_N_pres
        )

        # calc last term
        marg_sums = self.node_marg.sum(axis=0)
        last_term = np.einsum("qrt,tq,tr->", self.model._beta, marg_sums, marg_sums)
        last_term /= self.model._tot_N_pres

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
            tp_e_sums = self.twopoint_e_marg.sum(axis=(-2, -1), keepdims=True)
            self.twopoint_e_marg = np.divide(
                self.twopoint_e_marg,
                tp_e_sums,
                where=tp_e_sums > 0,
                out=np.zeros_like(self.twopoint_e_marg),
            )
            # self.twopoint_e_marg[self.twopoint_e_marg < TOL] = TOL
            # self.twopoint_e_marg = np.divide(self.twopoint_e_marg,self.twopoint_e_marg.sum(axis=(-2, -1),keepdims=True),where=
        return self.twopoint_e_marg

    def update_twopoint_temp_marg(self):
        # recall t msgs in shape (i,t,q,2), w t from 0 to T-2, and final dim (backwards from t+1, forwards from t)
        self.twopoint_t_marg = np.einsum(
            "itq,itr,qr->itqr",
            self._psi_t[..., 1],
            self._psi_t[..., 0],
            self.trans_prob,
        )
        tp_t_sums = self.twopoint_t_marg.sum(axis=(-2, -1), keepdims=True)
        self.twopoint_t_marg = np.divide(
            self.twopoint_t_marg,
            tp_t_sums,
            where=tp_t_sums > 0,
            out=np.zeros_like(self.twopoint_t_marg),
        )
        # self.twopoint_t_marg[self.twopoint_t_marg < TOL] = TOL
        # self.twopoint_t_marg /= self.twopoint_t_marg.sum(axis=(-2, -1))[
        #     ..., np.newaxis, np.newaxis
        # ]
        self.twopoint_t_marg[~self._pres_trans, ...] = 0.0  # set to zero if not present
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
