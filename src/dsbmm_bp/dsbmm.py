from numba import (
    jit,
    njit,
    prange,
    int32,
    float32,
    int64,
    float64,
    # unicode_type,
    typeof,
    gdb,
)
from numba.types import unicode_type, ListType, bool_, Array
from numba.typed import List, Dict
from numba.experimental import jitclass

# from numba_dsbmm_methods import * # TODO: implement separately to allow parallelisation + GPU usage
# from numba_bp_methods import *
import numpy as np

from utils import numba_ix, nb_poisson_lkl_int, nb_ib_lkl

# import utils

# from sklearn.cluster import MiniBatchKMeans

# TODO: don't hardcode these
TOL = 1e-50  # min value permitted for msgs etc (for numerical stability)

# NB for numba, syntax of defining an array like e.g. float32[:,:,:]
# defines array types with no particular layout
# (producing code that accepts both non-contiguous
# and contiguous arrays), but you can specify a particular
# contiguity by using the ::1 index either at the beginning
# or the end of the index specification:

# >>> numba.float32[::1]
# array(float32, 1d, C)
# >>> numba.float32[:, :, ::1]
# array(float32, 3d, C)
# >>> numba.float32[::1, :, :]
# array(float32, 3d, F)

# base_spec = [
#     ("A", float32[:]),  # an array field
#     ("X", float32[:]),  # an array field
#     ("Z", int32[:]),  # an integer array field
# ]
X_ex = List.empty_list(float64[:, :, ::1])
# X_ex.append(np.empty((1, 1, 1), dtype=np.float64))
X_type = typeof(X_ex)

meta_types_ex = List.empty_list(unicode_type)
meta_types_type = typeof(meta_types_ex)
meta_params_ex = List.empty_list(typeof(np.empty((1, 1, 1), dtype=np.float64)))
# meta_params_ex.append(np.empty((1, 1, 1), dtype=np.float64))
meta_params_type = typeof(meta_params_ex)
tp_e_marg_ex = List.empty_list(ListType(Array(float64, ndim=3, layout="C")))
# NB Array(dtype,ndim=k,layout="C") is equiv to dtype[:,...(k - 1 times),::1]
# if want Fortran layout then place ::1 in first loc rather than last
tp_e_marg_type = typeof(tp_e_marg_ex)
nbrs_ex = List.empty_list(ListType(Array(int64, ndim=1, layout="C")))
nbrs_type = typeof(nbrs_ex)
# this decorator ensures types of each base field, and means all methods are compiled into nopython fns
# further types are inferred from type annotations
@jitclass  # (base_spec)
class DSBMMBase:
    # A: np.ndarray  # assume N x N x T array s.t. (i,j,t)th position confers information about connection from i to j at time t
    A: float64[:, :, ::1]
    X: X_type
    X_poisson: float64[:, :, ::1]
    X_ib: float64[:, :, ::1]
    # X: list
    # [
    #     np.ndarray
    # ]  # assume S x N x T x Ds array s.t. (s)(i,t,ds) posn is info about ds dim of sth type metadata of i at time t
    Z: int32[:, ::1]
    T: int
    N: int
    E: int64[::1]  # num_edges in each timestep
    Q: int
    # meta_types: list[str]
    meta_types: meta_types_type
    meta_dims: int32[::1]
    deg_corr: bool
    degs: float64[:, :, ::1]  # N x T x [in,out]
    kappa: float64[:, :, ::1]  # Q x T x [in,out]
    edgemat: float64[:, :, ::1]
    deg_entropy: float
    _alpha: float64[::1]  # init group probs
    _pi: Array(float64, ndim=2, layout="C")  # group transition mat
    _lam: float64[:, :, ::1]  # block pois params in DC case
    _beta: Array(float64, ndim=3, layout="C")  # block edge probs in binary NDC case
    # _meta_params: list[np.ndarray]  # params for metadata dists
    _meta_params: meta_params_type
    node_marg: float64[:, :, ::1]
    twopoint_time_marg: float64[:, :, :, ::1]  # assume N x T - 1 x Q x Q (i,t,q,qprime)
    # twopoint_edge_marg: float64[:, :, :, :, ::1]  # assume N x N x T x Q x Q (i,j,t,q,r)
    twopoint_edge_marg: tp_e_marg_type  # [t][i][j_idx,q,r] where j_idx is idx where nbrs[t][i]==j
    meta_lkl: float64[:, :, ::1]  # N x T x Q array of meta lkl term for i at t in q
    nbrs: nbrs_type
    _edge_locs: int64[:, ::1]
    diff: float64

    def __init__(
        self,
        # data=None,
        A,
        X,
        X_poisson,
        X_ib,
        Z,
        Q,
        deg_corr,
        meta_types,
    ):
        # if data is not None:
        #     self.A = data["A"]
        #     self.X = data["X"]
        #     self.Z = data.get("Z", None)
        # else:
        self.A = A
        self.N = self.A.shape[0]
        self.E = np.array(list(map(np.count_nonzero, self.A.transpose(2, 0, 1))))
        self.T = self.A.shape[-1]
        self._edge_locs = np.array(
            list(zip(*self.A.nonzero()))
        )  # E x 3 array w rows [i,j,t] where A[i,j,t]!=0
        self.Z = Z
        self.Q = (
            Q if Q is not None else len(np.unique(self.Z))
        )  # will return 0 if both Q and Z None
        if X_poisson is not None and X_ib is not None:
            tmp = List()
            self.X_poisson = X_poisson
            self.X_ib = X_ib
            tmp.append(self.X_poisson)
            tmp.append(self.X_ib)
            # TODO: generalise meta_dims
            self.meta_dims = np.array(
                [X_poisson.shape[-1], X_ib.shape[-1]], dtype=np.int32
            )
            self.X = tmp
            tmp2 = List()
            tmp2.append(np.zeros((self.Q, self.T, self.meta_dims[0])))
            tmp2.append(np.zeros((self.Q, self.T, self.meta_dims[1])))
            self._meta_params = tmp2
        # else: # TODO: fix for loading general X
        #     self.X = X

        assert self.A is not None
        assert self.X is not None
        self.deg_corr = deg_corr
        self.degs = self.compute_degs(self.A)
        self.kappa = self.compute_group_degs()
        self.edgemat = self.compute_block_edgemat()
        self.deg_entropy = -(self.degs * np.log(self.degs)).sum()

        self.meta_types = meta_types
        tmp = List()
        for t in range(self.T):
            tmp2 = List()
            for i in range(self.N):
                tmp2.append(self.A[i, :, t].nonzero()[0])
            tmp.append(tmp2)
        self.nbrs = tmp

        self._alpha = np.zeros(self.Q)
        self._beta = np.zeros((self.Q, self.Q, self.T))
        self._lam = np.zeros((self.Q, self.Q, self.T))
        self._pi = np.zeros((self.Q, self.Q))

        tmp = List()
        for t in range(self.T):
            tmp2 = List()
            for i in range(self.N):
                tmp2.append(
                    np.zeros((len(self.nbrs[t][i]), self.Q, self.Q), dtype=np.float64)
                )
            tmp.append(tmp2)
        self.twopoint_edge_marg = tmp

        if self.Z is None:
            assert self.Q > 0
            # TODO: allow multiple types of initialisation, e.g.
            # fixed over time on concatenated adj mats and metadata,
            # just concat adj mats, and same but allowing to vary over time
            # kmeans_mat = np.concatenate(
            #     [
            #         self.A.reshape(self.N, -1),
            #         self.X.transpose(1, 2, 0, 3).reshape(self.N, -1),
            #     ],
            #     axis=1,
            # )  # done for fixing labels over time
            # TODO: uncomment after considering more
            # kmeans_labels = MiniBatchKMeans(
            #     n_clusters=self.Q,
            #     #   random_state=0, # TODO: consider allowing fixing this for reproducibility
            #     batch_size=20,
            #     max_iter=10,
            # ).fit_predict(kmeans_mat)
            # self.Z = np.tile(kmeans_labels, (1, self.T))

        self.diff = 0.0

    @property
    def num_nodes(self):
        return self.N

    @property
    def num_groups(self):
        return self.Q

    @property
    def num_timesteps(self):
        return self.T

    @property
    def get_deg_entropy(self):
        return self.deg_entropy

    @property
    def num_edges(self):
        # return number of edges in each slice - important as expect to affect BP computational complexity linearly
        return self.E

    @property
    def alpha(self):
        return self._alpha

    @property
    def pi(self):
        return self._pi

    @property
    def lam(self):
        return self._lam

    @property
    def beta(self):
        return self._beta

    @property
    def meta_params(self):
        return self._meta_params

    def get_degree(self, i, t):
        return self.degs[i, t, :]

    def get_degree_vec(self, t):
        return self.degs[:, t, :]

    def get_groups(self, t):
        return self.Z[:, t]

    def get_entropy(self):
        pass

    def compute_degs(self, A):
        """Compute in-out degree matrix from given temporal adjacency mat

        Args:
            A (_type_): _description_

        Returns:
            _type_: _description_
        """
        return np.dstack((A.sum(axis=1), A.sum(axis=0)))

    def compute_group_degs(self):
        """Compute group in- and out-degrees for current node memberships
        """
        kappa = np.zeros((self.Q, self.T, 2))
        for q in range(self.Q):
            for t in range(self.T):
                kappa[q, t, :] = self.degs[self.Z[:, t] == q, t, :].sum(axis=0)
        return kappa

    def compute_block_edgemat(self):
        """Compute number of edges between each pair of blocks

        Returns:
            _type_: _description_
        """
        edgemat = np.zeros((self.Q, self.Q, self.T))
        for q in range(self.Q):
            for r in range(self.Q):
                for t in range(self.T):
                    i_idxs = (self.Z[:, t] == q).nonzero()[0]
                    j_idxs = (self.Z[:, t] == r).nonzero()[0]
                    for i in i_idxs:
                        for j in j_idxs:
                            edgemat[q, r, t] += self.A[i, j, t]
                    # if len(row_idxs) > 0 and len(col_idxs > 0):
                    #     tmp = numba_ix(self.A[:, :, t], row_idxs, col_idxs,)
                    #     edgemat[q, r, t] = tmp.sum()

        # numpy impl
        # self.edgemat = np.array([[[self.A[np.ix_(self.Z[:,t]==q,self.Z[:,t]==r),t].sum() for t in range(self.T)]
        #                           for r in range(self.Q)]
        #                          for q in range(self.Q)])

        return edgemat

    def compute_log_likelihood(self):
        """Compute log likelihood of model for given memberships 

            In DC case this corresponds to usual DSBMM with exception of each timelice now has log lkl
                \sum_{q,r=1}^Q m_{qr} \log\frac{m_{qr}}{\kappa_q^{out}\kappa_r^{in}},
            (ignoring constants w.r.t. node memberships) 
            
        Returns:
            _type_: _description_
        """
        pass

    def update_params(self, init):
        """Given marginals, update parameters suitably

        Args:
            messages (_type_): _description_
        """
        # first init of parameters given initial groups if init=True, else use provided marginals
        self.update_alpha(init)
        self.update_pi(init)
        if self.deg_corr:
            self.update_lambda(init)
        else:
            # NB only implemented for binary case
            self.update_beta(init)
        self.update_meta_params(init)
        self.calc_meta_lkl()

    def calc_meta_lkl(self):
        """Use current meta params to calculate meta lkl terms"""
        self.meta_lkl = np.ones((self.N, self.T, self.Q))
        for s, mt in enumerate(self.meta_types):
            # print(f"Updating params for {mt} dist")
            if mt == "poisson":
                # print("In Poisson")
                pois_params = self._meta_params[s]  # shape (Q x T x 1)
                # recall X[s] has shape (N x T x Ds), w Ds = 1 here
                for t in range(self.T):
                    for q in range(self.Q):
                        for i in range(self.N):
                            # potentially could speed up further but
                            # loops still v efficient in numba
                            self.meta_lkl[i, t, q] *= nb_poisson_lkl_int(
                                self.X[s][i, t, 0], pois_params[q, t, 0]
                            )
                print("\tUpdated Poisson lkl contribution")
            elif mt == "indep bernoulli":
                # print("In IB")
                ib_params = self._meta_params[s]  # shape (Q x T x L)
                # recall X[s] has shape (N x T x Ds), w Ds = L here
                for t in range(self.T):
                    for q in range(self.Q):
                        for i in range(self.N):
                            self.meta_lkl[i, t, q] *= nb_ib_lkl(
                                self.X[s][i, t, :], ib_params[q, t, :]
                            )
                print("\tUpdated IB lkl contribution")
            else:
                raise NotImplementedError(
                    "Yet to implement metadata distribution of given type \nOptions are 'poisson' or 'indep bernoulli'"
                )  # NB can't use string formatting for print in numba

    def set_node_marg(self, values):
        self.node_marg = values

    def set_twopoint_time_marg(self, values):
        self.twopoint_time_marg = values

    def set_twopoint_edge_marg(self, values):
        self.twopoint_edge_marg = values

    def update_alpha(self, init):
        if init:
            # case of no marginals / true partition provided to calculate most likely params
            self._alpha = np.array([(self.Z[:, 0] == q).mean() for q in range(self.Q)])
            self._alpha[self._alpha < TOL] = TOL
            self._alpha /= self._alpha.sum()
        else:
            # print("Updating alpha")
            tmp = (
                self.node_marg[:, 0, :].sum(axis=0) / self.N
            )  # NB mean w axis argument not supported by numba beyond single integer
            tmp[tmp < TOL] = TOL
            tmp /= tmp.sum()
            tmp_diff = np.abs(tmp - self._alpha).mean()
            print("Alpha diff:", tmp_diff)
            self.diff += tmp_diff
            self._alpha = tmp

    def update_pi(
        self, init,
    ):
        if init:
            qqprime_trans = np.zeros((self.Q, self.Q))
            for q in range(self.Q):
                for qprime in range(self.Q):
                    for t in range(1, self.T):
                        tm1_idxs = self.Z[:, t - 1] == q
                        t_idxs = self.Z[:, t] == qprime
                        qqprime_trans[q, qprime] += (tm1_idxs * t_idxs).sum()

        else:
            qqprime_trans = self.twopoint_time_marg.sum(axis=0).sum(
                axis=0
            ) / np.expand_dims(
                self.node_marg[:, :-1, :].sum(axis=0).sum(axis=0), 1
            )  # need to do sums twice as numba axis argument only takes integers (rather than axis=(0,1) as we would want)
            # self.correct_pi()
        for q in range(self.Q):
            for qprime in range(self.Q):
                if qqprime_trans[q, qprime] < TOL:
                    qqprime_trans[q, qprime] = TOL
        qqprime_trans = qqprime_trans / np.expand_dims(
            qqprime_trans.sum(axis=-1), 1
        )  # normalise rows
        tmp_diff = np.abs(qqprime_trans - self._pi).mean()
        print("Pi diff:", tmp_diff)
        self.diff += tmp_diff
        self._pi = qqprime_trans

    def correct_pi(self):
        # print("correcting")
        for q in range(self.Q):
            for qprime in range(self.Q):
                if self._pi[q, qprime] < TOL:
                    self._pi[q, qprime] = TOL
        self._pi = self._pi / self._pi.sum(axis=-1)

    def update_lambda(self, init):
        lam_num = np.zeros((self.Q, self.Q, self.T))
        lam_den = np.zeros((self.Q, self.Q, self.T))
        if init:
            for t in range(self.T):
                for i in range(self.N):
                    for j in range(self.N):
                        lam_num[self.Z[i, t], self.Z[j, t], t] += self.A[i, j, t]
            # np.array(
            #     [
            #         [
            #             [
            #                 self.A[self.Z[:, t] == q, self.Z[:, t] == r, t].sum()
            #                 for r in range(self.Q)
            #             ]
            #             for q in range(self.Q)
            #         ]
            #         for t in range(self.T)
            #     ]
            # )
            # lam_den = np.array(
            #     [
            #         [self.degs[self.Z[:, t] == q].sum() for q in range(self.Q)]
            #         for t in range(self.T)
            #     ]
            # )
            # lam_den = np.einsum("tq,tr->tqr", lam_den, lam_den)
            for q in range(self.Q):
                for r in range(self.Q):
                    for t in range(self.T):
                        lam_den[q, r, t] = (
                            self.kappa[q, t, 0] * self.kappa[r, t, 0]
                        )  # TODO: check right in directed case
                        if lam_num[q, r, t] < TOL:
                            lam_num[q, r, t] = TOL
                        if lam_den[q, r, t] < TOL:
                            lam_den[q, r, t] = 1.0
            # lam_den = np.array(
            #     [
            #         [
            #             [
            #                 self.kappa[q, t, 0] * self.kappa[r, t, 0]
            #                 for t in range(self.T)
            #             ]
            #             for r in range(self.Q)
            #         ]
            #         for q in range(self.Q)
            #     ]
            # )
        else:
            # lam_num = np.einsum("ijtqr,ijt->qrt", self.twopoint_edge_marg, self.A)
            for q in range(self.Q):
                for r in range(self.Q):
                    for i, j, t in self._edge_locs:
                        j_idx = self.nbrs[t][i] == j
                        lam_num[q, r, t] += (
                            self.twopoint_edge_marg[t][i][j_idx, q, r] * self.A[i, j, t]
                        )
                    for t in range(self.T):
                        if lam_num[q, r, t] < TOL:
                            lam_num[q, r, t] = TOL

            # lam_den = np.einsum("itq,it->qt", self.node_marg, self.degs)
            # lam_den = np.einsum("qt,rt->qrt", lam_den, lam_den)
            marg_kappa = np.zeros((self.Q, self.T))
            for q in range(self.Q):
                for t in range(self.T):
                    marg_kappa[q, t] = (
                        self.node_marg[:, t, q] * self.degs[:, t, 0]
                    ).sum(
                        axis=0
                    )  # TODO: again check this uses right deg if directed
            for q in range(self.Q):
                for r in range(self.Q):
                    for t in range(self.T):
                        lam_den[q, r, t] = marg_kappa[q, t] * marg_kappa[r, t]
                        if lam_den[q, r, t] < TOL:
                            lam_den[q, r, t] = 1.0
        # NB use relative rather than absolute difference here as lam could be large
        tmp = lam_num / lam_den
        tmp_diff = np.abs((tmp - self._lam) / self._lam).mean()
        print("Lambda diff:", tmp_diff)
        self.diff += tmp_diff
        self._lam = tmp

    def update_beta(self, init):
        beta_num = np.zeros((self.Q, self.Q, self.T))
        beta_den = np.zeros((self.Q, self.Q, self.T))
        if init:
            # beta_num = np.array(
            #     [
            #         [
            #             [
            #                 (self.A[self.Z[:, t] == q, self.Z[:, t] == r, t] > 0).sum()
            #                 for r in range(self.Q)
            #             ]
            #             for q in range(self.Q)
            #         ]
            #         for t in range(self.T)
            #     ]
            # )
            for t in range(self.T):
                for i in range(self.N):
                    for j in range(self.N):
                        beta_num[self.Z[i, t], self.Z[j, t], t] += self.A[i, j, t]
            # beta_den = np.array(
            #     [
            #         [self.degs[self.Z[:, t] == q].sum() for q in range(self.Q)]
            #         for t in range(self.T)
            #     ]
            # )
            # beta_den = np.einsum("tq,tr->tqr", beta_den, beta_den)
            for q in range(self.Q):
                for r in range(self.Q):
                    for t in range(self.T):
                        beta_den[q, r, t] = (
                            self.kappa[q, t, 0] * self.kappa[r, t, 0]
                        )  # TODO: check right in directed case
                        if beta_num[q, r, t] < TOL:
                            beta_num[q, r, t] = TOL
                        if beta_den[q, r, t] < TOL:
                            beta_den[q, r, t] = 1.0

        else:
            # beta_num = np.einsum(
            #     "ijtqr,ijt->qrt", self.twopoint_edge_marg, (self.A > 0)
            # )
            for q in range(self.Q):
                for r in range(self.Q):
                    for i, j, t in self._edge_locs:
                        j_idx = np.nonzero(self.nbrs[t][i] == j)[0]
                        # print(self.twopoint_edge_marg[t][i][j_idx, q, r])
                        # assert j_idx.sum() == 1
                        val = self.twopoint_edge_marg[t][i][j_idx, q, r][0]
                        try:
                            assert not np.isnan(val)
                        except:
                            print("(i,j,t):", i, j, t)
                            print("A[i,j,t] = ", self.A[i, j, t])
                            print("twopoint marg: ", val)
                            raise RuntimeError("Problem updating beta")
                        beta_num[q, r, t] += val
                    for t in range(self.T):
                        if beta_num[q, r, t] < TOL:
                            beta_num[q, r, t] = TOL
            # beta_den = np.einsum("itq,it->qt", self.node_marg, self.degs)
            # beta_den = np.einsum("qt,rt->qrt", beta_den, beta_den)
            group_marg = np.zeros((self.Q, self.T))
            for q in range(self.Q):
                for t in range(self.T):
                    group_marg[q, t] = self.node_marg[:, t, q].sum(
                        axis=0
                    )  # TODO: again check this uses right deg if directed
            for q in range(self.Q):
                for r in range(self.Q):
                    for t in range(self.T):
                        beta_den[q, r, t] = group_marg[q, t] * group_marg[r, t]
                        if beta_den[q, r, t] < TOL:
                            beta_den[q, r, t] = 1.0
        tmp = beta_num / beta_den
        tmp_diff = np.abs(tmp - self._beta).mean()
        print("Beta diff:", tmp_diff)
        self.diff += tmp_diff
        self._beta = tmp

    def update_meta_params(self, init):
        for s, mt in enumerate(self.meta_types):
            # print(f"Updating params for {mt} dist")
            if mt == "poisson":
                # print("In Poisson")
                self.update_poisson_meta(s, init)
                print("\tUpdated Poisson")
            elif mt == "indep bernoulli":
                # print("In IB")
                self.update_indep_bern_meta(s, init)
                print("\tUpdated IB")
            else:
                raise NotImplementedError(
                    "Yet to implement metadata distribution of given type \nOptions are 'poisson' or 'indep bernoulli'"
                )  # NB can't use string formatting for print in numba

    def update_poisson_meta(self, s, init):
        xi = np.zeros((self.Q, self.T, 1))
        zeta = np.zeros((self.Q, self.T, 1))
        if init:
            # xi = np.array(
            #     [
            #         [(self.Z[:, t] == q).sum() for t in range(self.T)]
            #         for q in range(self.Q)
            #     ]
            # )
            for q in range(self.Q):
                for t in range(self.T):
                    xi[q, t, 0] = (self.Z[:, t] == q).sum()
                    zeta[q, t, 0] = self.X[s][self.Z[:, t] == q, t, 0].sum()
                    if xi[q, t, 0] < TOL:
                        xi[q, t, 0] = 1.0
                    if zeta[q, t, 0] < TOL:
                        zeta[q, t, 0] = TOL
            # zeta = np.array(
            #     [
            #         [self.X[s][self.Z[:, t] == q, t, 0].sum() for t in range(self.T)]
            #         for q in range(self.Q)
            #     ]
            # )
            # gdb()
        else:
            xi[:, :, 0] = self.node_marg.sum(axis=0).transpose(1, 0)
            # zeta = np.einsum("itq,itd->qt", self.node_marg, self.X[s])
            for q in range(self.Q):
                for t in range(self.T):
                    zeta[q, t, 0] = (self.node_marg[:, t, q] * self.X[s][:, t, 0]).sum()
                    if xi[q, t, 0] < TOL:
                        xi[q, t, 0] = 1.0
                    if zeta[q, t, 0] < TOL:
                        zeta[q, t, 0] = TOL
        # NB again use relative error here as could be large
        tmp = zeta / xi
        tmp_diff = np.abs((tmp - self._meta_params[s]) / self._meta_params[s]).mean()
        print("Poisson diff: ", tmp_diff)
        self.diff += tmp_diff
        self._meta_params[s] = tmp

    def update_indep_bern_meta(self, s, init):
        # TODO: handle correct normalisation
        xi = np.zeros((self.Q, self.T, 1))
        L = self.X[s].shape[-1]
        rho = np.zeros((self.Q, self.T, L))
        if init:
            # xi = np.array(
            #     [
            #         [(self.Z[:, t] == q).sum() for t in range(self.T)]
            #         for q in range(self.Q)
            #     ]
            # )
            for q in range(self.Q):
                for t in range(self.T):
                    xi[q, t, 0] = (self.Z[:, t] == q).sum()
                    rho[q, t, :] = self.X[s][self.Z[:, t] == q, t, :].sum(axis=0)
                    if xi[q, t, 0] < TOL:
                        xi[q, t, 0] = 1.0
                    for l in range(L):
                        if rho[q, t, l] < TOL:
                            rho[q, t, l] = TOL

            # rho = np.array(
            #     [
            #         [
            #             self.X[s][self.Z[:, t] == q, t, :].sum(axis=0)
            #             for t in range(self.T)
            #         ]
            #         for q in range(self.Q)
            #     ]
            # )
        else:
            xi[:, :, 0] = self.node_marg.sum(axis=0).transpose(1, 0)
            # rho = np.einsum("itq,itl->qtl", self.node_marg, self.X[s])
            for q in range(self.Q):
                for t in range(self.T):
                    rho[q, t, :] = (
                        np.expand_dims(self.node_marg[:, t, q], 1) * self.X[s][:, t, :]
                    ).sum(axis=0)
                    if xi[q, t, 0] < TOL:
                        xi[q, t, 0] = 1.0
                    for l in range(L):
                        if rho[q, t, l] < TOL:
                            rho[q, t, l] = TOL
        tmp = rho / xi
        tmp_diff = np.abs(tmp - self._meta_params[s]).mean()
        print("IB diff: ", tmp_diff)
        self.diff += tmp_diff
        self._meta_params[s] = tmp

    def zero_diff(self):
        self.diff = 0.0

    def set_Z_by_MAP(self):
        for i in range(self.N):
            for t in range(self.T):
                self.Z[i, t] = np.argmax(self.node_marg[i, t, :])


class DSBMM:
    """Pure Python wrapper around DSBMMBase to allow optional/keyword arguments
    """

    def __init__(
        self,
        # data=None,
        A=None,
        X=None,
        X_poisson=None,
        X_ib=None,
        Z=None,
        Q=None,
        deg_corr=False,
        meta_types=["poisson", "indep bernoulli"],
    ):
        # if data is not None:
        #     self.A = data["A"]
        #     self.X = data["X"]
        #     self.Z = data.get("Z", None)
        # else:
        self.jit_model = DSBMMBase(A, X, X_poisson, X_ib, Z, Q, deg_corr, meta_types)
        # self.A = A
        # if X_poisson is not None and X_ib is not None:
        #     tmp = List()
        #     self.X_poisson = X_poisson
        #     self.X_ib = X_ib
        #     tmp.append(self.X_poisson)
        #     tmp.append(self.X_ib)
        #     self.X = tmp
        # else: # TODO: fix for loading general X
        #     self.X = X
        # self.Z = Z
        # assert self.A is not None
        # assert self.X is not None

        # if self.Z is None:
        #     assert self.Q > 0
        # TODO: allow multiple types of initialisation, e.g.
        # fixed over time on concatenated adj mats and metadata,
        # just concat adj mats, and same but allowing to vary over time
        # kmeans_mat = np.concatenate(
        #     [
        #         self.A.reshape(self.N, -1),
        #         self.X.transpose(1, 2, 0, 3).reshape(self.N, -1),
        #     ],
        #     axis=1,
        # )  # done for fixing labels over time
        # TODO: uncomment after considering more
        # kmeans_labels = MiniBatchKMeans(
        #     n_clusters=self.Q,
        #     #   random_state=0, # TODO: consider allowing fixing this for reproducibility
        #     batch_size=20,
        #     max_iter=10,
        # ).fit_predict(kmeans_mat)
        # self.Z = np.tile(kmeans_labels, (1, self.T))

        # self.update_parameters()

    @property
    def num_nodes(self):
        return self.jit_model.N

    @property
    def num_groups(self):
        return self.jit_model.Q

    @property
    def num_timesteps(self):
        return self.jit_model.T

    @property
    def get_deg_entropy(self):
        return self.jit_model.deg_entropy

    @property
    def num_edges(self):
        # return number of edges in each slice - important as expect to affect BP computational complexity linearly
        return self.jit_model.E

    @property
    def alpha(self):
        return self.jit_model._alpha

    @property
    def pi(self):
        return self.jit_model._pi

    @property
    def lam(self):
        return self.jit_model._lam

    @property
    def beta(self):
        return self.jit_model._beta

    @property
    def meta_params(self):
        return self.jit_model._meta_params

    def get_degree(self, i, t):
        return self.jit_model.degs[i, t, :]

    def get_degree_vec(self, t):
        return self.jit_model.degs[:, t, :]

    def get_groups(self, t):
        return self.jit_model.Z[:, t]

    def get_entropy(self):
        pass

    def compute_degs(self, A=None):
        """Compute in-out degree matrix from given temporal adjacency mat

        Args:
            A (_type_): _description_

        Returns:
            _type_: _description_
        """
        if A is None:
            A = self.A
        return self.jit_model.compute_degs(A)

    def compute_group_degs(self):
        """Compute group in- and out-degrees for current node memberships
        """
        return self.jit_model.compute_group_degs()

    def compute_block_edgemat(self):
        """Compute number of edges between each pair of blocks

        Returns:
            _type_: _description_
        """

        # numpy impl
        # self.edgemat = np.array([[[self.A[np.ix_(self.Z[:,t]==q,self.Z[:,t]==r),t].sum() for t in range(self.T)]
        #                           for r in range(self.Q)]
        #                          for q in range(self.Q)])

        return self.jit_model.compute_block_edgemat()

    def compute_log_likelihood(self):
        """Compute log likelihood of model for given memberships 

            In DC case this corresponds to usual DSBMM with exception of each timelice now has log lkl
                \sum_{q,r=1}^Q m_{qr} \log\frac{m_{qr}}{\kappa_q^{out}\kappa_r^{in}},
            (ignoring constants w.r.t. node memberships) 
            
        Returns:
            _type_: _description_
        """
        return self.jit_model.compute_log_likelihood()

    def update_params(self, init=False):
        """Given marginals, update parameters suitably

        Args:
            messages (_type_): _description_
        """
        # first init of parameters given initial groups if init=True, else use provided marginals
        self.jit_model.update_alpha(init)
        print("\tUpdated alpha")
        self.jit_model.update_pi(init)
        print("\tUpdated pi")
        if self.jit_model.deg_corr:
            self.jit_model.update_lambda(init)
            print("\tUpdated lambda")
        else:
            # NB only implemented for binary case
            self.jit_model.update_beta(init)
            print("\tUpdated beta")
        self.jit_model.update_meta_params(init)
        print("\tUpdated meta")
        self.jit_model.calc_meta_lkl()

    def set_node_marg(self, values):
        self.jit_model.node_marg = values

    def set_twopoint_time_marg(self, values):
        self.jit_model.twopoint_time_marg = values

    def set_twopoint_edge_marg(self, values):
        self.jit_model.twopoint_edge_marg = values

    def update_alpha(self, init=False):
        self.jit_model.update_alpha(init)

    def update_pi(
        self, init=False,
    ):
        self.jit_model.update_pi(init)
        # qqprime_trans = np.array(
        #     [
        #         [
        #             [
        #                 ((self.Z[:, t - 1] == q) * (self.Z[:, t] == qprime)).sum()
        #                 for qprime in range(self.Q)
        #             ]
        #             for q in range(self.Q)
        #         ]
        #         for t in range(1, self.T)
        #     ]
        # ).sum(axis=-1)
        # print("Before norm:", qqprime_trans)
        # qqprime_trans[qqprime_trans < TOL] = TOL # can't use 2d bools in numba
        # print("After:", qqprime_trans)

    def update_lambda(self, init=False):
        # np.array(
        #     [
        #         [
        #             [
        #                 self.A[self.Z[:, t] == q, self.Z[:, t] == r, t].sum()
        #                 for r in range(self.Q)
        #             ]
        #             for q in range(self.Q)
        #         ]
        #         for t in range(self.T)
        #     ]
        # )
        # lam_den = np.array(
        #     [
        #         [self.degs[self.Z[:, t] == q].sum() for q in range(self.Q)]
        #         for t in range(self.T)
        #     ]
        # )
        # lam_den = np.einsum("tq,tr->tqr", lam_den, lam_den)
        # lam_den = np.array(
        #     [
        #         [
        #             [
        #                 self.kappa[q, t, 0] * self.kappa[r, t, 0]
        #                 for t in range(self.T)
        #             ]
        #             for r in range(self.Q)
        #         ]
        #         for q in range(self.Q)
        #     ]
        # )
        # or
        # lam_num = np.einsum("ijtqr,ijt->qrt", self.twopoint_edge_marg, self.A)
        # lam_den = np.einsum("itq,it->qt", self.node_marg, self.degs)
        # lam_den = np.einsum("qt,rt->qrt", lam_den, lam_den)
        self.jit_model.update_lambda(init)

    def update_beta(self, init=False):
        # beta_num = np.array(
        #     [
        #         [
        #             [
        #                 (self.A[self.Z[:, t] == q, self.Z[:, t] == r, t] > 0).sum()
        #                 for r in range(self.Q)
        #             ]
        #             for q in range(self.Q)
        #         ]
        #         for t in range(self.T)
        #     ]
        # )

        # beta_den = np.array(
        #     [
        #         [self.degs[self.Z[:, t] == q].sum() for q in range(self.Q)]
        #         for t in range(self.T)
        #     ]
        # )
        # beta_den = np.einsum("tq,tr->tqr", beta_den, beta_den)
        # or
        # beta_num = np.einsum(
        #     "ijtqr,ijt->qrt", self.twopoint_edge_marg, (self.A > 0)
        # )
        # beta_den = np.einsum("itq,it->qt", self.node_marg, self.degs)
        # beta_den = np.einsum("qt,rt->qrt", beta_den, beta_den)
        self.jit_model.update_beta(init)

    def update_meta_params(self, init=False):
        self.jit_model.update_meta_params(init)

    def update_poisson_meta(self, s, init=False):
        # xi = np.array(
        #     [
        #         [(self.Z[:, t] == q).sum() for t in range(self.T)]
        #         for q in range(self.Q)
        #     ]
        # )
        # zeta = np.array(
        #     [
        #         [self.X[s][self.Z[:, t] == q, t, 0].sum() for t in range(self.T)]
        #         for q in range(self.Q)
        #     ]
        # )

        # zeta = np.einsum("itq,itd->qt", self.node_marg, self.X[s])
        self.jit_model.update_poisson_meta(s, init)

    def update_indep_bern_meta(self, s, init=False):
        # xi = np.array(
        #     [
        #         [(self.Z[:, t] == q).sum() for t in range(self.T)]
        #         for q in range(self.Q)
        #     ]
        # )
        # rho = np.array(
        #     [
        #         [
        #             self.X[s][self.Z[:, t] == q, t, :].sum(axis=0)
        #             for t in range(self.T)
        #         ]
        #         for q in range(self.Q)
        #     ]
        # )
        # rho = np.einsum("itq,itl->qtl", self.node_marg, self.X[s])
        self.jit_model.update_indep_bern_meta(s, init)

    def zero_diff(self):
        self.jit_model.zero_diff()

    def set_Z_by_MAP(self):
        self.jit_model.set_Z_by_MAP()
        self.Z = self.jit_model.Z

