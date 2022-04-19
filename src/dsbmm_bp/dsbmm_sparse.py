# TODO: implement DSBMM again but using sparse formulation so can fully take advantage of scaling

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

from scipy import sparse
import csr

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

sparse_A_ex = List()
sparse_A_ex.append(csr.create_empty(1, 1))
sparse_A_type = typeof(sparse_A_ex)
# this decorator ensures types of each base field, and means all methods are compiled into nopython fns
# further types are inferred from type annotations
@jitclass  # (base_spec)
class DSBMMSparseBase:
    # A: np.ndarray  # assume N x N x T array s.t. (i,j,t)th position confers information about connection from i to j at time t
    A: sparse_A_type
    X: X_type
    X_poisson: float64[:, :, ::1]
    X_ib: float64[:, :, ::1]
    # X: list
    # [
    #     np.ndarray
    # ]  # assume S x N x T x Ds array s.t. (s)(i,t,ds) posn is info about ds dim of sth type metadata of i at time t
    Z: int32[:, ::1]
    _n_qt: int64[:, ::1]  # Q x T array holding num nodes in group q at time t
    T: int
    N: int
    E: int64[::1]  # num_edges in each timestep
    Q: int
    # meta_types: list[str]
    meta_types: meta_types_type
    meta_dims: int32[::1]
    deg_corr: bool
    directed: bool
    use_meta: bool
    degs: float64[:, :, ::1]  # N x T x [in,out]
    kappa: float64[:, :, ::1]  # Q x T x [in,out]
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
    _edge_vals: float64[:, ::1]
    diff: float64
    verbose: bool

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
        directed,
        use_meta,
        meta_types,
        verbose,
    ):
        # if data is not None:
        #     self.A = data["A"]
        #     self.X = data["X"]
        #     self.Z = data.get("Z", None)
        # else:
        self.A = A  # assuming list of T sparse adjacency matrices (N x N) (made in right format already by Python wrapper)
        self.N = A[0].nrows
        self.E = np.array([A_t.nnz for A_t in self.A])
        self.T = len(A)
        # TODO: consider making this unique if undirected (i.e. so
        # don't loop over both i and j each time)
        # Make edge_vals E x 4 array w rows [i,j,t,val] where A[i,j,t]==val != 0
        # (i.e. just COO format - easier to translate prev loops over)
        self._edge_vals = np.zeros((np.array([A_t.nnz for A_t in self.A]).sum(), 4))
        pos = 0
        for t, A_t in enumerate(self.A):
            row_idx = A_t.transpose().colinds.astype(np.float64)
            col_idx = A_t.colinds.astype(np.float64)
            t_vals = np.ones(A_t.nnz) * t
            vals = A_t.values
            tmp = np.vstack((row_idx, col_idx, t_vals, vals)).T
            self._edge_vals[pos : pos + A_t.nnz] = tmp
            pos += A_t.nnz

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
        self.directed = directed
        self.use_meta = use_meta
        self._n_qt = self.compute_group_counts()
        self.degs = self.compute_degs(self.A)
        self.kappa = self.compute_group_degs()
        self.deg_entropy = -(self.degs * np.log(self.degs)).sum()

        self.meta_types = meta_types
        tmp = List()
        for t in range(self.T):
            tmp2 = List()
            for i in range(self.N):
                tmp2.append(self.A[t].row_cs(i).astype(np.int64))
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

        self.diff = 1.0
        self.verbose = verbose

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

    def compute_group_counts(self):
        n_qt = np.zeros((self.Q, self.T), dtype=np.int64)
        for q in range(self.Q):
            for t in range(self.T):
                n_qt[q, t] = (self.Z[:, t] == q).sum()
        return n_qt

    def compute_degs(self, A):
        """Compute in-out degree matrix from given temporal adjacency mat

        Args:
            A (_type_): _description_

        Returns:
            _type_: _description_
        """
        in_degs = np.zeros((self.N, self.T))
        out_degs = np.zeros((self.N, self.T))
        A_T = [A_t.transpose() for A_t in A]
        for t in range(self.T):
            for i in range(self.N):
                in_degs[i, t] = A_T[t].row_vs(i).sum()
                out_degs[i, t] = A[t].row_vs(i).sum()

        return np.dstack((in_degs, out_degs))

    def compute_group_degs(self):
        """Compute group in- and out-degrees for current node memberships
        """
        kappa = np.zeros((self.Q, self.T, 2))
        for q in range(self.Q):
            for t in range(self.T):
                kappa[q, t, :] = self.degs[self.Z[:, t] == q, t, :].sum(axis=0)
        return kappa

    def compute_log_likelihood(self):
        """Compute log likelihood of model for given memberships 

            In DC case this corresponds to usual DSBMM with exception of each timelice now has log lkl
                \sum_{q,r=1}^Q m_{qr} \log\frac{m_{qr}}{\kappa_q^{out}\kappa_r^{in}},
            (ignoring constants w.r.t. node memberships) 
            
        Returns:
            _type_: _description_
        """
        pass

    def update_params(self, init, learning_rate):
        """Given marginals, update parameters suitably

        Args:
            messages (_type_): _description_
        """
        # first init of parameters given initial groups if init=True, else use provided marginals
        # TODO: remove prints after fix
        self.update_alpha(init, learning_rate)
        if self.verbose:
            print(self._alpha)
        self.update_pi(init, learning_rate)
        if self.verbose:
            print(self._pi)
        if self.deg_corr:
            self.update_lambda(init, learning_rate)
            if self.verbose:
                print(self._lam)
        else:
            # NB only implemented for binary case
            self.update_beta(init, learning_rate)
            if self.verbose:
                print(self._beta)
        self.update_meta_params(init, learning_rate)
        if self.verbose:
            print(self._meta_params)
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
                if self.verbose:
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
                if self.verbose:
                    print("\tUpdated IB lkl contribution")
            else:
                raise NotImplementedError(
                    "Yet to implement metadata distribution of given type \nOptions are 'poisson' or 'indep bernoulli'"
                )  # NB can't use string formatting for print in numba
        for i in range(self.N):
            for t in range(self.T):
                for q in range(self.Q):
                    if self.meta_lkl[i, t, q] < TOL:
                        self.meta_lkl[i, t, q] = TOL
                    elif self.meta_lkl[i, t, q] > 1 - TOL:
                        self.meta_lkl[i, t, q] = 1 - TOL

    def set_node_marg(self, values):
        self.node_marg = values

    def set_twopoint_time_marg(self, values):
        self.twopoint_time_marg = values

    def set_twopoint_edge_marg(self, values):
        self.twopoint_edge_marg = values

    def calc_free_energy(self):
        f_site = 0.0
        for i in range(self.N):
            for t in range(self.T):
                nbrs = self.nbrs[t][i]
                for q in range(self.Q):
                    a = 0.0
                    for j_idx, j in enumerate(nbrs):
                        b = 0.0
                        for r in range(self.Q):
                            b += (
                                self._beta[r, q, t] * self._psi_e[t][i][j_idx, r]
                            )  # * deg_i*deg_j if DC
                        a += np.log(b)
                        # ... basically f_site = 1/N \sum_i log(Z_i) for Z_i the norm factors of marginals
                        # similarly then calc f_link = 1/2N \sum_ij log(Z_ij) for the twopoint marg norms (for us these will included time margs)
                        # then final term is something like \sum_qr p_qr * alpha_q * alpha_r  (CHECK)
                        # Total free energy is then -f_site + f_link - final term, and smaller better (?)

    def update_alpha(self, init, learning_rate):
        if init:
            # case of no marginals / true partition provided to calculate most likely params
            self._alpha = np.array([(self.Z == q).mean() for q in range(self.Q)])
            self._alpha /= self._alpha.sum()
            self._alpha[self._alpha < TOL] = TOL
            self._alpha /= self._alpha.sum()
            # self._alpha[self._alpha > 1 - TOL] = 1 - TOL
        else:
            # if DC, seems like should multiply marg by degree prior to sum - unseen for directed case but can calculate
            tmp = np.zeros(self.Q)
            # print("Updating alpha")
            #
            for q in range(self.Q):
                tmp[q] = self.node_marg[
                    :, :, q
                ].mean()  # NB mean w axis argument not supported by numba beyond single integer

            tmp /= tmp.sum()
            tmp[tmp < TOL] = TOL
            tmp /= tmp.sum()
            # tmp[tmp > 1 - TOL] = 1 - TOL

            tmp = learning_rate * tmp + (1 - learning_rate) * self._alpha
            tmp_diff = np.abs(tmp - self._alpha).mean()
            if np.isnan(tmp_diff):
                raise RuntimeError("Problem updating alpha")
            if self.verbose:
                print("Alpha diff:", np.round_(tmp_diff, 3))
            self.diff += tmp_diff
            self._alpha = tmp

    def update_pi(self, init, learning_rate):
        qqprime_trans = np.zeros((self.Q, self.Q))
        if init:
            for q in range(self.Q):
                for qprime in range(self.Q):
                    for t in range(1, self.T):
                        tm1_idxs = self.Z[:, t - 1] == q
                        t_idxs = self.Z[:, t] == qprime
                        qqprime_trans[q, qprime] += (tm1_idxs & t_idxs).sum()
            qqprime_trans /= np.expand_dims(qqprime_trans.sum(axis=1), 1)
            # TODO: provide different cases for non-uniform init clustering
            # NB for uniform init clustering this provides too homogeneous pi, and is more important now
            p = 0.8
            qqprime_trans = p * qqprime_trans + (1 - p) * np.random.rand(
                *qqprime_trans.shape
            )

        else:
            qqprime_trans = self.twopoint_time_marg.sum(axis=0).sum(axis=0)
            # qqprime_trans /= np.expand_dims(
            #     self.node_marg[:, :-1, :].sum(axis=0).sum(axis=0), 1
            # )  # need to do sums twice as numba axis argument
            # # only takes integers (rather than axis=(0,1) as
            # # we would want) - can't use this as node_marg sums could
            # # be tiny / zero
            # below is unnecessary as can enforce normalisation directly
            # - just introduces instability
            # tot_marg = self.node_marg[:, :-1, :].sum(axis=0).sum(axis=0)
            # print("tot marg sums:", tot_marg)
            # for q in range(self.Q):
            #     if tot_marg[q] > TOL:
            #         qqprime_trans[q, :] = qqprime_trans[q, :] / tot_marg[q]
            #     else:
            #         raise RuntimeError("Problem with node marginals")
            # qqprime_trans[q, :] = TOL
            # self.correct_pi()
        qqprime_trans = qqprime_trans / np.expand_dims(
            qqprime_trans.sum(axis=1), 1
        )  # normalise rows
        for q in range(self.Q):
            for qprime in range(self.Q):
                if qqprime_trans[q, qprime] < TOL:
                    qqprime_trans[q, qprime] = TOL
                if qqprime_trans[q, qprime] > 1 - TOL:
                    qqprime_trans[q, qprime] = 1 - TOL
        if not init:
            tmp = learning_rate * qqprime_trans + (1 - learning_rate) * self._pi
            tmp_diff = np.abs(tmp - self._pi).mean()
            if np.isnan(tmp_diff):
                raise RuntimeError("Problem updating pi")
            if self.verbose:
                print("Pi diff:", np.round_(tmp_diff, 3))
            self.diff += tmp_diff
            self._pi = tmp
        else:
            self._pi = qqprime_trans

    def correct_pi(self):
        # print("correcting")
        for q in range(self.Q):
            for qprime in range(self.Q):
                if self._pi[q, qprime] < TOL:
                    self._pi[q, qprime] = TOL
                if self._pi[q, qprime] > 1 - TOL:
                    self._pi[q, qprime] = 1 - TOL
        self._pi = self._pi / self._pi.sum(axis=-1)

    def update_lambda(self, init, learning_rate):
        # TODO: fix on basis of beta below
        lam_num = np.zeros((self.Q, self.Q, self.T))
        lam_den = np.zeros((self.Q, self.Q, self.T))
        if init:
            for i, j, t, val in self._edge_vals:
                i, j, t = int(i), int(j), int(t)
                lam_num[self.Z[i, t], self.Z[j, t], t] += val

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
                for t in range(self.T):
                    # TODO: check if this is correct, as makes vanishing
                    lam_den[q, q, t] = (
                        self.kappa[q, t, 1] * self.kappa[q, t, 0]
                    )  # TODO: check right for r==q
                    if lam_num[q, q, t] < TOL:
                        lam_num[q, q, t] = TOL
                    if lam_den[q, q, t] < TOL:
                        lam_den[q, q, t] = 1.0
                    for r in range(q + 1, self.Q):
                        lam_den[q, r, t] = (
                            self.kappa[q, t, 1] * self.kappa[r, t, 0]
                        )  # TODO: check right in directed case
                        if lam_num[q, r, t] < TOL:
                            lam_num[q, r, t] = TOL
                        if lam_den[q, r, t] < TOL:
                            lam_den[q, r, t] = 1.0
                        if not self.directed:
                            lam_num[r, q, t] = lam_num[q, r, t]
                            lam_den[r, q, t] = lam_den[q, r, t]
                    if self.directed:
                        for r in range(q):
                            lam_den[q, r, t] = (
                                self.kappa[q, t, 1] * self.kappa[r, t, 0]
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
                for r in range(q, self.Q):
                    for i, j, t, val in self._edge_vals:
                        i, j, t = int(i), int(j), int(t)
                        j_idx = self.nbrs[t][i] == j
                        # if r==q: # TODO: special treatment?
                        lam_num[q, r, t] += (
                            self.twopoint_edge_marg[t][i][j_idx, q, r] * val
                        )
                    for t in range(self.T):
                        if lam_num[q, r, t] < TOL:
                            lam_num[q, r, t] = TOL
                        if not self.directed:
                            lam_num[r, q, t] = lam_num[q, r, t]
                if self.directed:
                    for r in range(q):
                        for i, j, t, val in self._edge_vals:
                            i, j, t = int(i), int(j), int(t)
                            j_idx = self.nbrs[t][i] == j
                            # if r==q: # TODO: special treatment?
                            lam_num[q, r, t] += (
                                self.twopoint_edge_marg[t][i][j_idx, q, r] * val
                            )

            # lam_den = np.einsum("itq,it->qt", self.node_marg, self.degs)
            # lam_den = np.einsum("qt,rt->qrt", lam_den, lam_den)
            marg_kappa_out = np.zeros((self.Q, self.T))
            marg_kappa_in = np.zeros((self.Q, self.T))
            for q in range(self.Q):
                for t in range(self.T):
                    marg_kappa_out[q, t] = (
                        self.node_marg[:, t, q] * self.degs[:, t, 1]
                    ).sum(
                        axis=0
                    )  # TODO: again check this uses right deg if directed
                    marg_kappa_in[q, t] = (
                        self.node_marg[:, t, q] * self.degs[:, t, 0]
                    ).sum(axis=0)
            for q in range(self.Q):
                for t in range(self.T):
                    for r in range(q, self.Q):
                        lam_den[q, r, t] = marg_kappa_out[q, t] * marg_kappa_in[r, t]
                        if lam_den[q, r, t] < TOL:
                            lam_den[q, r, t] = 1.0
                        if not self.directed:
                            lam_den[r, q, t] = lam_den[q, r, t]
                    if self.directed:
                        for r in range(q):
                            lam_den[q, r, t] = (
                                marg_kappa_out[q, t] * marg_kappa_in[r, t]
                            )
                            if lam_den[q, r, t] < TOL:
                                lam_den[q, r, t] = 1.0
        # NB use relative rather than absolute difference here as lam could be large
        tmp = lam_num / lam_den
        if not init:
            tmp = learning_rate * tmp + (1 - learning_rate) * self._lam
            tmp_diff = np.abs((tmp - self._lam) / self._lam).mean()
            if np.isnan(tmp_diff):
                raise RuntimeError("Problem updating lambda")
            if self.verbose:
                print("Lambda diff:", np.round_(tmp_diff, 3))
            self.diff += tmp_diff
            self._lam = tmp
        else:
            self._lam = tmp

    def update_beta(self, init, learning_rate):
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
            for i, j, t, _ in self._edge_vals:
                i, j, t = int(i), int(j), int(t)
                beta_num[self.Z[i, t], self.Z[j, t], t] += 1
            for q in range(self.Q):
                # enforce uniformity for identifiability
                tmp = 0.0
                for t in range(self.T):
                    tmp += beta_num[q, q, t]
                for t in range(self.T):
                    beta_num[q, q, t] = tmp
            # beta_den = np.array(
            #     [
            #         [self.degs[self.Z[:, t] == q].sum() for q in range(self.Q)]
            #         for t in range(self.T)
            #     ]
            # )
            # beta_den = np.einsum("tq,tr->tqr", beta_den, beta_den)
            # print("beta_num:", beta_num.transpose(2, 0, 1))
            # print("kappa:", self.kappa)
            for q in range(self.Q):
                for t in range(self.T):
                    for r in range(q, self.Q):
                        beta_den[q, r, t] = (
                            self._n_qt[q, t] * self._n_qt[r, t]
                        )  # TODO: check right in directed case, and if r==q
                        # if beta_num[q, r, t] < TOL:
                        #     beta_num[q, r, t] = TOL
                        if beta_den[q, r, t] < TOL:
                            beta_den[
                                q, r, t
                            ] = 1.0  # this is same as how prev people have handled (effectively just don't do division if will cause problems, as num will
                            # be v. small anyway)
                        if not self.directed and r != q:
                            beta_den[r, q, t] = beta_den[q, r, t]
                            beta_num[
                                q, r, t
                            ] /= 2.0  # done elsewhere, TODO: check basis
                            beta_num[r, q, t] = beta_num[q, r, t]
                    if self.directed:
                        for r in range(q):
                            beta_den[q, r, t] = self._n_qt[q, t] * self._n_qt[r, t]
                            if beta_den[q, r, t] < TOL:
                                beta_den[q, r, t] = 1.0
            for q in range(self.Q):
                # enforce uniformity for identifiability
                tmp = 0.0
                for t in range(self.T):
                    tmp += beta_den[q, q, t]
                for t in range(self.T):
                    beta_den[q, q, t] = tmp
            # print("beta_den:", beta_den)

        else:
            # beta_num = np.einsum(
            #     "ijtqr,ijt->qrt", self.twopoint_edge_marg, (self.A > 0)
            # )
            for q in range(self.Q):
                for r in range(q, self.Q):
                    if r != q:
                        for i, j, t, a_ijt in self._edge_vals:
                            i, j, t = int(i), int(j), int(t)
                            j_idx = np.nonzero(self.nbrs[t][i] == j)[0]
                            # print(self.twopoint_edge_marg[t][i][j_idx, q, r])
                            # assert j_idx.sum() == 1
                            val = self.twopoint_edge_marg[t][i][j_idx, q, r][0]
                            try:
                                assert not np.isnan(val)
                            except:
                                print("(i,j,t):", i, j, t)
                                print("A[i,j,t] = ", a_ijt)
                                print("twopoint marg: ", val)
                                raise RuntimeError("Problem updating beta")
                            beta_num[q, r, t] += val
                        if not self.directed:
                            for t in range(self.T):
                                beta_num[q, r, t] /= 2.0
                                beta_num[r, q, t] = beta_num[q, r, t]
                    else:
                        # enforce uniformity across t for identifiability
                        for i, j, t, a_ijt in self._edge_vals:
                            i, j, t = int(i), int(j), int(t)
                            j_idx = np.nonzero(self.nbrs[t][i] == j)[0]
                            # print(self.twopoint_edge_marg[t][i][j_idx, q, r])
                            # assert j_idx.sum() == 1
                            val = self.twopoint_edge_marg[t][i][j_idx, q, r][0]
                            try:
                                assert not np.isnan(val)
                            except:
                                print("(i,j,t):", i, j, t)
                                print("A[i,j,t] = ", a_ijt)
                                print("twopoint marg: ", val)
                                raise RuntimeError("Problem updating beta")
                            for tprime in range(self.T):
                                beta_num[q, r, tprime] += (
                                    val / 2.0
                                )  # TODO: check if this should also be here

                if self.directed:
                    for r in range(q):
                        for i, j, t, _ in self._edge_vals:
                            i, j, t = int(i), int(j), int(t)
                            j_idx = np.nonzero(self.nbrs[t][i] == j)[0]
                            val = self.twopoint_edge_marg[t][i][j_idx, q, r][0]
                            beta_num[q, r, t] += val

            # beta_den = np.einsum("itq,it->qt", self.node_marg, self.degs)
            # beta_den = np.einsum("qt,rt->qrt", beta_den, beta_den)
            group_marg = np.zeros((self.Q, self.T))
            for q in range(self.Q):
                for t in range(self.T):
                    group_marg[q, t] = self.node_marg[
                        :, t, q
                    ].sum()  # TODO: again check this is right if directed
            for q in range(self.Q):
                for t in range(self.T):
                    for r in range(q, self.Q):
                        if r != q:
                            beta_den[q, r, t] = group_marg[q, t] * group_marg[r, t]

                            if beta_den[q, r, t] < TOL:
                                beta_den[q, r, t] = 1.0
                            # if not self.directed: same either way (should this be?)
                            beta_den[r, q, t] = beta_den[q, r, t]
                        else:
                            for tprime in range(self.T):
                                # again enforce uniformity for identifiability
                                beta_den[q, r, t] += (
                                    group_marg[q, tprime] * group_marg[r, tprime]
                                )

        # TODO: fix for case where beta_den very small (just consider using logs)
        # correct for numerical stability
        tmp = beta_num / beta_den
        for q in range(self.Q):
            for r in range(self.Q):
                for t in range(self.T):
                    if tmp[q, r, t] < TOL:
                        tmp[q, r, t] = TOL
                    elif tmp[q, r, t] > 1 - TOL:
                        tmp[q, r, t] = 1 - TOL
        if not init:
            tmp = learning_rate * tmp + (1 - learning_rate) * self._beta
            tmp_diff = np.abs(tmp - self._beta).mean()
            if np.isnan(tmp_diff):
                raise RuntimeError("Problem updating beta")
            if self.verbose:
                print("Beta diff:", np.round_(tmp_diff, 3))
            self.diff += tmp_diff
            self._beta = tmp
        else:
            self._beta = tmp

    def update_meta_params(self, init, learning_rate):
        for s, mt in enumerate(self.meta_types):
            # print(f"Updating params for {mt} dist")
            if mt == "poisson":
                # print("In Poisson")
                self.update_poisson_meta(s, init, learning_rate)
                if self.verbose:
                    print("\tUpdated Poisson")
            elif mt == "indep bernoulli":
                # print("In IB")
                self.update_indep_bern_meta(s, init, learning_rate)
                if self.verbose:
                    print("\tUpdated IB")
            else:
                raise NotImplementedError(
                    "Yet to implement metadata distribution of given type \nOptions are 'poisson' or 'indep bernoulli'"
                )  # NB can't use string formatting for print in numba

    def update_poisson_meta(self, s, init, learning_rate):
        xi = np.ones((self.Q, self.T, 1))
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
        # TODO: fix for case where xi small - rather than just setting as 1 when less than
        # 1e-50 (just consider using logs)
        tmp = zeta / xi
        if not init:
            tmp = learning_rate * tmp + (1 - learning_rate) * self._meta_params[s]
            tmp_diff = np.abs(
                (tmp - self._meta_params[s]) / self._meta_params[s]
            ).mean()
            if np.isnan(tmp_diff):
                raise RuntimeError("Problem updating poisson params")
            if self.verbose:
                print("Poisson diff: ", np.round_(tmp_diff, 3))
            self.diff += tmp_diff
            self._meta_params[s] = tmp
        else:
            self._meta_params[s] = tmp

    def update_indep_bern_meta(self, s, init, learning_rate):
        xi = np.ones((self.Q, self.T, 1))
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
                    # for l in range(L):
                    #     if rho[q, t, l] < TOL:
                    #         rho[q, t, l] = TOL
        # TODO: fix for xi very small (just use logs)
        tmp = rho / xi
        for q in range(self.Q):
            for t in range(self.T):
                for l in range(L):
                    if tmp[q, t, l] < TOL:
                        tmp[q, t, l] = TOL
                    elif tmp[q, t, l] > 1 - TOL:
                        tmp[q, t, l] = 1 - TOL
        if not init:
            tmp = learning_rate * tmp + (1 - learning_rate) * self._meta_params[s]
            tmp_diff = np.abs(tmp - self._meta_params[s]).mean()
            if np.isnan(tmp_diff):
                raise RuntimeError("Problem updating IB params")
            if self.verbose:
                print("IB diff: ", np.round_(tmp_diff, 3))
            self.diff += tmp_diff
            self._meta_params[s] = tmp
        else:
            self._meta_params[s] = tmp

    def zero_diff(self):
        self.diff = 0.0

    def set_Z_by_MAP(self):
        for i in range(self.N):
            for t in range(self.T):
                self.Z[i, t] = np.argmax(self.node_marg[i, t, :])


class DSBMMSparse:
    """Pure Python wrapper around DSBMMSparseBase to allow optional/keyword arguments
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
        directed=False,
        use_meta=True,  # control use of metadata or not (for debug)
        meta_types=["poisson", "indep bernoulli"],
        verbose=False,
    ):
        # if data is not None:
        #     self.A = data["A"]
        #     self.X = data["X"]
        #     self.Z = data.get("Z", None)
        # else:
        # assume A passed as list of scipy sparse matrices (CSR format)
        tmp = List()
        for A_t in A:
            tmp.append(csr.CSR.from_scipy(A_t))
        A = tmp
        self.A = A
        self.jit_model = DSBMMSparseBase(
            A,
            X,
            X_poisson,
            X_ib,
            Z,
            Q,
            deg_corr,
            directed,
            use_meta,
            meta_types,
            verbose,
        )
        self.directed = directed
        self.verbose = verbose
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

    def compute_group_counts(self):
        return self.jit_model.compute_group_counts()

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

    def compute_log_likelihood(self):
        """Compute log likelihood of model for given memberships 

            In DC case this corresponds to usual DSBMM with exception of each timelice now has log lkl
                \sum_{q,r=1}^Q m_{qr} \log\frac{m_{qr}}{\kappa_q^{out}\kappa_r^{in}},
            (ignoring constants w.r.t. node memberships) 
            
        Returns:
            _type_: _description_
        """
        return self.jit_model.compute_log_likelihood()

    def update_params(self, init=False, learning_rate=0.2):
        """Given marginals, update parameters suitably

        Args:
            messages (_type_): _description_
        """
        # first init of parameters given initial groups if init=True, else use provided marginals
        # TODO: remove extra prints after fix
        self.jit_model.update_alpha(init, learning_rate)
        if self.verbose:
            print(self.jit_model._alpha)
            print("\tUpdated alpha")
        self.jit_model.update_pi(init, learning_rate)
        if self.verbose:
            print(self.jit_model._pi)
            print("\tUpdated pi")
        if self.jit_model.deg_corr:
            self.jit_model.update_lambda(init, learning_rate)
            if self.verbose:
                print("\tUpdated lambda")
        else:
            # NB only implemented for binary case
            self.jit_model.update_beta(init, learning_rate)
            if self.verbose:
                print(self.jit_model._beta.transpose(2, 0, 1))
                print("\tUpdated beta")
            if not self.directed:
                assert np.allclose(
                    self.jit_model._beta.transpose(1, 0, 2), self.jit_model._beta
                )
        self.jit_model.update_meta_params(init, learning_rate)
        if self.verbose:
            # print(self.jit_model._meta_params)
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

