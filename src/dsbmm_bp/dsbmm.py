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
from utils import numba_ix, nb_poisson_lkl, nb_ib_lkl

# from sklearn.cluster import MiniBatchKMeans

# base_spec = [
#     ("A", float32[:]),  # an array field
#     ("X", float32[:]),  # an array field
#     ("Z", int32[:]),  # an integer array field
# ]
X_ex = List.empty_list(float64[:, :, :])
# X_ex.append(np.empty((1, 1, 1), dtype=np.float64))
X_type = typeof(X_ex)

meta_types_ex = List.empty_list(unicode_type)
meta_types_type = typeof(meta_types_ex)
meta_params_ex = List.empty_list(typeof(np.empty((1, 1, 1), dtype=np.float64)))
# meta_params_ex.append(np.empty((1, 1, 1), dtype=np.float64))
meta_params_type = typeof(meta_params_ex)
# this decorator ensures types of each base field, and means all methods are compiled into nopython fns
# further types are inferred from type annotations
@jitclass  # (base_spec)
class DSBMMBase:
    # A: np.ndarray  # assume N x N x T array s.t. (i,j,t)th position confers information about connection from i to j at time t
    A: float64[:, :, :]
    X: X_type
    X_poisson: float64[:, :, :]
    X_ib: float64[:, :, :]
    # X: list
    # [
    #     np.ndarray
    # ]  # assume S x N x T x Ds array s.t. (s)(i,t,ds) posn is info about ds dim of sth type metadata of i at time t
    Z: int32[:, :]
    T: int
    N: int
    E: int64[:]  # num_edges in each timestep
    Q: int
    # meta_types: list[str]
    meta_types: meta_types_type
    meta_dims: int32[:]
    deg_corr: bool
    degs: float64[:, :, :]  # N x T x [in,out]
    kappa: float64[:, :, :]  # Q x T x [in,out]
    edgemat: float64[:, :, :]
    deg_entropy: float
    _alpha: float64[:]  # init group probs
    _pi: Array(float64, ndim=2, layout="C")  # group transition mat
    _lam: float64[:, :, :]  # block pois params in DC case
    _beta: Array(float64, ndim=3, layout="C")  # block edge probs in binary NDC case
    # _meta_params: list[np.ndarray]  # params for metadata dists
    _meta_params: meta_params_type
    node_marg: float64[:, :, :]
    twopoint_time_marg: float64[:, :, :, :]  # assume N x T x Q x Q (i,t,q,qprime)
    twopoint_edge_marg: float64[:, :, :, :, :]  # assume N x N x T x Q x Q (i,j,t,q,r)
    meta_lkl: float64[:, :, :]  # N x T x Q array of meta lkl term for i at t in q

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
            tmp2.append(np.zeros((self.N, self.T, self.meta_dims[0])))
            tmp2.append(np.zeros((self.N, self.T, self.meta_dims[1])))
            self._meta_params = tmp2
        # else: # TODO: fix for loading general X
        #     self.X = X
        self.Z = Z
        assert self.A is not None
        assert self.X is not None
        self.N = self.A.shape[0]
        self.E = np.array(list(map(np.count_nonzero, self.A.transpose(2, 0, 1))))
        self.T = self.A.shape[-1]
        self.Q = (
            Q if Q is not None else len(np.unique(self.Z))
        )  # will return 0 if both Q and Z None
        self.deg_corr = deg_corr
        self.degs = self.compute_degs(self.A)
        self.kappa = self.compute_group_degs()
        self.edgemat = self.compute_block_edgemat()

        self.deg_entropy = -(self.degs * np.log(self.degs)).sum()

        self.meta_types = meta_types

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

        # self.update_parameters()

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
        self.meta_lkl = np.zeros((self.N, self.T, self.Q))
        for s, mt in enumerate(self.meta_types):
            # print(f"Updating params for {mt} dist")
            if mt == "poisson":
                # print("In Poisson")
                pois_params = self._meta_params[s]  # shape (Q x T x 1)
                # TODO: Finish this
                print("\tUpdated Poisson lkl contribution")
            elif mt == "indep bernoulli":
                # print("In IB")
                ib_params = self._meta_params[s]  # shape (Q x T x L)
                # TODO: Finish this
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
        else:
            # print("Updating alpha")
            self._alpha = (
                self.node_marg[:, 0, :].sum(axis=0) / self.N
            )  # NB mean w axis argument not supported by numba

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
            qqprime_trans = qqprime_trans / qqprime_trans.sum(axis=-1)  # normalise rows
            self._pi = qqprime_trans
        else:
            self._pi = self.twopoint_time_marg.sum(axis=0).sum(axis=0) / self.node_marg[
                :, :-1, :
            ].sum(
                axis=0
            )  # need to do first sum twice as numba axis argument only takes integers (rather than axis=(0,1) as we would want)
            self.correct_pi()

    def correct_pi(self):
        self._pi = self._pi / self._pi.sum(axis=1)

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
            self._lam = lam_num / lam_den
        else:
            # lam_num = np.einsum("ijtqr,ijt->qrt", self.twopoint_edge_marg, self.A)
            for q in range(self.Q):
                for r in range(self.Q):
                    for t in range(self.T):
                        for i in range(self.N):
                            for j in range(self.N):
                                lam_num[q, r, t] += (
                                    self.twopoint_edge_marg[i, j, t, q, r]
                                    * self.A[i, j, t]
                                )
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
            self._lam = lam_num / lam_den

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

            self._beta = beta_num / beta_den
        else:
            # beta_num = np.einsum(
            #     "ijtqr,ijt->qrt", self.twopoint_edge_marg, (self.A > 0)
            # )
            for q in range(self.Q):
                for r in range(self.Q):
                    for t in range(self.T):
                        for i in range(self.N):
                            for j in range(self.N):
                                beta_num[q, r, t] += self.twopoint_edge_marg[
                                    i, j, t, q, r
                                ] * (self.A[i, j, t] > 0)
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
            self._beta = beta_num / beta_den

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
            # zeta = np.array(
            #     [
            #         [self.X[s][self.Z[:, t] == q, t, 0].sum() for t in range(self.T)]
            #         for q in range(self.Q)
            #     ]
            # )
            # gdb()
            self._meta_params[s] = zeta / xi
        else:
            xi[:, :, 0] = self.node_marg.sum(axis=0).transpose(1, 0)
            # zeta = np.einsum("itq,itd->qt", self.node_marg, self.X[s])
            for q in range(self.Q):
                for t in range(self.T):
                    zeta[q, t, 0] = (self.node_marg[:, t, q] * self.X[s][:, t, 0]).sum()
            self._meta_params[s] = zeta / xi

    def update_indep_bern_meta(self, s, init):
        # TODO: handle correct normalisation
        xi = np.zeros((self.Q, self.T, 1))
        rho = np.zeros((self.Q, self.T, self.X[s].shape[-1]))
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
            # rho = np.array(
            #     [
            #         [
            #             self.X[s][self.Z[:, t] == q, t, :].sum(axis=0)
            #             for t in range(self.T)
            #         ]
            #         for q in range(self.Q)
            #     ]
            # )
            self._meta_params[s] = rho / xi
        else:
            xi[:, :, 0] = self.node_marg.sum(axis=0).transpose(1, 0)
            # rho = np.einsum("itq,itl->qtl", self.node_marg, self.X[s])
            for q in range(self.Q):
                for t in range(self.T):
                    rho[q, t, :] = (self.node_marg[:, t, q] * self.X[s][:, t, :]).sum(
                        axis=0
                    )
            self._meta_params[s] = rho / xi


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

