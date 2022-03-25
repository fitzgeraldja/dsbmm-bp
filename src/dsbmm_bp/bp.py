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

# from numba_dsbmm_methods import *
# from numba_bp_methods import *
import numpy as np
from utils import numba_ix

from dsbmm import DSBMM, DSBMMBase

X_ex = List.empty_list(float64[:, :, :])
X_type = typeof(X_ex)
psi_e_ex = List.empty_list(ListType(Array(float64, ndim=2, layout="C")))
psi_e_type = typeof(psi_e_ex)
# psi_t_ex = List.empty_list(ListType(float64[:, :])) # unnecessary for same reasons as below
# psi_t_type = typeof(psi_t_ex)
nbrs_ex = List.empty_list(ListType(Array(int64, ndim=1, layout="C")))
nbrs_type = typeof(nbrs_ex)
twopoint_e_ex = List.empty_list(ListType(Array(float64, ndim=3, layout="C")))
twopoint_e_type = typeof(
    twopoint_e_ex
)  # TODO: replace as array type of size (Q^2 sum_t E_t), along w lookup table for idx of i,j,t as this should be sufficient
# given implementation below
# twopoint_t_ex = List.empty_list(ListType(float64[:,:])) # unnecessary for t marg, as memory size O(NTQ^2) at most O(Q^2\sum_t E_t) size of constrained formulation
# of e marg (for sparse networks w E_t approx N for each t), and arrays generally preferable structure if significant overall savings not possible


@jitclass
class BPBase:
    N: int
    Q: int
    T: int
    _beta: float  # temperature
    deg_corr: bool
    model: DSBMMBase.class_type.instance_type
    A: float64[:, :, :]
    # X: list[np.ndarray]
    X: X_type
    c_qrt: float64[:, :, :]  # N*p_qrt
    n_qt: float64[:, :]  # prior prob
    _psi_e: psi_e_type  # spatial messages
    _psi_t: Array(
        float64, ndim=5, layout="C"
    )  # temporal messages, assume (i,t,q,q',2) for t in 0 to T-2, where first loc of final dim is backward messages from t+1 to t
    # and second loc is forward messages from t to t+1
    _h: float64[:, :]  # external field for each group, with shape (Q,T)
    node_marg: Array(
        float64, ndim=3, layout="C"
    )  # marginal group probabilities for each node
    # twopoint_e_marg: float64[:, :, :, :, :]
    twopoint_e_marg: twopoint_e_type
    twopoint_t_marg: Array(float64, ndim=4, layout="C")
    # - NB this also referred to as nu, eta or
    # other terms elsewhere
    _pres_nodes: bool_[:, :]
    _pres_trans: bool_[:, :]
    _edge_locs: int64[:, :]
    _trans_locs: int64[:, :]
    nbrs: nbrs_type

    def __init__(self, dsbmm: DSBMMBase.class_type.instance_type):
        self.model = dsbmm
        self.T = self.model.T
        self.N = self.model.N
        self.Q = self.model.Q
        self.deg_corr = self.model.deg_corr
        self.A = self.model.A
        self.X = self.model.X
        # start with given membership and corresponding messages, iterate until reach fixed point
        # given messages at fixed point, can update parameters to their most likely values - these emerge naturally
        # by requiring that the fixed point provides stationary free energy w.r.t. parameters
        self.get_neighbours()
        self.node_marg = np.zeros((self.N, self.T, self.Q))
        self._zero_twopoint_e_marg()
        # print(self._edge_locs)
        pass

    @property
    def n_timesteps(self):
        return self.T

    @property
    def beta(self):
        # temperature for Boltzmann dist
        # looks like this enters eqns via external field, i.e. e^-beta*h_q/N rather than e^-h_q/N
        # and also then in derivatives, hence possible need to have fns calculating vals of derivs  - need to be careful
        return self._beta

    @beta.setter
    def set_beta(self, value):
        self._beta = value

    def fit(self, data):
        pass

    def get_neighbours(self):
        # initially assuming undirected
        # TODO: directed version
        self._pres_nodes = (
            self.A.sum(axis=1) > 0
        )  # returns N x T boolean array w i,t True if i present in net at time t
        self._pres_trans = (
            self._pres_nodes[:, :-1] * self._pres_nodes[:, 1:]
        )  # returns N x T-1 array w i,t true if i
        # present at both t and t+1 (so can measure trans)
        self._edge_locs = np.array(
            list(zip(*self.A.nonzero()))
        )  # E x 3 array w rows [i,j,t] where A[i,j,t]!=0
        self._trans_locs = np.array(
            list(zip(*self._pres_trans.nonzero()))
        )  # N_trans x 2
        tmp = List()
        for t in range(self.T):
            tmp2 = List()
            for i in range(self.N):
                tmp2.append(self.A[i, :, t].nonzero()[0])
            tmp.append(tmp2)
        # tmp = [
        #     [self.A[i, :, t].nonzero()[0] for i in range(self.N)] for t in range(self.T)
        # ]  # so self.nbrs[t][i] gives
        self.nbrs = tmp
        # nbrs of i at t (possibly none)

        # want to have spatial and temporal messages, each of which relies on the other as well
        # so need to be able to (i) recall temporal messages (forward and backward) for a given node at a given time
        # (ii) be able to recall spatial messages for a given node at a given time
        # For each of these respectively, important things are (i) is node present at previous/next timestep
        # (ii) which nodes are neighbours at that time (and so whose messages you should collect)
        # suggests structuring separately as spatial/temporal messages, with spatial messages a (ragged) T length list of
        # N^t x d^t_i lists, where easy to know timestep, but then need idx of i, and which j these correspond to
        # (as will need messages sent to j to update i connected to j)

        # as need idxs, alt would be init memory for all spatial messages (\sum_t E^t x Q)
        # and all temporal messages (\sum_t=2 pres_trans x Q x 2 (as forwards and backwards))
        # then structure such that just loop over all in random order - nonzeros should give these
        # Key is that when consider a known message from i to j, can recover idxs for

    def init_messages(self, mode):
        if mode == "random":
            # initialise by random messages
            # self._psi_e = [
            #     [
            #         np.random.rand(len(self.nbrs[t][i]), self.Q)
            #         if len(self.nbrs[t][i]) > 0
            #         else None
            #         for i in range(self.N)
            #     ]
            #     for t in range(self.T)
            # ]
            # self._psi_e = [
            #     [
            #         msg / msg.sum(axis=1) if msg is not None else None
            #         for msg in self._psi_e[t]
            #     ]
            #     for t in range(self.T)
            # ]
            tmp = List()
            for t in range(self.T):
                tmp2 = List()
                for i in range(self.N):
                    n_nbrs = len(self.nbrs[t][i])
                    if n_nbrs > 0:
                        msg = np.random.rand(n_nbrs, self.Q)
                        # msg /= msg.sum(axis=1)[:, np.newaxis]
                        msg /= np.expand_dims(msg.sum(axis=1), 1)
                    else:
                        msg = np.empty((1, self.Q), dtype=np.float64)
                    tmp2.append(msg)
                # print("Trying to update psi_e")
                tmp.append(tmp2)
            self._psi_e = tmp
            # print("psi_e updated")
            # self._psi_t = [
            #     [np.random.rand(2, self.Q) for i in range(self.N)]
            #     for t in range(self.T - 1)
            # ]
            # self._psi_t = [
            #     [msg / msg.sum(axis=1) for msg in self._psi_t[t]]
            #     for t in range(self.T - 1)
            # ]
            self._psi_t = np.random.rand(self.N, self.T - 1, self.Q, self.Q, 2)
            # self._psi_t /= self._psi_t.sum(axis=3)[:, :, :, np.newaxis, :]
            self._psi_t /= np.expand_dims(self._psi_t.sum(axis=3), 3)
        elif mode == "partial":
            # initialise by partly planted partition - others left random
            pass
        elif mode == "planted":
            # initialise by given partition
            pass

        pass

    def update_messages(self,):
        # assume want to visit messages in (almost) random order - here only random over edge locations, then use
        # i vals as choice of loc to update temporal messages
        np.random.shuffle(self._edge_locs)
        for i, j, t in self._edge_locs:
            self.update_spatial_message(i, j, t)
        print("\tUpdated spatial messages")
        np.random.shuffle(self._trans_locs)
        for i, t in self._trans_locs:
            self.update_temporal_messages(i, t)
        print("\tUpdated temporal messages")

    def update_spatial_message(self, i, j, t):
        j_idx = self.nbrs[t][i] == j
        cavity_term = self.cavity_spatial_message(i, j, t)

        marg = self.meta_prob(i, t) * np.exp(self._h[:, t]) * cavity_term
        try:
            if self._pres_trans[i, t]:
                # i pres at t and t+1, can include backward temp msg term
                marg *= self.backward_temp_msg_term(i, t)
        except:
            # t must be T-1, so no t+1, that's fine - doing this way means don't have to check t val for every edge
            assert t == self.T - 1
        try:
            if self._pres_trans[i, t - 1]:
                # i pres at t-1 and t, can include forward temp msg term
                marg *= self.forward_temp_msg_term(i, t)
        except:
            # t must be 0, so no t-1, that's fine
            assert t == 0
        self._psi_e[t][i][j_idx] = marg

    def cavity_spatial_message(self, i, j, t):
        # sum_r(p_rq^t *
        # self._psi_e[t][k][i_idx (=self.nbrs[t][k]==i)][r]
        # for k in self.nbrs[t][i]!=j (= self.nbrs[t][i][j_idx]))
        nbrs = np.array([k for k in self.nbrs[t][i] if k != j])
        # sum_terms = np.array(
        #     [
        #         self.block_edge_prob[:, :, t].T
        #         @ self._psi_e[t][k][self.nbrs[t][k] == i]
        #         for k in nbrs
        #     ]
        # )  # |N_i| - 1 x Q
        # return np.prod(sum_terms, axis=0)
        msg = np.ones((self.Q,))
        if len(nbrs) > 0:
            beta = self.block_edge_prob[:, :, t]
            for k in nbrs:
                if len(self.nbrs[t][k] > 0):
                    idx = self.nbrs[t][k] == i
                    if idx.sum() == 0:
                        print("Fault:", k, t)
                else:
                    print("Fault:", k, t)
                ktoi_msgs = self._psi_e[t][k][idx, :].reshape(
                    -1
                )  # for whatever reason this stays 2d, so need to flatten first
                # print("jtoi_msg:", jtoi_msgs.shape)
                tmp = beta.T @ ktoi_msgs
                # print("summed:", tmp.shape)
                msg *= tmp
        else:
            print("Fault:", i, t)
        return msg

    def forward_temp_msg_term(self, i, t):
        # sum_qprime(self.trans_prob(qprime,q)*self._psi_t[i,t-1,qprime,q,1])
        # from t-1 to t
        try:
            prod = self.trans_prob.T @ self._psi_t[i, t - 1, :, :, 1]
            return np.diag(prod)
        except:
            # must have t=0 so t-1 outside of range, no forward message, but do have alpha instead
            return self.model._alpha

    def backward_temp_msg_term(self, i, t):
        # sum_qprime(self.trans_prob(q,qprime)*self._psi_t[i,t,qprime,q,0])
        # from t+1 to t
        try:
            prod = self.trans_prob @ self._psi_t[i, t, :, :, 0]
            return np.diag(prod)
        except:
            # t=T outside of range, so no backward message
            return np.ones((self.Q,))

    def update_temporal_messages(self, i, t):
        # know that i present at t and t+1, so want to update forward from t,
        # backward from t+1
        self.update_backward_temporal_message(i, t)
        self.update_forward_temporal_message(i, t)

    def update_forward_temporal_message(self, i, t):
        # size N x T - 1 x Q, w itq entry corresponding to temporal message from i at t to
        # i at t+1 (i.e. covering t=0 to T-2)
        marg = (
            self.meta_prob(i, t)
            * self.forward_temp_msg_term(i, t)
            * np.exp(self._h[:, t])
            * self.spatial_msg_term(i, t)
        )
        self._psi_t[t][i][1] = marg

    def update_backward_temporal_message(self, i, t):
        # size N x T - 1 x Q, w itq entry corresponding to temporal message from i at t+1
        # to i at t (i.e. covering t=0 to T-2 again)
        try:
            marg = (
                self.meta_prob(i, t + 1)
                * self.backward_temp_msg_term(i, t + 1)
                * np.exp(self._h[:, t + 1])
                * self.spatial_msg_term(i, t + 1)
            )
            self._psi_t[t][i][0] = marg
        except:
            # at t=T, no backward msg
            assert t == self.T - 1

    def spatial_msg_term(self, i, t):
        msg = np.ones((self.Q,))
        nbrs = self.nbrs[t][i]
        beta = self.block_edge_prob[:, :, t]
        if len(nbrs) > 0:
            # print("Beta:", beta)
            # sum_terms = np.array(
            #     [
            #         self.block_edge_prob[:, :, t].T
            #         @ self._psi_e[t][j][self.nbrs[t][j] == i]
            #         for j in nbrs
            #     ]
            # )  # |N_i| x Q
            # np.prod(sum_terms, axis=0) # can't use axis kwarg for prod in numba
            # for sum_term in sum_terms:
            #     msg *= sum_term
            for j in nbrs:
                if len(self.nbrs[t][j] > 0):
                    idx = self.nbrs[t][j] == i
                    if idx.sum() == 0:
                        print("Fault:", j, t)
                else:
                    print("Fault:", j, t)
                jtoi_msgs = self._psi_e[t][j][idx, :].reshape(
                    -1
                )  # for whatever reason this stays 2d, so need to flatten first
                # print("jtoi_msg:", jtoi_msgs.shape)
                tmp = beta.T @ jtoi_msgs
                # print("summed:", tmp.shape)
                msg *= tmp
        else:
            print("Fault:", i, t)
        return msg

    def meta_prob(self, i, t):
        return self.model.meta_lkl[i, t, :]

    @property
    def trans_prob(self):
        # transition matrix
        return self.model._pi

    def correct_messages(self):
        # make sure messages sum to one over groups, and are nonzero for numerical stability
        pass

    def collect_messages(self):
        pass

    def store_messages(self,):
        pass

    def learning_step(self,):
        # this should fix normalisation of expected marginals so sum to one - might not due to learning rate. Unsure if necessary
        pass

    @property
    def block_edge_prob(self):
        # return beta_{qr}^t as Q x Q x T array
        # TODO: alter suitably for DC case
        return self.model._beta

    def init_h(self):
        # update within each timestep is unchanged from static case,
        # i.e. = \sum_r \sum_i \psi_r^{it} p_{rq}^t
        # self._h = np.einsum("itr,rqt->qt", self.node_marg, self.block_edge_prob)
        self._h = (
            self.block_edge_prob.transpose(1, 0, 2) * self.node_marg.sum(axis=0).T
        ).sum(axis=1)

    def update_h(self, i, t, old_marg):
        self._h += self.block_edge_prob[:, :, t].T @ (
            self.node_marg[i, t, :] - old_marg
        )

    def convergence(self):
        pass

    def compute_free_energy(self):
        pass

    def compute_entropy(self):
        pass

    def compute_overlap(self):
        pass

    def get_marginal_entropy(self):
        pass

    def update_node_marg(self):
        for i in range(self.N):
            for t in range(self.T):
                if self.model.degs[i, t, :].sum() > 0:
                    tmp = self.spatial_msg_term(i, t)
                    tmp *= self.backward_temp_msg_term(i, t)
                    tmp *= self.forward_temp_msg_term(i, t)
                    tmp /= tmp.sum()
                    self.node_marg[i, t, :] = tmp
                else:
                    self.node_marg[i, t, :] = 0.0

    def update_twopoint_marginals(self):
        # TODO: ask Leto to check the two-point marginals eqns, as there's a chance they should include product of metaprob terms for i,j at t.
        # node_marg = None
        self.update_twopoint_spatial_marg()
        print("\tUpdated twopoint spatial marg")
        self.update_twopoint_temp_marg()
        print("\tUpdated twopoint temp marg")
        # self.twopoint_marginals = [twopoint_e_marg, twopoint_t_marg]
        return (self.twopoint_e_marg, self.twopoint_t_marg)

    def _zero_twopoint_e_marg(self):
        # instantiate as list construction more expensive
        # duplicate spatial msg idea for twopoint marg, so have \psi^{ijt}_{qr} = twopoint_e_marg[t][i][j_idx in nbrs[t][i],qr], then minimal memory required
        # self.twopoint_e_marg = List.empty_list(
        #     ListType(float64[:, :, :])
        # )  # np.zeros((self.N,self.N,self.T,self.Q,self.Q))
        tmp = List()
        for t in range(self.T):
            t_tmp = List()
            for i in range(self.N):
                i_tmp = np.zeros((len(self.nbrs[t][i]), self.Q, self.Q))
                t_tmp.append(i_tmp)
            tmp.append(t_tmp)
        self.twopoint_e_marg = tmp

    def update_twopoint_spatial_marg(self):
        # p_qrt = block_edge_prob
        # psi_e in shape [t][i][j_idx in nbrs[t][i],q] (list(list(2d array)))
        for i, j, t in self._edge_locs:
            # print(i, j, t)
            j_idx = self.nbrs[t][i] == j
            i_idx = self.nbrs[t][j] == i
            tmp = self.block_edge_prob[:, :, t]
            tmp *= np.outer(self._psi_e[t][i][j_idx, :], self._psi_e[t][j][i_idx, :])
            tmp *= np.outer(self._psi_e[t][j][i_idx, :], self._psi_e[t][i][j_idx, :])
            self.twopoint_e_marg[t][i][j_idx, :, :] = tmp / tmp.sum()

    def update_twopoint_temp_marg(self):
        # recall t msgs in shape (i,t,q,q',2), w t from 0 to T-2, and final dim (backwards from t+1, forwards from t)
        self.twopoint_t_marg = np.zeros(
            (self.N, self.T - 1, self.Q, self.Q)
        )  # TODO: move instantiation to init
        for i in range(self.N):
            for t in range(self.T - 1):
                tmp = self.trans_prob
                tmp *= (
                    self._psi_t[i, t, :, :, 1] * self._psi_t[i, t, :, :, 0].T
                )  # TODO: check this
                tmp *= self._psi_t[i, t, :, :, 0].T * self._psi_t[i, t, :, :, 1]
                self.twopoint_t_marg[i, t, :, :] = tmp / tmp.sum()


class BP:
    """Pure Python wrapper of BPBase class to allow optional/keyword arguments
    """

    def __init__(self, dsbmm: DSBMM):
        self.model = dsbmm
        self.jit_model = BPBase(dsbmm.jit_model)
        self.jit_model.get_neighbours()

    @property
    def n_timesteps(self):
        return self.jit_model.T

    @property
    def beta(self):
        # temperature for Boltzmann dist
        # looks like this enters eqns via external field, i.e. e^-beta*h_q/N rather than e^-h_q/N
        # and also then in derivatives, hence possible need to have fns calculating vals of derivs  - need to be careful
        return self.jit_model._beta

    @beta.setter
    def set_beta(self, value):
        self.jit_model._beta = value

    def fit(self, data):
        pass

    def get_neighbours(self):
        self.jit_model.get_neighbours()

    def init_messages(self, mode="random"):
        self.jit_model.init_messages(mode)

    def update_messages(self,):
        self.jit_model.update_messages()

    def update_spatial_message(self, i, j, t):
        self.jit_model.update_spatial_message(i, j, t)

    def cavity_spatial_message(self, i, j, t):
        # sum_r(p_rq^t *
        # self._psi_e[t][k][i_idx (=self.nbrs[t][k]==i)][r]
        # for k in self.nbrs[t][i]!=j (= self.nbrs[t][i][j_idx]))
        return self.jit_model.cavity_spatial_message(i, j, t)

    def forward_temp_msg_term(self, i, t):
        return self.jit_model.forward_temp_msg_term(i, t)

    def backward_temp_msg_term(self, i, t):
        return self.jit_model.backward_temp_msg_term(i, t)

    def update_temporal_messages(self, i, t):
        self.jit_model.update_temporal_messages(i, t)

    def update_forward_temporal_message(self, i, t):
        self.jit_model.update_forward_temporal_message(i, t)

    def update_backward_temporal_message(self, i, t):
        self.jit_model.update_backward_temporal_message(i, t)

    def spatial_msg_term(self, i, t):
        return self.jit_model.spatial_msg_term(i, t)

    def meta_prob(self, i, t):
        return self.model.meta_lkl[i, t]

    @property
    def trans_prob(self):
        # transition matrix
        return self.model.jit_model._pi

    def correct_messages(self):
        # make sure messages sum to one over groups, and are nonzero for numerical stability
        pass

    def collect_messages(self):
        pass

    def store_messages(self,):
        pass

    def learning_step(self,):
        # this should fix normalisation of expected marginals so sum to one - might not due to learning rate. Unsure if necessary
        pass

    def block_edge_prob(self):
        # return beta_{qr}^t as Q x Q x T array
        # TODO: alter suitably for DC case
        pass

    def init_h(self):
        # update within each timestep is unchanged from static case,
        # i.e. = \sum_r \sum_i \psi_r^{it} p_{rq}^t
        self.jit_model.init_h()

    def update_h(self, i, t, old_marg):
        self.jit_model.update_h(i, t, old_marg)

    def convergence(self):
        pass

    def compute_free_energy(self):
        pass

    def compute_entropy(self):
        pass

    def compute_overlap(self):
        pass

    def get_marginal_entropy(self):
        pass

    def update_node_marg(self):
        self.jit_model.update_node_marg()

    def update_twopoint_marginals(self):
        self.jit_model.update_twopoint_marginals()
        return [self.jit_model.twopoint_e_marg, self.jit_model.twopoint_t_marg]

    def update_twopoint_spatial_marg(self):
        # p_qrt = block_edge_prob
        # psi_e in shape [t][i][j_idx in nbrs[t][i],q] (list(list(2d array)))
        self.jit_model.update_twopoint_spatial_marg()

    def update_twopoint_temp_marg(self):
        # recall t msgs in shape (i,t,q,q',2), w t from 0 to T-2, and final dim (backwards from t+1, forwards from t)
        self.jit_model.update_twopoint_temp_marg()

