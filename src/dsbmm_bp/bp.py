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

X_ex = List.empty_list(float64[:, :, ::1])
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

# TODO: don't hardcode these
TOL = 1e-50  # min value permitted for msgs etc (for numerical stability)
LARGE_DEG_THR = 20  # threshold of node degree above which log msgs calculated
# (for numerical stability)


@jitclass
class BPBase:
    N: int
    Q: int
    T: int
    _beta: float  # temperature
    deg_corr: bool
    model: DSBMMBase.class_type.instance_type
    A: float64[:, :, ::1]
    # X: list[np.ndarray]
    X: X_type
    # c_qrt: float64[:, :, :]  # N*p_qrt
    # n_qt: float64[:, :]  # prior prob
    _psi_e: psi_e_type  # spatial messages
    _psi_t: Array(
        float64, ndim=4, layout="C"
    )  # temporal messages, assume (i,t,q,q',2) for t in 0 to T-2, where first loc of final dim is backward messages from t+1 to t
    # and second loc is forward messages from t to t+1
    _h: float64[:, ::1]  # external field for each group, with shape (Q,T)
    node_marg: Array(
        float64, ndim=3, layout="C"
    )  # marginal group probabilities for each node
    # - NB this also referred to as nu, eta or
    # other terms elsewhere
    # twopoint_e_marg: float64[:, :, :, :, :]
    twopoint_e_marg: twopoint_e_type  # [t][i][len(nbrs[t][i]),Q,Q]
    twopoint_t_marg: Array(float64, ndim=4, layout="C")  # N x T - 1 X Q x Q
    _pres_nodes: bool_[:, ::1]
    _pres_trans: bool_[:, ::1]
    _edge_locs: int64[:, ::1]
    _trans_locs: int64[:, ::1]
    nbrs: nbrs_type
    n_msgs: int64
    msg_diff: float64

    def __init__(self, dsbmm: DSBMMBase.class_type.instance_type):
        self.model = dsbmm
        self.T = self.model.T
        self.N = self.model.N
        self.Q = self.model.Q
        self.deg_corr = self.model.deg_corr
        self.A = self.model.A
        self.n_msgs = np.count_nonzero(self.A) + self.N * (self.T - 1) * 2
        self.X = self.model.X
        # start with given membership and corresponding messages, iterate until reach fixed point
        # given messages at fixed point, can update parameters to their most likely values - these emerge naturally
        # by requiring that the fixed point provides stationary free energy w.r.t. parameters
        self.get_neighbours()
        self.node_marg = np.zeros((self.N, self.T, self.Q))
        self._zero_twopoint_e_marg()
        # print(self._edge_locs)
        self.msg_diff = 0.0

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
            # initialise by random messages and marginals
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
            ## INIT MARGINALS ##
            self.node_marg = np.random.rand(self.N, self.T, self.Q)
            self.node_marg /= np.expand_dims(self.node_marg.sum(axis=2), 2)
            ## INIT MESSAGES ##
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
            self._psi_e = tmp  # msgs from [t][i] to nbr j about i being in q (so 4d)
            # print("psi_e updated")
            # self._psi_t = [
            #     [np.random.rand(2, self.Q) for i in range(self.N)]
            #     for t in range(self.T - 1)
            # ]
            # self._psi_t = [
            #     [msg / msg.sum(axis=1) for msg in self._psi_t[t]]
            #     for t in range(self.T - 1)
            # ]
            self._psi_t = np.random.rand(self.N, self.T - 1, self.Q, 2)
            # self._psi_t /= self._psi_t.sum(axis=3)[:, :, :, np.newaxis, :]
            self._psi_t /= np.expand_dims(
                self._psi_t.sum(axis=2), 2
            )  # msgs from i at t forwards/backwards
            # about being in group q,
            # so again 4d
            # assert np.all((self._psi_t.sum(axis=3) - 1) ** 2 < 1e-14)
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
            ## INIT MARGINALS ##
            pass
            ## INIT MESSAGES ##
            pass

    def forward_temp_msg_term(self, i, t):
        # sum_qprime(self.trans_prob(qprime,q)*self._psi_t[i,t-1,qprime,1])
        # from t-1 to t
        try:
            # print(self.trans_prob)
            # print(self.trans_prob.T)
            out = np.ascontiguousarray(self.trans_prob.T) @ np.ascontiguousarray(
                self._psi_t[i, t - 1, :, 1]
            )
        except:
            # must have t=0 so t-1 outside of range, no forward message, but do have alpha instead
            assert t == 0
            out = self.model._alpha
        return out

    def backward_temp_msg_term(self, i, t):
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
        # sum_qprime(self.trans_prob(q,qprime)*self._psi_t[i,t,qprime,0])
        # from t+1 to t
        try:
            # TODO: remove pushes to contiguous array and just write out multiplication
            out = np.ascontiguousarray(self.trans_prob) @ np.ascontiguousarray(
                self._psi_t[i, t, :, 0]
            )
            try:
                assert not np.all(out < TOL)
            except:
                print("(i,t):", i, t)
                print("Trans:", self.trans_prob)
                print("Psi_t:", self._psi_t[i, t, :, 0])
                print("psi_t shape:", self._psi_t.shape)
                print("backward out:", out)
                raise RuntimeError("Problem with backward msg term")
        except:
            # t=T outside of range, so no backward message
            assert t == self.T - 1
            out = np.ones((self.Q,))
        return np.ascontiguousarray(out)

    def spatial_msg_term_small_deg(self, i, t, nbrs):
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
        beta = self.block_edge_prob[:, :, t]

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

        msg = np.ones((self.Q,))
        field_iter = np.zeros((len(nbrs), self.Q))
        for nbr_idx, j in enumerate(nbrs):
            if len(self.nbrs[t][j] > 0):
                idx = self.nbrs[t][j] == i
                if idx.sum() == 0:
                    print("Fault, i should be nbr of j:", j, t)
                    raise RuntimeError("Problem calc spatial msg term")
            else:
                print("Fault, j has no nbrs but should at least have i:", j, t)
                raise RuntimeError("Problem calc spatial msg term")
            jtoi_msgs = self._psi_e[t][j][idx, :].reshape(
                -1
            )  # for whatever reason this stays 2d, so need to flatten first
            # print("jtoi_msg:", jtoi_msgs.shape)
            tmp = np.ascontiguousarray(beta.T) @ np.ascontiguousarray(jtoi_msgs)
            field_iter[nbr_idx, :] = tmp
            # print("summed:", tmp.shape)
            msg *= tmp
            try:
                assert not np.all(msg < TOL)
            except:
                print("(i,j,t):", i, j, t)
                print("deg[i,t]:", len(nbrs))
                print("jtoi:", jtoi_msgs)
                print("Beta:", beta)
                print("tmp:", tmp)
                print("spatial msg term:", msg)
                raise RuntimeError("Problem vanishing spatial msg term")
        return msg, field_iter

    def spatial_msg_term_large_deg(self, i, t, nbrs):
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
        beta = self.block_edge_prob[:, :, t]
        # deg_i = len(nbrs)
        msg = np.zeros((self.Q,))
        log_field_iter = np.zeros((len(nbrs), self.Q))
        # TODO: make logger that tracks this
        # print("Large deg version used")
        max_log_msg = -100000.0
        for nbr_idx, j in enumerate(nbrs):
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
            tmp = np.log(np.ascontiguousarray(beta.T) @ np.ascontiguousarray(jtoi_msgs))
            log_field_iter[nbr_idx, :] = tmp
            # print("summed:", tmp.shape)
            msg += tmp
            max_msg_log = msg.max()
            if max_msg_log > max_log_msg:
                max_log_msg = max_msg_log
        return msg, max_log_msg, log_field_iter

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

    def update_h(self, i, t, sign):
        self._h[:, t] += (
            sign
            * np.ascontiguousarray(self.block_edge_prob[:, :, t].T)
            @ np.ascontiguousarray(self.node_marg[i, t, :])
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
        node_update_order = np.arange(self.N)
        np.random.shuffle(node_update_order)
        time_update_order = np.arange(self.T)
        np.random.shuffle(time_update_order)
        for i in node_update_order:
            for t in time_update_order:
                self.update_h(i, t, -1.0)
                nbrs = self.nbrs[t][i]
                deg_i = len(nbrs)
                if deg_i > 0:
                    if deg_i < LARGE_DEG_THR:
                        spatial_msg_term, field_iter = self.spatial_msg_term_small_deg(
                            i, t, nbrs
                        )
                        tmp = spatial_msg_term
                        if t < self.T - 1:
                            back_term = self.backward_temp_msg_term(i, t)
                            tmp *= back_term
                        ## UPDATE BACKWARDS MESSAGES FROM i AT t ##
                        if t > 0:
                            tmp_backwards_msg = tmp.copy()
                            tmp_backwards_msg[tmp_backwards_msg < TOL] = TOL
                            tmp_backwards_msg /= tmp_backwards_msg.sum()
                            self.msg_diff += (
                                np.abs(
                                    tmp_backwards_msg - self._psi_t[i, t - 1, :, 0]
                                ).mean()
                                / self.n_msgs
                            )
                            self._psi_t[i, t - 1, :, 0] = tmp_backwards_msg
                            tmp *= self.forward_temp_msg_term(i, t)
                        ## UPDATE SPATIAL MESSAGES FROM i AT t ##
                        tmp_spatial_msg = np.expand_dims(tmp, 0) / field_iter
                        for nbr_idx in range(deg_i):
                            for q in range(self.Q):
                                if tmp_spatial_msg[nbr_idx, q] < TOL:
                                    tmp_spatial_msg[nbr_idx, q] = TOL
                        tmp_spatial_msg /= np.expand_dims(
                            tmp_spatial_msg.sum(axis=1), 1
                        )
                        self.msg_diff += (
                            np.abs(tmp_spatial_msg - self._psi_e[t][i]).mean()
                            * deg_i
                            / self.n_msgs
                        )  # NB need to mult by deg_i so weighted correctly
                        self._psi_e[t][i] = tmp_spatial_msg
                        ## UPDATE FORWARDS MESSAGES FROM i AT t ##
                        if t < self.T - 1:
                            tmp_forwards_msg = (
                                tmp / back_term
                            )  # TODO: fix if back_term << 1
                            tmp_forwards_msg[tmp_forwards_msg < TOL] = TOL
                            tmp_forwards_msg /= tmp_forwards_msg.sum()
                            self.msg_diff += (
                                np.abs(
                                    tmp_forwards_msg - self._psi_t[i, t, :, 1]
                                ).mean()
                                / self.n_msgs
                            )
                            self._psi_t[i, t, :, 1] = tmp_forwards_msg
                        ## UPDATE MARGINAL OF i AT t ##
                        tmp_marg = tmp
                        tmp_marg[tmp_marg < TOL] = TOL
                        tmp_marg = tmp_marg / tmp_marg.sum()
                        self.node_marg[i, t, :] = tmp_marg

                    else:
                        (
                            spatial_msg_term,
                            max_log_spatial_msg_term,
                            log_field_iter,
                        ) = self.spatial_msg_term_large_deg(i, t, nbrs)
                        tmp = spatial_msg_term
                        back_term = np.log(self.backward_temp_msg_term(i, t))
                        tmp += back_term
                        ## UPDATE BACKWARDS MESSAGES FROM i AT t ##
                        if t > 0:
                            tmp_backwards_msg = np.exp(tmp - max_log_spatial_msg_term)
                            tmp_backwards_msg[tmp_backwards_msg < TOL] = TOL
                            tmp_backwards_msg /= tmp_backwards_msg.sum()
                            self._psi_t[i, t - 1, :, 0] = tmp_backwards_msg
                        tmp += np.log(self.forward_temp_msg_term(i, t))
                        ## UPDATE SPATIAL MESSAGES FROM i AT t ##
                        tmp_spatial_msg = np.expand_dims(tmp, 0) - log_field_iter
                        log_field_iter_max = np.array(
                            [nbr_fld.max() for nbr_fld in tmp_spatial_msg]
                        )
                        tmp_spatial_msg = np.exp(
                            tmp_spatial_msg - np.expand_dims(log_field_iter_max, 1)
                        )
                        for nbr_idx in range(deg_i):
                            for q in range(self.Q):
                                if tmp_spatial_msg[nbr_idx, q] < TOL:
                                    tmp_spatial_msg[nbr_idx, q] = TOL
                        tmp_spatial_msg /= np.expand_dims(
                            tmp_spatial_msg.sum(axis=1), 1
                        )
                        self._psi_e[t][i] = tmp_spatial_msg
                        ## UPDATE FORWARDS MESSAGES FROM i AT t ##
                        if t < self.T - 1:
                            tmp_forwards_msg = np.exp(
                                tmp - max_log_spatial_msg_term - back_term
                            )
                            tmp_forwards_msg[tmp_forwards_msg < TOL] = TOL
                            tmp_forwards_msg /= tmp_forwards_msg.sum()
                            self._psi_t[i, t, :, 1] = tmp_forwards_msg
                        ## UPDATE MARGINAL OF i AT t ##
                        tmp_marg = np.exp(tmp - max_log_spatial_msg_term)
                        tmp_marg[tmp_marg < TOL] = TOL
                        tmp_marg /= tmp_marg.sum()
                        self.node_marg[i, t, :] = tmp_marg

                else:
                    self.node_marg[i, t, :] = 0.0
                self.update_h(i, t, 1.0)

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
            j_idx = (
                self.nbrs[t][i] == j
            )  # TODO: create inverse array that holds these values
            #       in memory rather than calc on the fly each time
            i_idx = self.nbrs[t][j] == i
            tmp = self.block_edge_prob[:, :, t]
            tmp *= np.outer(self._psi_e[t][i][j_idx, :], self._psi_e[t][j][i_idx, :])
            tmp *= np.outer(self._psi_e[t][j][i_idx, :], self._psi_e[t][i][j_idx, :])
            for q in range(self.Q):
                for r in range(self.Q):
                    if tmp[q, r] < TOL:
                        tmp[q, r] = TOL
            self.twopoint_e_marg[t][i][j_idx, :, :] = tmp / tmp.sum()

    def update_twopoint_temp_marg(self):
        # recall t msgs in shape (i,t,q,q',2), w t from 0 to T-2, and final dim (backwards from t+1, forwards from t)
        self.twopoint_t_marg = np.zeros(
            (self.N, self.T - 1, self.Q, self.Q)
        )  # TODO: move instantiation to init
        for i in range(self.N):
            for t in range(self.T - 1):
                tmp = self.trans_prob
                tmp *= np.outer(
                    self._psi_t[i, t, :, 1], self._psi_t[i, t, :, 0]
                )  # TODO: check this
                tmp *= np.outer(self._psi_t[i, t, :, 0], self._psi_t[i, t, :, 1])
                for q in range(self.Q):
                    for qprime in range(self.Q):
                        if tmp[q, qprime] < TOL:
                            tmp[q, qprime] = TOL
                self.twopoint_t_marg[i, t, :, :] = tmp / tmp.sum()

    def zero_diff(self):
        self.msg_diff = 0.0


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

    # def spatial_msg_term(self, i, t):
    #     return self.jit_model.spatial_msg_term(i, t)

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

    def zero_diff(self):
        self.jit_model.zero_diff()
