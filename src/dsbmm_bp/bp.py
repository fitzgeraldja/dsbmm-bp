# TODO: Allow X to be ordinal categorical encoding rather than OHE for memory efficiency,
# TODO: think about how could allow batching.
import numpy as np
import yaml
from dsbmm import DSBMM
from dsbmm import DSBMMBase
from numba import bool_
from numba import float64
from numba import int32
from numba import int64
from numba import typeof
from numba.experimental import jitclass
from numba.typed import List
from numba.types import Array
from numba.types import ListType

# from numba import float32

# from numba.types import unicode_type
# from utils import numba_ix

# from numba_dsbmm_methods import *
# from numba_bp_methods import *

X_ex = List.empty_list(float64[:, :, ::1])
X_type = typeof(X_ex)
psi_e_ex = List.empty_list(ListType(Array(float64, ndim=2, layout="C")))  # noqa: F821
psi_e_type = typeof(psi_e_ex)
# psi_t_ex = List.empty_list(ListType(float64[:, :])) # unnecessary for same reasons as below
# psi_t_type = typeof(psi_t_ex)
nbrs_ex = List.empty_list(ListType(Array(int32, ndim=1, layout="C")))  # noqa: F821
nbrs_type = typeof(nbrs_ex)
twopoint_e_ex = List.empty_list(
    ListType(Array(float64, ndim=3, layout="C"))
)  # noqa: F821
# TODO: replace as array type of size (Q^2 sum_t E_t), along w lookup table for idx of i,j,t as this should be sufficient
# given implementation below - would this actually result in speedup over lru cache implementation?
# twopoint_t_ex = List.empty_list(ListType(float64[:,:])) # unnecessary for t marg, as memory size O(NTQ^2) at most O(Q^2\sum_t E_t) size of constrained formulation
# of e marg (for sparse networks w E_t approx N for each t), and arrays generally preferable structure if significant overall savings not possible
twopoint_e_type = typeof(twopoint_e_ex)

with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
TOL = config["tol"]  # min value permitted for msgs etc (for numerical stability)
LARGE_DEG_THR = config[
    "large_deg_thr"
]  # threshold of node degree above which log msgs calculated
# (for numerical stability)
RANDOM_ONLINE_UPDATE_MSG = config[
    "random_online_update_msg"
]  # if true then update messages online (always using most recent vals), otherwise update all messages simultaneously


@jitclass
class BPBase:
    N: int
    Q: int
    T: int
    _beta: float  # temperature
    deg_corr: bool
    directed: bool
    use_meta: bool
    model: DSBMMBase.class_type.instance_type
    A: float64[:, :, ::1]
    # X: list[np.ndarray]
    X: X_type
    # c_qrt: float64[:, :, :]  # N*p_qrt
    # n_qt: float64[:, :]  # prior prob
    _psi_e: psi_e_type  # spatial messages
    _psi_t: Array(
        float64, ndim=4, layout="C"  # noqa: F821
    )  # temporal messages, assume (i,t,q,q',2) for t in 0 to T-2, where first loc of final dim is backward messages from t+1 to t
    # and second loc is forward messages from t to t+1
    _h: float64[:, ::1]  # external field for each group, with shape (Q,T)
    node_marg: Array(
        float64, ndim=3, layout="C"  # noqa: F821
    )  # marginal group probabilities for each node
    # - NB this also referred to as nu, eta or
    # other terms elsewhere
    # twopoint_e_marg: float64[:, :, :, :, :]
    twopoint_e_marg: twopoint_e_type  # [t][i][len(nbrs[t][i]),Q,Q]
    twopoint_t_marg: Array(
        float64, ndim=4, layout="C"  # noqa: F821
    )  # N x T - 1 X Q x Q
    _pres_nodes: bool_[:, ::1]
    _pres_trans: bool_[:, ::1]
    _edge_vals: float64[:, ::1]
    _trans_locs: int64[:, ::1]
    nbrs: nbrs_type
    n_msgs: int64
    msg_diff: float64
    verbose: bool

    def __init__(self, dsbmm: DSBMMBase.class_type.instance_type):
        self.model = dsbmm
        self.T = self.model.T
        self.N = self.model.N
        self.Q = self.model.Q
        self.deg_corr = self.model.deg_corr
        self.directed = self.model.directed
        self.use_meta = self.model.use_meta
        self.verbose = self.model.verbose
        self.A = self.model.A
        self.n_msgs = self.model.E.sum() + self.N * (self.T - 1) * 2
        self.X = self.model.X
        # start with given membership and corresponding messages, iterate until reach fixed point
        # given messages at fixed point, can update parameters to their most likely values - these emerge naturally
        # by requiring that the fixed point provides stationary free energy w.r.t. parameters
        self.get_neighbours()
        self.node_marg = np.zeros((self.N, self.T, self.Q))
        self._zero_twopoint_e_marg()
        self.msg_diff = 0.0
        self.twopoint_t_marg = np.zeros((self.N, self.T - 1, self.Q, self.Q))

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
        # TODO: directed version!
        self._pres_nodes = np.zeros(
            (self.N, self.T), dtype=bool_
        )  # N x T boolean array w i,t True if i present in net at time t

        self._pres_nodes = (self.A.sum(axis=0) > 0) | (self.A.sum(axis=1) > 0)
        self._pres_trans = (
            self._pres_nodes[:, :-1] * self._pres_nodes[:, 1:]
        )  # returns N x T-1 array w i,t true if i
        # present at both t and t+1 (so can measure trans)
        self._edge_vals = (
            self.model._edge_vals
        )  # E x 4 array w rows [i,j,t,val] where A[i,j,t]==val != 0
        self._trans_locs = np.array(
            list(zip(*self._pres_trans.nonzero()))
        )  # N_trans x 2
        tmp = List()
        for t in range(self.T):
            tmp2 = List()
            for i in range(self.N):
                tmp2.append(np.nonzero(self.A[i, :, t])[0].astype(np.int32))
            tmp.append(tmp2)
        # tmp = [
        #     [self.A[i, :, t].nonzero()[0] for i in range(self.N)] for t in range(self.T)
        # ]  # so self.nbrs[t][i] gives
        self.nbrs = tmp
        # nbrs of i at t (possibly none)
        # TODO: make inverse lookup for nbrs:
        # void blockmodel:: graph_build_neis_inv()
        # {
        #     //build graph_neis_inv
        #     graph_neis_inv.resize(N);
        #     for (int i=0;i<N;i++){
        #         graph_neis_inv[i].resize(graph_neis[i].size());
        #         for (int idxij=0;idxij<graph_neis[i].size();idxij++){
        #             int j=graph_neis[i][idxij];
        #             int target=0;
        #             for (int idxji=0;idxji<graph_neis[j].size();idxji++){
        #                 if (graph_neis[j][idxji]==i){
        #                     target = idxji;
        #                     break;
        #                 }
        #             }
        #             graph_neis_inv[i][idxij]=target;
        #         }
        #     }
        # }

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

    # TODO: implement lru cached nbr and inverse nbr lookup functions
    # - should basically be same efficiency as putting in memory

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
                    # assert np.isnan(msg).sum() == 0
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
            # assert np.isnan(self._psi_t).sum() == 0
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
            # TODO: don't hardcode
            p = 0.8
            ## INIT MARGINALS ##
            one_hot_Z = np.zeros((self.N, self.T, self.Q))
            for i in range(self.N):
                for t in range(self.T):
                    one_hot_Z[i, t, self.model.Z[i, t]] = 1.0

            self.node_marg = p * one_hot_Z + (1 - p) * np.random.rand(
                self.N, self.T, self.Q
            )
            self.node_marg /= np.expand_dims(self.node_marg.sum(axis=2), 2)

            ## INIT MESSAGES ##
            tmp = List()
            for t in range(self.T):
                tmp2 = List()
                for i in range(self.N):
                    n_nbrs = len(self.nbrs[t][i])
                    if n_nbrs > 0:
                        msg = p * np.expand_dims(one_hot_Z[i, t, :], 0) + (
                            1 - p
                        ) * np.random.rand(n_nbrs, self.Q)
                        msg /= np.expand_dims(msg.sum(axis=1), 1)
                    else:
                        # print("WARNING: empty nodes not properly handled yet")
                        msg = np.empty((1, self.Q), dtype=np.float64)
                    # assert np.isnan(msg).sum() == 0
                    tmp2.append(msg)
                # print("Trying to update psi_e")
                tmp.append(tmp2)
            self._psi_e = tmp

            self._psi_t = (1 - p) * np.random.rand(self.N, self.T - 1, self.Q, 2)
            for t in range(self.T - 1):
                self._psi_t[:, t, :, 0] += p * one_hot_Z[:, t + 1, :]
                self._psi_t[:, t, :, 1] += p * one_hot_Z[:, t, :]
            self._psi_t /= np.expand_dims(self._psi_t.sum(axis=2), 2)

    def forward_temp_msg_term(self, i, t):
        # sum_qprime(self.trans_prob(qprime,q)*self._psi_t[i,t-1,qprime,1])
        # from t-1 to t
        try:
            # print(self.trans_prob)
            # print(self.trans_prob.T)
            out = np.ascontiguousarray(self.trans_prob.T) @ np.ascontiguousarray(
                self._psi_t[i, t - 1, :, 1]
            )
            for q in range(self.Q):
                if out[q] < TOL:
                    out[q] = TOL
        except IndexError:
            # must have t=0 so t-1 outside of range, no forward message, but do have alpha instead - stopped now as need for backward term
            assert t == 0
            # out = self.model._alpha
        return out

    def backward_temp_msg_term(self, i, t):
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
        # sum_qprime(self.trans_prob(q,qprime)*self._psi_t[i,t,qprime,0])
        # from t+1 to t
        try:
            # TODO: remove pushes to contiguous array and just write out multiplication!
            out = np.ascontiguousarray(self.trans_prob) @ np.ascontiguousarray(
                self._psi_t[i, t, :, 0]
            )
            for q in range(self.Q):
                if out[q] < TOL:
                    out[q] = TOL
            # try:
            #     assert not np.all(out < TOL)
            #     assert np.isnan(out).sum() == 0
            #     assert np.all(out >= 0)
            # except:
            #     print("(i,t):", i, t)
            #     print("Trans:", self.trans_prob)
            #     print("Psi_t back:", self._psi_t[i, t, :, 0])
            #     print("psi_t shape:", self._psi_t.shape)
            #     print("backward out:", out)
            #     raise RuntimeError("Problem with backward msg term")
        except IndexError:
            # t=T outside of range, so no backward message
            assert t == self.T - 1
            out = np.ones((self.Q,))
        return np.ascontiguousarray(out)

    def spatial_msg_term_small_deg(self, i, t, nbrs):
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
            # tmp = np.ascontiguousarray(beta.T) @ np.ascontiguousarray(
            #     jtoi_msgs
            # )
            tmp = np.zeros(self.Q)
            if self.deg_corr:
                for q in range(self.Q):
                    for r in range(self.Q):
                        tmp[q] += (
                            self.model.compute_DC_lkl(
                                i,
                                j,
                                t,
                                r,
                                q,
                                self.A[i, j, t],
                            )
                            * jtoi_msgs[r]
                        )
            else:
                for q in range(self.Q):
                    for r in range(self.Q):
                        tmp[q] += beta[r, q] * jtoi_msgs[r]
            # for q in range(self.Q):
            #     if tmp[q] < TOL:
            #         tmp[q] = TOL
            # try:
            #     assert not np.isnan(tmp).sum() > 0
            #     assert not np.isinf(tmp).sum() > 0
            # except:
            #     # print("A[t]:", self.A[t])
            #     print("(i,j,t):", i, j, t)
            #     print("deg[i,t]:", len(nbrs))
            #     print("jtoi:", jtoi_msgs)
            #     print("full j msgs:", self._psi_e[t][j])
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
        if self.deg_corr:
            msg *= np.exp(-self.model.degs[i, t, 1] * self._h[:, t])
        else:
            msg *= np.exp(-1.0 * self._h[:, t])
        msg *= self.meta_prob(i, t)
        # try:
        #     assert not np.isinf(msg).sum() > 0
        # except:
        #     print("(i,t):", i, t)
        #     print("deg[i,t]:", len(nbrs))
        #     print("beta:", beta)
        #     print("meta:", self.meta_prob(i, t))
        #     print("exp(-h):", np.exp(-1.0 * self._h[:, t]))
        #     print("spatial msg term:", msg)
        #     raise RuntimeError("Problem with either meta or external field terms")
        # msg[msg < TOL] = TOL
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
        max_log_msg = -1000000.0
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
            # tmp = np.log(
            #     np.ascontiguousarray(beta.T) @ np.ascontiguousarray(jtoi_msgs)
            # )
            tmp = np.zeros(self.Q)
            if self.deg_corr:
                for q in range(self.Q):
                    for r in range(self.Q):
                        tmp[q] += (
                            self.model.compute_DC_lkl(i, j, t, r, q, self.A[i, j, t])
                            * jtoi_msgs[r]
                        )
            else:
                for q in range(self.Q):
                    for r in range(self.Q):
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
        if self.deg_corr:
            msg -= self._h[:, t] * self.model.degs[i, t, 1]
        else:
            msg -= self._h[
                :, t
            ]  # NB don't need / self.N as using p_ab to calc, not c_ab
        msg += np.log(self.meta_prob(i, t))
        return msg, max_log_msg, log_field_iter

    def meta_prob(self, i, t):
        if self.use_meta:
            return self.model.meta_lkl[i, t, :]
        else:
            return np.ones(self.Q)

    @property
    def trans_prob(self):
        # transition matrix
        return self.model._pi

    def correct_messages(self):
        # make sure messages sum to one over groups, and are nonzero for numerical stability
        pass

    def collect_messages(self):
        pass

    def store_messages(
        self,
    ):
        pass

    def learning_step(
        self,
    ):
        # this should fix normalisation of expected marginals so sum to one - might not due to learning rate. Unsure if necessary
        pass

    @property
    def block_edge_prob(self):
        if self.deg_corr:
            # will refer to as beta, but really omega or lambda (DC edge factor)
            return self.model._lam
        else:
            # return beta_{qr}^t as Q x Q x T array
            return self.model._beta

    def init_h(self):
        # update within each timestep is unchanged from static case,
        # i.e. = \sum_r \sum_i \psi_r^{it} p_{rq}^t
        # self._h = np.einsum("itr,rqt->qt", self.node_marg, self.block_edge_prob)
        # self._h = (
        #     self.block_edge_prob.transpose(1, 0, 2) * self.node_marg.sum(axis=0).T
        # ).sum(axis=1)
        self._h = np.zeros((self.Q, self.T))
        if self.deg_corr:
            for q in range(self.Q):
                for t in range(self.T):
                    for i in range(self.N):
                        for r in range(self.Q):
                            self._h[q, t] += (
                                self.block_edge_prob[r, q, t]
                                * self.node_marg[i, t, r]
                                * self.model.degs[i, t, 1]
                            )
        else:
            for q in range(self.Q):
                for t in range(self.T):
                    for i in range(self.N):
                        for r in range(self.Q):
                            self._h[q, t] += (
                                self.block_edge_prob[r, q, t] * self.node_marg[i, t, r]
                            )
        # print("h after init:", self._h)

    def update_h(self, i, t, sign):
        # self._h[:, t] += (
        #     sign
        #     * np.ascontiguousarray(self.block_edge_prob[:, :, t].T)
        #     @ np.ascontiguousarray(self.node_marg[i, t, :])
        # )
        if self.deg_corr:
            for q in range(self.Q):
                for r in range(self.Q):
                    self._h[q, t] += (
                        sign
                        * self.block_edge_prob[r, q, t]
                        * self.node_marg[i, t, r]
                        * self.model.degs[i, t, 1]
                    )
        else:
            # try:
            #     assert np.isnan(self.node_marg[i, t, :]).sum() == 0
            # except:
            #     print("i, t:", i, t)
            #     print("node_marg:", self.node_marg[i, t, :])
            #     raise ValueError("Problem with node marg")
            for q in range(self.Q):
                for r in range(self.Q):
                    self._h[q, t] += (
                        sign * self.block_edge_prob[r, q, t] * self.node_marg[i, t, r]
                    )

    def convergence(self):
        pass

    def compute_free_energy(self):
        f_site = 0.0
        f_link = 0.0
        last_term = 0.0  # something like average degree, but why?
        for i in range(self.N):
            for t in range(self.T):
                if self._pres_nodes[i, t]:
                    nbrs = self.nbrs[t][i]
                    deg_i = len(nbrs)
                    if deg_i > 0:
                        if deg_i < LARGE_DEG_THR:
                            (
                                spatial_msg_term,
                                field_iter,
                            ) = self.spatial_msg_term_small_deg(i, t, nbrs)
                            tmp = spatial_msg_term.copy()
                            if t == 0:
                                tmp *= self.model._alpha
                            back_term = np.ones(self.Q)
                            if t < self.T - 1:
                                if self._pres_trans[i, t]:
                                    back_term = self.backward_temp_msg_term(i, t)
                                    # REMOVE CHECK AFTER FIX
                                    if np.isnan(back_term).sum() > 0:
                                        print("i,t:", i, t)
                                        print("back:", back_term)
                                        raise RuntimeError("Problem w back term")
                                    tmp *= back_term
                            forward_term = np.ones(self.Q)
                            if t > 0:
                                if self._pres_trans[i, t - 1]:
                                    # add back message to f_link
                                    f_link += tmp.sum()
                                    forward_term = self.forward_temp_msg_term(i, t)
                                else:
                                    # node present at t but not t-1, use alpha instead
                                    forward_term = self.model._alpha
                                tmp *= forward_term
                            tmp_spatial_msg = np.ones((deg_i, self.Q))
                            for nbr_idx in range(deg_i):
                                for q in range(self.Q):
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
                            if t < self.T - 1 and self._pres_trans[i, t]:
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
                            ) = self.spatial_msg_term_large_deg(i, t, nbrs)
                            tmp = spatial_msg_term
                            if t == 0:
                                tmp += np.log(self.model._alpha)
                            back_term = np.zeros(self.Q)
                            if t < self.T - 1:
                                if self._pres_trans[i, t]:
                                    back_term = np.log(
                                        self.backward_temp_msg_term(i, t)
                                    )
                                tmp += back_term
                            forward_term = np.zeros(self.Q)
                            if t > 0:
                                if self._pres_trans[i, t - 1]:
                                    tmp_backwards_msg = np.exp(
                                        tmp - max_log_spatial_msg_term
                                    )
                                    # add backwards msg to f_link
                                    f_link += tmp_backwards_msg.sum()

                                    forward_term = np.log(
                                        self.forward_temp_msg_term(i, t)
                                    )
                                else:
                                    # node present at t but not t-1, so use alpha instead
                                    forward_term = np.log(self.model._alpha)
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
                            if t < self.T - 1:
                                if self._pres_trans[i, t]:
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
        f_site /= self.N * self.T
        f_link /= self.N * self.T
        tmp_alpha = np.ones(self.Q)
        for q in range(self.Q):
            tmp_alpha[q] = self.model._alpha[q]
            for r in range(self.Q):
                last_term += tmp_alpha[q] * tmp_alpha[r] * self.model._beta[q, r, 0]
        for t in range(self.T - 1):
            tmp_alphaqm1 = tmp_alpha.copy()
            tmp_alpha = np.zeros(self.Q)
            for q in range(self.Q):
                for qprime in range(self.Q):
                    tmp_alpha[q] += self.model._pi[qprime, q] * tmp_alphaqm1[qprime]
            for q in range(self.Q):
                for r in range(self.Q):
                    last_term += (
                        tmp_alpha[q] * tmp_alpha[r] * self.model._beta[q, r, t + 1]
                    )
        last_term /= 2 * self.T
        return f_link - f_site - last_term

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
        time_update_order = np.arange(self.T)
        if RANDOM_ONLINE_UPDATE_MSG:
            np.random.shuffle(node_update_order)
            np.random.shuffle(time_update_order)
        for i in node_update_order:
            for t in time_update_order:
                if self._pres_nodes[i, t]:
                    nbrs = self.nbrs[t][i]
                    deg_i = len(nbrs)
                    if deg_i > 0:
                        if deg_i < LARGE_DEG_THR:
                            (
                                spatial_msg_term,
                                field_iter,
                            ) = self.spatial_msg_term_small_deg(i, t, nbrs)
                            self.update_h(i, t, -1.0)
                            # REMOVE CHECK AFTER FIX
                            if np.isnan(spatial_msg_term).sum() > 0:
                                print("i,t:", i, t)
                                print("spatial:", spatial_msg_term)
                                print("deg[i,t]", self.model.degs[i, t])
                                print("beta:", self.block_edge_prob[:, :, t])
                                raise RuntimeError("Problem w spatial term")
                            tmp = spatial_msg_term.copy()
                            if t == 0:
                                tmp *= self.model._alpha
                            back_term = np.ones(self.Q)
                            if t < self.T - 1:
                                if self._pres_trans[i, t]:
                                    back_term = self.backward_temp_msg_term(i, t)
                                    # REMOVE CHECK AFTER FIX
                                    if np.isnan(back_term).sum() > 0:
                                        print("i,t:", i, t)
                                        print("back:", back_term)
                                        raise RuntimeError("Problem w back term")
                                    tmp *= back_term
                            ## UPDATE BACKWARDS MESSAGES FROM i AT t ##
                            forward_term = np.ones(self.Q)
                            if t > 0:
                                if self._pres_trans[i, t - 1]:
                                    tmp_backwards_msg = tmp.copy()
                                    if tmp_backwards_msg.sum() > 0:
                                        tmp_backwards_msg /= tmp_backwards_msg.sum()
                                    for q in range(self.Q):
                                        if tmp_backwards_msg[q] < TOL:
                                            tmp_backwards_msg[q] = TOL
                                        # if tmp_backwards_msg[q] > 1 - TOL:
                                        #     tmp_backwards_msg[q] = 1 - TOL
                                    # tmp_backwards_msg[tmp_backwards_msg > 1 - TOL] = 1 - TOL
                                    tmp_backwards_msg /= tmp_backwards_msg.sum()
                                    self.msg_diff += (
                                        np.abs(
                                            tmp_backwards_msg
                                            - self._psi_t[i, t - 1, :, 0]
                                        ).mean()
                                        / self.n_msgs
                                    )
                                    self._psi_t[i, t - 1, :, 0] = tmp_backwards_msg
                                    forward_term = self.forward_temp_msg_term(i, t)
                                else:
                                    # node present at t but not t-1, use alpha instead
                                    forward_term = self.model._alpha
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
                            tmp_spatial_msg = np.ones((deg_i, self.Q))
                            for nbr_idx in range(deg_i):
                                for q in range(self.Q):
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
                                    tmp_spatial_msg[nbr_idx, :] /= tmp_spat_sums[
                                        nbr_idx
                                    ]
                            for nbr_idx in range(deg_i):
                                for q in range(self.Q):
                                    if tmp_spatial_msg[nbr_idx, q] < TOL:
                                        tmp_spatial_msg[nbr_idx, q] = TOL
                                    # if tmp_spatial_msg[nbr_idx, q] > 1 - TOL:
                                    #     tmp_spatial_msg[nbr_idx, q] = 1 - TOL
                            tmp_spatial_msg /= np.expand_dims(
                                tmp_spatial_msg.sum(axis=1), 1
                            )
                            self.msg_diff += (
                                np.abs(tmp_spatial_msg - self._psi_e[t][i]).mean()
                                * deg_i
                                / self.n_msgs
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
                            self._psi_e[t][i] = tmp_spatial_msg
                            ## UPDATE FORWARDS MESSAGES FROM i AT t ##
                            if t < self.T - 1 and self._pres_trans[i, t]:
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
                                self.msg_diff += (
                                    np.abs(
                                        tmp_forwards_msg - self._psi_t[i, t, :, 1]
                                    ).mean()
                                    / self.n_msgs
                                )
                                self._psi_t[i, t, :, 1] = tmp_forwards_msg
                            ## UPDATE MARGINAL OF i AT t ##
                            tmp_marg = tmp
                            if tmp_marg.sum() > 0:
                                tmp_marg /= tmp_marg.sum()
                            tmp_marg[tmp_marg < TOL] = TOL
                            tmp_marg = tmp_marg / tmp_marg.sum()
                            # tmp_marg[tmp_marg > 1 - TOL] = 1 - TOL
                            self.node_marg[i, t, :] = tmp_marg

                        else:
                            (
                                spatial_msg_term,
                                max_log_spatial_msg_term,
                                log_field_iter,
                            ) = self.spatial_msg_term_large_deg(i, t, nbrs)
                            self.update_h(i, t, -1.0)
                            tmp = spatial_msg_term
                            if t == 0:
                                tmp += np.log(self.model._alpha)
                            back_term = np.zeros(self.Q)
                            if t < self.T - 1:
                                if self._pres_trans[i, t]:
                                    back_term = np.log(
                                        self.backward_temp_msg_term(i, t)
                                    )
                                tmp += back_term
                            ## UPDATE BACKWARDS MESSAGES FROM i AT t ##
                            forward_term = np.zeros(self.Q)
                            if t > 0:
                                if self._pres_trans[i, t - 1]:
                                    tmp_backwards_msg = np.exp(
                                        tmp - max_log_spatial_msg_term
                                    )
                                    if tmp_backwards_msg.sum() > 0:
                                        tmp_backwards_msg /= tmp_backwards_msg.sum()
                                    tmp_backwards_msg[tmp_backwards_msg < TOL] = TOL
                                    # tmp_backwards_msg[tmp_backwards_msg > 1 - TOL] = TOL
                                    tmp_backwards_msg /= tmp_backwards_msg.sum()
                                    self._psi_t[i, t - 1, :, 0] = tmp_backwards_msg
                                    forward_term = np.log(
                                        self.forward_temp_msg_term(i, t)
                                    )
                                else:
                                    # node present at t but not t-1, so use alpha instead
                                    forward_term = np.log(self.model._alpha)
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
                                    tmp_spatial_msg[nbr_idx, :] /= tmp_spat_sums[
                                        nbr_idx
                                    ]
                            for nbr_idx in range(deg_i):
                                for q in range(self.Q):
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
                            self._psi_e[t][i] = tmp_spatial_msg
                            ## UPDATE FORWARDS MESSAGES FROM i AT t ##
                            if t < self.T - 1:
                                if self._pres_trans[i, t]:
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
                                    self._psi_t[i, t, :, 1] = tmp_forwards_msg
                            ## UPDATE MARGINAL OF i AT t ##
                            tmp_marg = np.exp(tmp - max_log_spatial_msg_term)
                            if tmp_marg.sum() > 0:
                                tmp_marg /= tmp_marg.sum()
                            tmp_marg[tmp_marg < TOL] = TOL
                            # tmp_marg[tmp_marg > 1 - TOL] = 1 - TOL
                            tmp_marg /= tmp_marg.sum()
                            self.node_marg[i, t, :] = tmp_marg
                        self.update_h(i, t, 1.0)
                    else:
                        # print("WARNING: disconnected nodes not yet handled properly")
                        # print("i,t:", i, t)
                        raise RuntimeError(
                            "Problem with measuring presence - deg = 0 but saying present"
                        )
                else:
                    self.node_marg[i, t, :] = 0.0

        if np.isnan(self.msg_diff):
            for i in range(self.N):
                for t in range(self.T - 1):
                    if np.isnan(self.node_marg[i, t]).sum() > 0:
                        print("nans for node marg @ (i,t)=", i, t)
                    if np.isnan(self._psi_e[t][i]).sum() > 0:
                        print("nans for spatial msgs @ (i,t)=", i, t)
                    if np.isnan(self._psi_t[i, t]).sum() > 0:
                        print("nans for temp marg @ (i,t)=", i, t)
                if np.isnan(self.node_marg[i, self.T - 1]).sum() > 0:
                    print("nans for node marg @ (i,t)=", i, self.T - 1)
                if np.isnan(self._psi_e[self.T - 1][i]).sum() > 0:
                    print("nans for spatial msgs @ (i,t)=", i, self.T - 1)
            raise RuntimeError("Problem updating messages")

    def update_twopoint_marginals(self):
        # TODO: ask Leto to check the two-point marginals eqns, as there's a chance they should include product of metaprob terms for i,j at t.
        # node_marg = None
        self.update_twopoint_spatial_marg()
        if self.verbose:
            print("\tUpdated twopoint spatial marg")
        self.update_twopoint_temp_marg()
        if self.verbose:
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
        for i, j, t, a_ijt in self._edge_vals:
            i, j, t, a_ijt = int(i), int(j), int(t), float(a_ijt)
            # print(i, j, t)
            j_idx = self.nbrs[t][i] == j
            # TODO: create inverse array that holds these values
            #       in memory rather than calc on the fly each time
            i_idx = self.nbrs[t][j] == i
            # tmp = np.outer(self._psi_e[t][i][j_idx, :], self._psi_e[t][j][i_idx, :])
            # if not self.directed:
            #     tmp += np.outer(
            #         self._psi_e[t][j][i_idx, :], self._psi_e[t][i][j_idx, :]
            #     )
            tmp = np.zeros((self.Q, self.Q))
            for q in range(self.Q):
                tmp[q, q] += (
                    self._psi_e[t][i][j_idx, q] * self._psi_e[t][j][i_idx, q]
                )[0]
                for r in range(q + 1, self.Q):
                    tmp[q, r] += (
                        self._psi_e[t][i][j_idx, q] * self._psi_e[t][j][i_idx, r]
                    )[0]
                    if not self.directed:
                        tmp[q, r] += (
                            self._psi_e[t][j][i_idx, q] * self._psi_e[t][i][j_idx, r]
                        )[0]
                        tmp[r, q] = tmp[q, r]
                if self.directed:
                    for r in range(q):
                        tmp[q, r] += (
                            self._psi_e[t][i][j_idx, q] * self._psi_e[t][j][i_idx, r]
                        )[0]
            if self.deg_corr:
                for q in range(self.Q):
                    tmp[q, q] *= self.model.compute_DC_lkl(i, j, t, q, q, a_ijt)
                    for r in range(q + 1, self.Q):
                        tmp[q, r] *= self.model.compute_DC_lkl(i, j, t, q, r, a_ijt)
                        if not self.directed:
                            tmp[r, q] = tmp[q, r]
                    if self.directed:
                        for r in range(q):
                            tmp[q, r] *= self.model.compute_DC_lkl(i, j, t, q, r, a_ijt)
            else:
                tmp *= self.block_edge_prob[:, :, t]
            if tmp.sum() > 0:
                tmp /= tmp.sum()
            for q in range(self.Q):
                for r in range(self.Q):
                    if tmp[q, r] < TOL:
                        tmp[q, r] = TOL
            tmp /= tmp.sum()
            self.twopoint_e_marg[t][i][j_idx, :, :] = tmp

    def update_twopoint_temp_marg(self):
        # recall t msgs in shape (i,t,q,2), w t from 0 to T-2, and final dim (backwards from t+1, forwards from t)
        for i in range(self.N):
            for t in range(self.T - 1):
                if self._pres_trans[i, t]:
                    tmp = np.zeros((self.Q, self.Q))
                    for q in range(self.Q):
                        for qprime in range(self.Q):
                            tmp[q, qprime] += self.trans_prob[q, qprime] * (
                                self._psi_t[i, t, q, 1]
                                * self._psi_t[i, t, qprime, 0]
                                # + self._psi_t[i, t, qprime, 1] * self._psi_t[i, t, q, 0]
                            )
                    # tmp = np.outer(
                    #     self._psi_t[i, t, :, 1], self._psi_t[i, t, :, 0]
                    # )
                    # TODO: check this
                    # tmp += np.outer(self._psi_t[i, t, :, 0], self._psi_t[i, t, :, 1])
                    # tmp *= self.trans_prob
                    if tmp.sum() > 0:
                        tmp /= tmp.sum()
                    for q in range(self.Q):
                        for qprime in range(self.Q):
                            if tmp[q, qprime] < TOL:
                                tmp[q, qprime] = TOL
                    tmp /= tmp.sum()
                    self.twopoint_t_marg[i, t, :, :] = tmp

    def zero_diff(self):
        self.msg_diff = 0.0


class BP:
    """Pure Python wrapper of BPBase class to allow optional/keyword arguments"""

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

    def update_messages(
        self,
    ):
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

    def store_messages(
        self,
    ):
        pass

    def learning_step(
        self,
    ):
        # this should fix normalisation of expected marginals so sum to one - might not due to learning rate. Unsure if necessary
        pass

    def block_edge_prob(self):
        if self.deg_corr:
            # will refer to as beta, but really omega or lambda (DC edge factor)
            return self.model.jit_model._lam
        else:
            # return beta_{qr}^t as Q x Q x T array
            return self.model.jit_model._beta

    def init_h(self):
        # update within each timestep is unchanged from static case,
        # i.e. = \sum_r \sum_i \psi_r^{it} p_{rq}^t
        self.jit_model.init_h()

    def update_h(self, i, t, old_marg):
        self.jit_model.update_h(i, t, old_marg)

    def convergence(self):
        pass

    def compute_free_energy(self):
        return self.jit_model.compute_free_energy()

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
