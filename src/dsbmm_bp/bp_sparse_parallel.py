# Immediate way of making sparse is to create an E x 4 matrix much as before, i.e. i,j,t, val rows
# edge list, as don't actually need any matrix operations (can implement directly - faster
# in numba and more clear anyway)
# But things to think about are
# - Calculating degrees
# - Storing lookup idxs (i.e. so when looking for j\to i for j a nbr of i, can find adj of j
# and know pos of i)
# The latter of these is more complex but could do via e.g. NT x max_deg len array, where then
# row i*t contains all edgelist indices corresponding to i at t, as then could immediately find
# adj of j - would then need to construct an inverse lookup matrix of same size where i*t row
# contains the position of i in the row corresponding to the node in that place in the original
# lookup mat - e.g. if orig[i*t,0] = j_idx s.t. edgelist[j_idx] = i, j, t, val, then want
# inv[i*t,0] = i_idx_for_j s.t. edgelist[i_idx_for_j] = j, i, t, val

# However, can also just take advantage of existing CSR implementations, along with again first
# identifying the index of i in row of j for faster lookup

# Actually the original implementation was already memory efficient for edge parameters - key
# change is just storage of A, and handling all uses of this within (should be doable)

# NB CSR implementation used only allows consideration of matrices w up to 2^31 ~ 2 B
# nonzero entries - this should be sufficient for most problems considered, and would likely
# already suggest batching (currently unimplemented)

# ADVANTAGES OF CSR:
# - Efficient arithmetic operations between matrices (addition / multiplication etc)
# - Efficient row slicing
# - Fast matrix vector products
# DISADVANTAGES OF CSR:
# - Slow column slicing operations (consider CSC instead - easy to obtain as transpose)
# - Changes to sparsity structure are expensive (suitable alts like LIL or DOK currently
#   unavailable in suitable numba form I believe) - this is not a concern herein as all
#   sparse matrices (i.e. just A) are left unchanged throughout the process

# Assumption is that only N is large, so N^2 terms dominate - not using sparse formulations
# for other params - possible problems could particularly emerge from next largest
# variables:
# - X (size S x N x T x Ds) which could be large if any metadata types are very large
#   (though unlikely in this case to be particularly sparse unless categorical, and
#   in this case if num categories ~ N it is unlikely to be hugely useful in this
#   framework anyway), or T is large (long time series - should further aggregate)
# - twopoint_edge_marg (size N x T x Q x Q)  which could be large if T and/or Q is large
#   (assumption is that Q ~ log(N) which holds in general for most real data)
# No immediately obvious way of sparsifying these, with the exception of X for categorical data
# with many categories (currently one-hot encoded (OHE)) - plan to deal with in future


# TODO: Allow X to be ordinal categorical encoding rather than OHE for memory efficiency,
# TODO: think about how could allow batching.

from numba import (
    int32,
    float32,
    int64,
    float64,
    bool_,
    # unicode_type,
    typeof,
)
from numba.types import unicode_type, ListType, bool_, Array
from numba.typed import List
from numba.experimental import jitclass
import csr
import yaml

import numpy as np
from utils import numba_ix

from dsbmm_sparse_parallel import DSBMMSparseParallel, DSBMMSparseParallelBase
import numba_bp_methods as parmeth

X_ex = List.empty_list(float64[:, :, ::1])
X_type = typeof(X_ex)
sparse_A_ex = List()
sparse_A_ex.append(csr.create_empty(1, 1))
sparse_A_type = typeof(sparse_A_ex)
psi_e_ex = List.empty_list(ListType(Array(float64, ndim=2, layout="C")))
psi_e_type = typeof(psi_e_ex)
# psi_t_ex = List.empty_list(ListType(float64[:, :])) # unnecessary for same reasons as below
# psi_t_type = typeof(psi_t_ex)
nbrs_ex = List.empty_list(ListType(Array(int32, ndim=1, layout="C")))
nbrs_type = typeof(nbrs_ex)
twopoint_e_ex = List.empty_list(ListType(Array(float64, ndim=3, layout="C")))
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
class BPSparseParallelBase:
    N: int
    Q: int
    T: int
    _beta: float  # temperature
    deg_corr: bool
    directed: bool
    use_meta: bool
    model: DSBMMSparseParallelBase.class_type.instance_type
    A: sparse_A_type
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
    _edge_vals: float64[:, ::1]
    _trans_locs: int64[:, ::1]
    nbrs: nbrs_type
    e_nbrs: nbrs_type
    nbrs_inv: nbrs_type
    n_msgs: int64
    msg_diff: float64
    verbose: bool

    def __init__(self, dsbmm: DSBMMSparseParallelBase.class_type.instance_type):
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
    def boltz_temp(self):
        # TODO: add Boltzmann temp
        # temperature for Boltzmann dist
        # looks like this enters eqns via external field, i.e. for beta = boltz_temp,
        # e^-beta*h_q/N rather than e^-h_q/N
        # and also then in derivatives, hence possible need to have fns calculating vals of derivs
        # - need to be careful
        return self._boltz_temp

    @boltz_temp.setter
    def set_boltz_temp(self, value):
        self._boltz_temp = value

    def get_neighbours(self):
        self._pres_nodes = np.zeros(
            (self.N, self.T), dtype=bool_
        )  # N x T boolean array w i,t True if i present in net at time t
        for t in range(self.T):
            for i in range(self.N):
                self._pres_nodes[self.A[t].row_cs(i), t] = True
                self._pres_nodes[self.A[t].transpose().row_cs(i), t] = True
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
                # NB messages are bi-directional even in directed case, as purpose is to more accurately
                # reflect joint marginals when information propagates more easily
                tmp_out = self.A[t].row_cs(i)
                tmp_in = self.A[t].transpose().row_cs(i)
                if not self.directed:
                    all_conn = np.unique(np.concatenate((tmp_out, tmp_in)))
                else:
                    # NB can't call unique in this case as would remove reciprocity
                    all_conn = np.concatenate((tmp_out, tmp_in))
                tmp2.append(all_conn)
            tmp.append(tmp2)

        self.nbrs = tmp

        # # nbrs of i at t (possibly none)

        # nbrs_inv[t][i][j_idx] corresponds to i_idx s.t. nbrs[t][j][i_idx]==i,
        # for j = nbrs[t][i][j_idx], so can collect messages to i
        # as psi_e[t][j][nbrs_inv[t][i][j_idx]] = psi^{j\to i}(t)
        tmp = List()
        for t in range(self.T):
            tmp2 = List()
            for i in range(self.N):
                tmp2.append(np.zeros_like(self.nbrs[t][i]))
            tmp.append(tmp2)

        for t in range(self.T):
            for i in range(self.N):
                for j_idx, j in enumerate(self.nbrs[t][i]):
                    i_idx = np.nonzero(self.nbrs[t][j] == i)[0]
                    tmp[t][i][j_idx] = i_idx
        self.nbrs_inv = tmp

        # now make lookup of e_idx s.t. if
        # e_nbrs[t][i][j_idx] = e_idx,
        # then
        # _edge_vals[e_idx] = i,j,t,val
        # and e_nbrs_inv s.t. if
        # e_nbrs_inv[t][i][j_idx] = e_idx
        # then
        # _edge_vals[e_idx] = j,i,t,val
        tmp = List()
        tmpalt = List()
        for t in range(self.T):
            tmp2 = List()
            tmp2alt = List()
            for i in range(self.N):
                tmp2.append(np.zeros_like(self.nbrs[t][i]))
                tmp2alt.append(np.zeros_like(self.nbrs[t][i]))
            tmp.append(tmp2)
            tmpalt.append(tmp2alt)

        for e_idx, (i, j, t, _) in enumerate(self._edge_vals):
            j_idx = np.nonzero(self.nbrs[t][i] == j)[0]
            i_idx = np.nonzero(self.nbrs[t][j] == i)[0]
            tmp[t][i][j_idx] = e_idx
            tmpalt[t][j][i_idx] = e_idx

        self.e_nbrs = tmp
        self.e_nbrs_inv = tmpalt

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

    def init_messages(self, mode):
        """Initialise messages and node marginals according to specified mode 
        - random 
        - partial (specified extent planted vs random, currently unimplemented)
        - planted (according to initial partition + some noise)
        Args:
            mode (_type_): _description_
            
        Initialises _psi_e, _psi_t, node_marg
        """
        self._psi_e, self._psi_t, self.node_marg = parmeth.nb_init_msgs(
            mode, self.N, self.T, self.Q, self.nbrs, self.model.Z
        )

    def meta_prob(self, i, t):
        if self.use_meta:
            return self.model.meta_lkl[i, t, :]
        else:
            return np.ones(self.Q)

    @property
    def trans_prob(self):
        # transition matrix
        return self.model._pi

    @property
    def block_edge_prob(self):
        if self.deg_corr:
            # will refer to as beta, but really omega AKA lambda (DC edge factor)
            return self.model._lam
        else:
            # return beta_{qr}^t as Q x Q x T array
            return self.model._beta

    def init_h(self):
        """Initialise external fields, h(t) at each timestep  
        
        Initialises _h
        """
        self._h = parmeth.nb_init_h(
            self.N,
            self.T,
            self.Q,
            self.model.degs,
            self.deg_corr,
            self.block_edge_prob,
            self.node_marg,
        )

    def convergence(self):
        pass

    def compute_free_energy(self):
        f_energy = parmeth.nb_compute_free_energy(
            self.N,
            self.T,
            self.Q,
            self.deg_corr,
            self.model.degs,
            self._pres_nodes,
            self._pres_trans,
            self.nbrs,
            self.nbrs_inv,
            self.e_nbrs_inv,
            self.n_msgs,
            self.block_edge_prob,
            self.trans_prob,
            self.model.dc_lkl,
            self._h,
            self.meta_prob,
            self.model._alpha,
            self.model._beta,
            self.model._pi,
            self._psi_e,
            self._psi_t,
        )
        return f_energy

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
        (
            self.node_marg,
            self._psi_e,
            self._psi_t,
            self.msg_diff,
        ) = parmeth.nb_update_node_marg(
            self.N,
            self.T,
            self.Q,
            self.deg_corr,
            self.model.degs,
            self._pres_nodes,
            self._pres_trans,
            self.nbrs,
            self.nbrs_inv,
            self.e_nbrs_inv,
            self.n_msgs,
            self.block_edge_prob,
            self.trans_prob,
            self.model.dc_lkl,
            self._h,
            self.meta_prob,
            self.model._alpha,
            self.node_marg,
            self._psi_e,
            self._psi_t,
            self.msg_diff,
        )

    def update_twopoint_marginals(self):
        (
            self.twopoint_e_marg,
            self.twopoint_t_marg,
        ) = parmeth.nb_update_twopoint_marginals(
            self.N,
            self.T,
            self.Q,
            self._edge_vals,
            self._pres_trans,
            self.nbrs,
            self.nbrs_inv,
            self.directed,
            self.deg_corr,
            self.model.dc_lkl,
            self.block_edge_prob,
            self.trans_prob,
            self._psi_e,
            self._psi_t,
            self.twopoint_e_marg,
            self.twopoint_t_marg,
            self.verbose,
        )
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

    def zero_diff(self):
        self.msg_diff = 0.0


class BPSparseParallel:
    """Pure Python wrapper of BPBase class to allow optional/keyword arguments
    """

    def __init__(self, dsbmm: DSBMMSparseParallel):
        self.model = dsbmm
        self.jit_model = BPSparseParallelBase(dsbmm.jit_model)
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
