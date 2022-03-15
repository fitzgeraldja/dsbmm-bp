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
)
from numba.types import unicode_type, ListType, bool_

from numba.typed import List, Dict
from numba.experimental import jitclass
import numpy as np
from utils import numba_ix

# from sklearn.cluster import MiniBatchKMeans

# base_spec = [
#     ("A", float32[:]),  # an array field
#     ("X", float32[:]),  # an array field
#     ("Z", int32[:]),  # an integer array field
# ]
X_ex = List.empty_list(float64[:, :, :])
X_type = typeof(X_ex)
meta_types_ex = List.empty_list(unicode_type)
meta_types_type = typeof(meta_types_ex)
meta_params_ex = List.empty_list(float64[:, :, :])
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
    deg_corr: bool
    degs: float64[:, :, :]  # N x T x [in,out]
    kappa: float64[:, :, :]  # Q x T x [in,out]
    edgemat: float64[:, :, :]
    deg_entropy: float
    _alpha: float64[:]  # init group probs
    _pi: float64[:, :]  # group transition mat
    _lam: float64[:, :, :]  # block pois params in DC case
    _beta: float64[:, :, :]  # block edge probs in binary NDC case
    # _meta_params: list[np.ndarray]  # params for metadata dists
    _meta_params: meta_params_type
    meta_lkl: float64[
        :, :
    ]  # TODO: impl, should be N x T array of meta lkl term for i at t

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
        self.A = A
        if X_poisson is not None and X_ib is not None:
            tmp = List()
            self.X_poisson = X_poisson
            self.X_ib = X_ib
            tmp.append(self.X_poisson)
            tmp.append(self.X_ib)
            self.X = tmp
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
        self.degs = self.compute_degs()
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

    def compute_degs(self, A=None):
        """Compute in-out degree matrix from given temporal adjacency mat

        Args:
            A (_type_): _description_

        Returns:
            _type_: _description_
        """
        if A is None:
            A = self.A
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

    def update_params(self, marginals=None):
        """Given marginals, update parameters suitably

        Args:
            messages (_type_): _description_
        """
        if marginals is None:
            # first init of parameters given initial groups
            self.update_alpha()
            self.update_pi()
            if self.deg_corr:
                self.update_lambda()
            else:
                # NB only implemented for binary case
                self.update_beta()
            self.update_meta_params()
        else:
            # marginals is list of arrays, [one-point, two-point spatial, two-point temporal]
            #                            = [N x T x Q, N x N x T x Q x Q, N x T - 1 x Q x Q]
            self.update_alpha(marginals)
            self.update_pi(marginals)
            if self.deg_corr:
                self.update_lambda(marginals)
            else:
                # NB only implemented for binary case
                self.update_beta(marginals)
            self.update_meta_params()

    def update_alpha(self, marginals=None):
        try:
            node_marg = marginals[0]
            self._alpha = node_marg[:, 0, :].mean(axis=0)
        except:
            self._alpha = np.array([(self.Z[:, 0] == q).mean() for q in range(self.Q)])

    def update_pi(self, marginals=None):
        try:
            node_marg, _, twopoint_time_marg = marginals
            self._pi = twopoint_time_marg.sum(axis=(0, 1)) / node_marg[:, :-1, :].sum(
                axis=(0, 1)
            )
            self.correct_pi()  # TODO: implement normalisation check
        except:
            qqprime_trans = np.array(
                [
                    [
                        [
                            (self.Z[:, t - 1] == q) * (self.Z[:, t] == qprime).sum()
                            for qprime in range(self.Q)
                        ]
                        for q in range(self.Q)
                    ]
                    for t in range(1, self.T)
                ]
            ).sum(axis=-1)
            qqprime_trans = qqprime_trans / qqprime_trans.sum(axis=-1)
            self._pi = qqprime_trans

    def update_lambda(self, marginals=None):
        try:
            node_marg, twopoint_edge_marg, _ = marginals
            lam_num = np.einsum("ijtqr,ijt->qrt", twopoint_edge_marg, self.A)
            lam_den = np.einsum("itq,it->qt", node_marg, self.degs)
            lam_den = np.einsum("qt,rt->qrt", lam_den, lam_den)
            self._lam = lam_num / lam_den
        except:
            lam_num = np.array(
                [
                    [
                        [
                            self.A[self.Z[:, t] == q, self.Z[:, t] == r, t].sum()
                            for r in range(self.Q)
                        ]
                        for q in range(self.Q)
                    ]
                    for t in range(self.T)
                ]
            )
            lam_den = np.array(
                [
                    [self.degs[self.Z[:, t] == q].sum() for q in range(self.Q)]
                    for t in range(self.T)
                ]
            )
            lam_den = np.einsum("tq,tr->tqr", lam_den, lam_den)
            self._lam = (lam_num / lam_den).transpose(1, 2, 0)

    def update_beta(self, marginals=None):
        try:
            node_marg, twopoint_edge_marg, _ = marginals
            beta_num = np.einsum("ijtqr,ijt->qrt", twopoint_edge_marg, (self.A > 0))
            beta_den = np.einsum("itq,it->qt", node_marg, self.degs)
            beta_den = np.einsum("qt,rt->qrt", beta_den, beta_den)
            self._beta = beta_num / beta_den
        except:
            beta_num = np.array(
                [
                    [
                        [
                            (self.A[self.Z[:, t] == q, self.Z[:, t] == r, t] > 0).sum()
                            for r in range(self.Q)
                        ]
                        for q in range(self.Q)
                    ]
                    for t in range(self.T)
                ]
            )
            beta_den = np.array(
                [
                    [self.degs[self.Z[:, t] == q].sum() for q in range(self.Q)]
                    for t in range(self.T)
                ]
            )
            beta_den = np.einsum("tq,tr->tqr", beta_den, beta_den)
            self._beta = (beta_num / beta_den).transpose(1, 2, 0)

    def update_meta_params(self, marginals=None):
        for s, mt in enumerate(self.meta_types):
            if mt == "poisson":
                self.update_poisson_meta(s, marginals=marginals)
            elif mt == "indep bernoulli":
                self.update_indep_bern_meta(s, marginals=marginals)
            else:
                raise NotImplementedError(
                    f"Yet to implement metadata distribution of type {mt} \nOptions are 'poisson' or 'indep bernoulli'"
                )

    def update_poisson_meta(self, s, marginals=None):
        try:
            node_marg, _, _ = marginals
            xi = node_marg.sum(axis=0).transpose(1, 0)
            zeta = np.einsum("itq,itd->qt", node_marg, self.X[s])
            self._meta_params[s] = zeta / xi
        except:
            xi = np.array(
                [
                    [(self.Z[:, t] == q).sum() for t in range(self.T)]
                    for q in range(self.Q)
                ]
            )
            zeta = np.array(
                [
                    [self.X[s][self.Z[:, t] == q, t, 0].sum() for t in range(self.T)]
                    for q in range(self.Q)
                ]
            )
            self._meta_params[s] = zeta / xi

    def update_indep_bern_meta(self, s, marginals=None):
        # TODO: handle correct normalisation
        try:
            node_marg, _, _ = marginals
            xi = node_marg.sum(axis=0).transpose(1, 0)
            rho = np.einsum("itq,itl->qtl", node_marg, self.X[s])
            self._meta_params[s] = rho / xi[:, :, np.newaxis]
        except:
            xi = np.array(
                [
                    [(self.Z[:, t] == q).sum() for t in range(self.T)]
                    for q in range(self.Q)
                ]
            )
            rho = np.array(
                [
                    [
                        self.X[s][self.Z[:, t] == q, t, :].sum(axis=0)
                        for t in range(self.T)
                    ]
                    for q in range(self.Q)
                ]
            )
            self._meta_params[s] = rho / xi[:, :, np.newaxis]


psi_e_ex = List.empty_list(ListType(float64[:, :]))
psi_e_type = typeof(psi_e_ex)
psi_t_ex = List.empty_list(ListType(float64[:, :]))
psi_t_type = typeof(psi_t_ex)
nbrs_ex = List.empty_list(ListType(int64[:]))
nbrs_type = typeof(nbrs_ex)
two_point_marg_ex = List.empty_list(float64[:, :, :])
two_point_marg_type = typeof(two_point_marg_ex)


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
    _psi_t: psi_t_type  # temporal messages
    _h: float64[:, :]  # external field for each group, with shape (Q,T)
    node_marg: float64[:, :, :]  # marginal group probabilities for each node
    two_point_marginals: two_point_marg_type
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
        self.N = dsbmm.N
        self.Q = dsbmm.Q
        self.deg_corr = dsbmm.deg_corr
        self.A = dsbmm.A
        self.X = dsbmm.X
        # start with given membership and corresponding messages, iterate until reach fixed point
        # given messages at fixed point, can update parameters to their most likely values - these emerge naturally
        # by requiring that the fixed point provides stationary free energy w.r.t. parameters
        self.get_neighbours()
        self.node_marg = np.zeros((self.N, self.T, self.Q))
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
        self.nbrs = [
            [self.A[i, :, t].nonzero()[0] for i in range(self.N)] for t in range(self.T)
        ]  # so self.nbrs[t][i] gives
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

    def init_messages(self, mode="random"):
        if mode == "random":
            # initialise by random messages
            self._psi_e = [
                [
                    np.random.rand(len(self.nbrs[t][i]), self.Q)
                    if len(self.nbrs[t][i]) > 0
                    else None
                    for i in range(self.N)
                ]
                for t in range(self.T)
            ]
            self._psi_e = [
                [
                    msg / msg.sum(axis=1) if msg is not None else None
                    for msg in self._psi_e[t]
                ]
                for t in range(self.T)
            ]
            self._psi_t = [
                [np.random.rand(2, self.Q) for i in range(self.N)]
                for t in range(self.T - 1)
            ]
            self._psi_t = [
                [msg / msg.sum(axis=1) for msg in self._psi_t[t]]
                for t in range(self.T - 1)
            ]
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
        np.random.shuffle(self._trans_locs)
        for i, t in self._trans_locs:
            self.update_temporal_messages(i, t)

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
        sum_terms = np.array(
            [
                self.block_edge_prob()[:, :, t].T
                @ self._psi_e[t][k][self.nbrs[t][k] == i]
                for k in nbrs
            ]
        )  # |N_i| - 1 x Q
        return np.prod(sum_terms, axis=0)

    def forward_temp_msg_term(self, i, t):
        # sum_r(self.trans_prob(r,q)*self._psi_t[t][i][1])
        # from t-1 to t
        try:
            prod = self.trans_prob.T @ self._psi_t[t - 1][i][1]
            return prod
        except:
            # must have t=0 so t-1 outside of range, no forward message
            return np.ones((self.Q,))

    def backward_temp_msg_term(self, i, t):
        # sum_r(self.trans_prob(q,r)*self._psi_t[t][i][0])
        # from t+1 to t
        try:
            prod = self.trans_prob @ self._psi_t[t][i][0]
            return prod
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
        marg = (
            self.meta_prob(i, t + 1)
            * self.backward_temp_msg_term(i, t + 1)
            * np.exp(self._h[:, t + 1])
            * self.spatial_msg_term(i, t + 1)
        )
        self._psi_t[t][i][0] = marg

    def spatial_msg_term(self, i, t):
        nbrs = self.nbrs[t][i]
        sum_terms = np.array(
            [
                self.block_edge_prob()[:, :, t].T
                @ self._psi_e[t][j][self.nbrs[t][j] == i]
                for j in nbrs
            ]
        )  # |N_i| x Q
        return np.prod(sum_terms, axis=0)

    def meta_prob(self, i, t):
        return self.model.meta_lkl[i, t]

    @property
    def trans_prob(self):
        # transition matrix
        return self.dsbmm._pi

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
        self._h = np.einsum("itr,rqt->qt", self.node_marg, self.block_edge_prob())

    def update_h(self, i, t, old_marg):
        self._h += self.block_edge_prob()[:, :, t].T @ (
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

    def update_node_marg(self, i, t):
        self.node_marg[i, t, :] = self._psi

    def calc_twopoint_marginals(self):
        # node_marg = None
        two_point_edge_marg = None
        two_point_time_marg = None
        self.two_point_marginals = [two_point_edge_marg, two_point_time_marg]
        return self.two_point_marginals

