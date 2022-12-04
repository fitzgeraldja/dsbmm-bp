import csr
import numpy as np
import yaml  # type: ignore
from numba import float64, int32, int64, set_num_threads, typeof
from numba.experimental import jitclass
from numba.typed import List
from numba.types import Array, ListType, bool_, unicode_type

from . import numba_dsbmm_methods as parmeth
from .dsbmm_base import DSBMMTemplate

# from numba import float32

# from scipy import sparse
# from utils import nb_ib_lkl
# from utils import nb_poisson_lkl_int
# from utils import numba_ix

# import utils

# from sklearn.cluster import MiniBatchKMeans

with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
TOL = config["tol"]  # min value permitted for msgs etc (for numerical stability)
NON_INFORMATIVE_INIT = config[
    "non_informative_init"
]  # initialise alpha, pi as uniform (True), or according to init part passed (False)

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
nbrs_ex = List.empty_list(ListType(Array(int32, ndim=1, layout="C")))
nbrs_type = typeof(nbrs_ex)

sparse_A_ex = List()
sparse_A_ex.append(csr.create_empty(1, 1))
sparse_A_type = typeof(sparse_A_ex)
# this decorator ensures types of each base field, and means all methods are compiled into nopython fns
# further types are inferred from type annotations


@jitclass  # (base_spec)
class DSBMMSparseParallelBase:
    # A: np.ndarray  # assume N x N x T array s.t. (i,j,t)th position confers information about connection from i to j at time t
    A: sparse_A_type  # type: ignore
    _pres_nodes: bool_[:, ::1]
    _pres_trans: bool_[:, ::1]
    _tot_N_pres: int64
    X: X_type  # type: ignore
    # X: list
    # [
    #     np.ndarray
    # ]  # assume S x N x T x Ds array s.t. (s)(i,t,ds) posn is info about ds dim of sth type metadata of i at time t
    S: int
    Z: int32[:, ::1]
    _n_qt: int64[:, ::1]  # Q x T array holding num nodes in group q at time t
    T: int
    N: int
    E: int64[::1]  # num_edges in each timestep
    Q: int
    # meta_types: list[str]
    meta_types: meta_types_type  # type: ignore
    meta_dims: int32[::1]
    deg_corr: bool
    directed: bool
    use_meta: bool
    tuning_param: float64
    degs: float64[:, :, ::1]  # N x T x [in,out]
    kappa: float64[:, :, ::1]  # Q x T x [in,out]
    # deg_entropy: float
    _alpha: float64[::1]  # init group probs
    # group transition mat
    _pi: Array(float64, ndim=2, layout="C")  # type: ignore # noqa: F821
    _lam: float64[:, :, ::1]  # block pois params in DC case
    # block edge probs in binary NDC case
    _beta: Array(float64, ndim=3, layout="C")  # type: ignore # noqa: F821
    # _meta_params: list[np.ndarray]  # params for metadata dists
    _meta_params: meta_params_type  # type: ignore
    node_marg: float64[:, :, ::1]
    twopoint_time_marg: float64[:, :, :, ::1]  # assume N x T - 1 x Q x Q (i,t,q,qprime)
    # twopoint_edge_marg: float64[:, :, :, :, ::1]  # assume N x N x T x Q x Q (i,j,t,q,r)
    twopoint_edge_marg: tp_e_marg_type  # type: ignore # [t][i][j_idx,q,r] where j_idx is idx where nbrs[t][i]==j
    dc_lkl: float64[:, :, ::1]  # E x Q x Q array of DC lkl of edge belonging to q, r
    meta_lkl: float64[:, :, ::1]  # N x T x Q array of meta lkl term for i at t in q
    nbrs: nbrs_type  # type: ignore
    _edge_vals: float64[:, ::1]
    diff: float64
    verbose: bool
    frozen: bool

    def __init__(
        self,
        A,
        X,
        Z,
        Q,
        deg_corr,
        directed,
        use_meta,
        meta_types,
        tuning_param,
        verbose,
        frozen,
    ):
        self.A = A  # assuming list of T sparse adjacency matrices (N x N) (made in right format already by Python wrapper)
        self.N = A[0].nrows
        self.E = np.array([A_t.nnz for A_t in self.A])
        self.T = len(A)
        self._pres_nodes = np.zeros(
            (self.N, self.T), dtype=bool_
        )  # N x T boolean array w i,t True if i present in net at time t
        for t in range(self.T):
            for i in range(self.N):
                self._pres_nodes[self.A[t].row_cs(i), t] = True
                self._pres_nodes[self.A[t].transpose().row_cs(i), t] = True
        self._tot_N_pres = self._pres_nodes.sum()
        self._pres_trans = (
            self._pres_nodes[:, :-1] * self._pres_nodes[:, 1:]
        )  # returns N x T-1 array w i,t true if i
        # Make edge_vals E x 4 array w rows [i,j,t,val] where A[i,j,t]==val != 0
        # (i.e. just COO format - easier to translate prev loops over)
        self._edge_vals = np.zeros((np.array([A_t.nnz for A_t in self.A]).sum(), 4))
        pos = 0
        for t, A_t in enumerate(self.A):
            row_idx = A_t.transpose().colinds.astype(np.float64)
            col_idx = A_t.colinds.astype(np.float64)
            t_vals = np.ones(A_t.nnz) * t
            vals = A_t.values.astype(np.float64)
            tmp = np.vstack((row_idx, col_idx, t_vals, vals)).T
            self._edge_vals[pos : pos + A_t.nnz] = tmp
            pos += A_t.nnz

        self.Z = Z
        for i in range(self.N):
            for t in range(self.T):
                if not self._pres_nodes[i, t]:
                    self.Z[i, t] = -1
        self.Q = (
            Q if Q is not None else len(np.unique(self.Z))
        )  # will return 0 if both Q and Z None
        self.use_meta = use_meta
        if self.use_meta:
            self.meta_types = meta_types
            if X is not None:
                tmp = List()
                tmp2 = List()
                self.S = len(X)
                assert len(self.meta_types) == self.S
                self.meta_dims = np.array([X_s.shape[-1] for X_s in X], dtype=np.int32)
                for s, mt in enumerate(self.meta_types):
                    tmp.append(X[s])  # assume X[s] is N x T x Ds array
                    # s.t. (s)(i,t,ds) posn is info
                    # about ds dim of sth type metadata
                    # of i at time t
                    tmp2.append(np.zeros((self.Q, self.T, self.meta_dims[s])))
                self.X = tmp
                self._meta_params = tmp2
            else:
                self.use_meta = False
                try:
                    assert self.use_meta == use_meta
                except Exception:  # AssertionError:
                    print(
                        "!" * 10,
                        "WARNING: no metadata passed but use_meta=True,",
                        "assuming want to use network alone",
                        "!" * 10,
                    )
        assert self.A is not None
        if self.use_meta:
            assert self.X is not None
        self.deg_corr = deg_corr
        self.directed = directed
        self.tuning_param = tuning_param
        self.degs = self.compute_degs(self.A)
        # if not NON_INFORMATIVE_INIT: # for some reason can't make these conditional in numba
        self.kappa = self.compute_group_degs()
        self._n_qt = self.compute_group_counts()
        # else:
        #     self.kappa = np.empty((1, 1, 1), dtype=np.float64)
        #     self._n_qt = np.empty((1, 1, 1), dtype=np.float64)
        # self.deg_entropy = -(self.degs * np.log(self.degs)).sum()

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

        # nbrs of i at t (possibly none)

        self._alpha = np.zeros(self.Q)
        self._beta = np.zeros((self.Q, self.Q, self.T))
        self._lam = np.zeros((self.Q, self.Q, self.T))
        self._pi = np.zeros((self.Q, self.Q))

        tmp = List()
        for t in range(self.T):
            tmp2 = List()
            for i in range(self.N):
                tmp2.append(
                    np.zeros(
                        (len(self.nbrs[t][i]), self.Q, self.Q),
                        dtype=np.float64,
                    )
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
        self.frozen = frozen

    @property
    def num_nodes(self):
        return self.N

    @property
    def num_groups(self):
        return self.Q

    @property
    def num_timesteps(self):
        return self.T

    # @property
    # def get_deg_entropy(self):
    #     return self.deg_entropy

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

        return np.dstack((in_degs, out_degs)).astype(float64)

    def compute_group_degs(self):
        """Compute group in- and out-degrees for current node memberships"""
        kappa = np.zeros((self.Q, self.T, 2))
        for q in range(self.Q):
            for t in range(self.T):
                kappa[q, t, :] = self.degs[self.Z[:, t] == q, t, :].sum(axis=0)
        return kappa.astype(float64)

    def compute_log_likelihood(self):
        """Compute log likelihood of model for given memberships

            In DC case this corresponds to usual DSBMM with exception of each timelice now has log lkl
                \\sum_{q,r=1}^Q m_{qr} \\log\frac{m_{qr}}{\\kappa_q^{out}\\kappa_r^{in}},
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
        if not self.frozen:
            # first init of parameters given initial groups if init=True, else use provided marginals
            self._alpha, self.diff = parmeth.nb_update_alpha(
                init,
                learning_rate,
                self.N,
                self.T,
                self.Q,
                self.Z,
                self._tot_N_pres,
                self._alpha,
                self.node_marg,
                self.diff,
                self.verbose,
            )
            if self.verbose:
                print(self._alpha)
                print("\tUpdated alpha")
            self._pi, self.diff = parmeth.nb_update_pi(
                init,
                learning_rate,
                self.N,
                self.T,
                self.Q,
                self.Z,
                self._pres_trans,
                self._pi,
                self.twopoint_time_marg,
                self.diff,
                self.verbose,
            )
            if self.verbose:
                print(self._pi)
                print("\tUpdated pi")
            if self.deg_corr:
                self._lam, self.diff = parmeth.nb_update_lambda(
                    init,
                    learning_rate,
                    self.directed,
                    self._edge_vals,
                    self.nbrs,
                    self.degs,
                    self.kappa,
                    self.T,
                    self.Q,
                    self.Z,
                    self._lam,
                    self.node_marg,
                    self.twopoint_edge_marg,
                    self.diff,
                    self.verbose,
                )
                if self.verbose:
                    print(self._lam)
                    print("\tUpdated lambda")
            else:
                # NB only implemented for binary case
                self._beta, self.diff = parmeth.nb_update_beta(
                    init,
                    learning_rate,
                    self.directed,
                    self._edge_vals,
                    self._pres_nodes,
                    self.nbrs,
                    self.degs,
                    self._n_qt,
                    self.T,
                    self.Q,
                    self.Z,
                    self._beta,
                    self.node_marg,
                    self.twopoint_edge_marg,
                    self.diff,
                    self.verbose,
                )
                # NB only implemented for binary case
                if self.verbose:
                    print(self._beta.transpose(2, 0, 1))
                    print("\tUpdated beta")
                if not self.directed:
                    assert np.all(
                        np.abs(self._beta.transpose(1, 0, 2) - self._beta) < 1e-10
                    )
            self._meta_params, self.diff = parmeth.nb_update_meta_params(
                init,
                learning_rate,
                self.meta_types,
                self.N,
                self.T,
                self.Q,
                self.Z,
                self.X,
                self._meta_params,
                self._pres_nodes,
                self.node_marg,
                self.diff,
                self.verbose,
            )
            if self.verbose:
                # print(self.jit_model._meta_params)
                print("\tUpdated meta")
        if not self.frozen or init:
            self.meta_lkl = parmeth.nb_calc_meta_lkl(
                self.N,
                self.T,
                self.Q,
                self.meta_types,
                self._meta_params,
                self.X,
                self.tuning_param,
                self._pres_nodes,
                self.verbose,
            )
            if self.deg_corr:
                self.dc_lkl = parmeth.nb_compute_DC_lkl(
                    self._edge_vals, self.Q, self.degs, self._lam
                )

    def set_node_marg(self, values):
        self.node_marg = values

    def set_twopoint_time_marg(self, values):
        self.twopoint_time_marg = values

    def set_twopoint_edge_marg(self, values):
        self.twopoint_edge_marg = values

    def zero_diff(self):
        self.diff = 0.0

    def set_Z_by_MAP(self):
        for i in range(self.N):
            for t in range(self.T):
                if self._pres_nodes[i, t]:
                    self.Z[i, t] = np.argmax(self.node_marg[i, t, :])
                else:
                    self.Z[i, t] = -1

    def set_alpha(self, params: float64[:]):
        self._alpha = params

    def set_pi(self, params: float64[:, :]):
        self._pi = params

    def set_beta(self, params: float64[:, :, :]):
        self._beta = params

    def set_lambda(self, params: float64[:, :, :]):
        self._lam = params

    def set_meta_params(self, params: meta_params_type):  # type: ignore
        assert len(params) == len(self._meta_params)
        for s, param in enumerate(params):  # type: ignore
            self._meta_params[s] = params[s]  # type: ignore


class DSBMMSparseParallel(DSBMMTemplate):
    """Pure Python wrapper around DSBMMSparseParallelBase to allow optional/keyword arguments
    Note the class this wraps is itself now a wrapper for separately njit'd functions, so as to allow
    parallelisation
    """

    def __init__(
        self,
        # data=None,
        A=None,
        X=None,
        Z=None,
        Q=None,
        deg_corr=False,
        directed=False,
        use_meta=True,  # control use of metadata or not (for debug)
        meta_types=["poisson", "indep bernoulli"],
        tuning_param=1.0,
        verbose=False,
        n_threads=None,
        frozen=False,
    ):
        """Initialise the class

        Args:
            A (list[scipy.sparse.csr_matrix], optional): (Sparse) adjacency matrices at each timestep,
                                                         each shape N x N. Defaults to None.
            X (list[np.ndarray], optional): Metadata, each entry shape N x T x Ds. Defaults to None.
            Z (np.ndarray, optional): Initial clustering, shape N x T. Defaults to None.
            Q (int, optional): Specified number of groups. Defaults to None (will try and infer).
            deg_corr (bool, optional): Use degree-corrected version. Defaults to False.
            directed (bool, optional): Use directed version - will symmetrise otherwise. Defaults to False.
            use_meta (bool, optional): Use metadata. Defaults to True.
            tuning_param (float, optional): Tuning parameter (eff. relative weight of metadata to edges).
                                            Defaults to 1.0.
            verbose (bool, optional): Verbosity. Defaults to False.
            n_threads (_type_, optional): Number of threads. Defaults to None (use all available).
        """
        if n_threads is not None:
            set_num_threads(n_threads)

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
        self.tuning_param = tuning_param
        self.frozen = frozen
        self.deg_corr = deg_corr
        self.jit_model = DSBMMSparseParallelBase(
            A,
            X,
            Z,
            Q,
            deg_corr,
            directed,
            use_meta,
            meta_types,
            tuning_param,
            verbose,
            frozen,
        )
        self.directed = directed
        self.verbose = verbose
