# TODO: Optionally reconvert to numpy for speed comparison
# - seems like likely faster for smaller nets
# numpy reimplementation of all methods for DSBMM class that reasonably stand to gain from doing so
# from numba import njit, prange
from typing import List

import numpy as np
import yaml  # type: ignore
from scipy import sparse
from scipy.special import gammaln

# from utils import nb_ib_lkl, nb_poisson_lkl_int

try:
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    TOL = config["tol"]  # min value permitted for msgs etc (for numerical stability)
    NON_INFORMATIVE_INIT = config[
        "non_informative_init"
    ]  # initialise alpha, pi as uniform (True), or according to init part passed (False)
except FileNotFoundError:
    TOL = 1e-6
    NON_INFORMATIVE_INIT = True


class NumpyDSBMM:
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
        meta_types=["poisson", "indep bernoulli", "categorical"],
        tuning_param=1.0,
        verbose=False,
        n_threads=None,
        frozen=False,
        alpha_use_all=True,
        non_informative_init=True,
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

        # assume A passed as list of scipy sparse matrices (CSR format)
        self.A = A
        self.tuning_param = tuning_param
        self.frozen = frozen
        self.deg_corr = deg_corr
        self.alpha_use_all = alpha_use_all
        self.non_informative_init = non_informative_init

        self.directed = directed
        self.verbose = verbose
        self.N = A[0].shape[0]
        self.E = np.array([A_t.nnz for A_t in self.A])
        self.T = len(A)
        self._edge_vals = {t: A_t.data for t, A_t in enumerate(self.A)}
        self._pres_nodes = np.array(
            np.vstack(
                [
                    np.logical_or(
                        self.A[t].sum(axis=0).squeeze() > 0,
                        self.A[t].sum(axis=1).squeeze() > 0,
                    )
                    for t in range(self.T)
                ]
            ).T
        )
        self._tot_N_pres = self._pres_nodes.sum()
        self._pres_trans = (
            self._pres_nodes[:, :-1] * self._pres_nodes[:, 1:]
        )  # returns N x T-1 array w i,t true if i present at t and t-1, for t = 1,...T-1

        self.Z = Z
        self.Z[~self._pres_nodes] = -1
        self.Q = (
            Q if Q is not None else len(np.unique(self.Z))
        )  # will return 0 if both Q and Z None
        self.use_meta = use_meta
        if self.use_meta:
            self.meta_types = meta_types
            if X is not None:
                self.S = len(X)
                assert len(self.meta_types) == self.S
                self.meta_dims = np.array([X_s.shape[-1] for X_s in X], dtype=np.int32)
                self.X = [np.ma.masked_invalid(Xs) for Xs in X]
                self._meta_params = [
                    np.zeros((self.Q, self.T, self.meta_dims[s])) for s in range(self.S)
                ]
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
        # if not self.non_informative_init: # for some reason can't make these conditional in numba
        self._kappa = self.compute_group_degs()
        self._n_qt = self.compute_group_counts()
        # else:
        #     self.kappa = np.empty((1, 1, 1), dtype=np.float64)
        #     self._n_qt = np.empty((1, 1, 1), dtype=np.float64)
        # self.deg_entropy = -(self.degs * np.log(self.degs)).sum()

        self._alpha = np.zeros(self.Q)
        self._beta = np.zeros((self.Q, self.Q, self.T))
        self._lam = np.zeros((self.Q, self.Q, self.T))
        self._pi = np.zeros((self.Q, self.Q))

        if self.Z is None:
            assert self.Q > 0

        self.diff = 0.0
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
        degs = np.zeros((self.N, self.T, 2))
        for t in range(self.T):
            degs[:, t, 0] = self.A[t].sum(axis=0).squeeze()  # in degs
            degs[:, t, 1] = self.A[t].sum(axis=1).squeeze()  # out degs

        return degs.astype(float)

    def compute_group_degs(self):
        """Compute group in- and out-degrees for current node memberships"""
        kappa = np.zeros((self.Q, self.T, 2))
        for q in range(self.Q):
            for t in range(self.T):
                kappa[q, t, :] = self.degs[self.Z[:, t] == q, t, :].sum(axis=0)
        return kappa.astype(float)

    def compute_log_likelihood(self):
        """Compute log likelihood of model for given memberships

            In DC case this corresponds to usual DSBMM with exception of each timelice now has log lkl
                \\sum_{q,r=1}^Q m_{qr} \\log\frac{m_{qr}}{\\kappa_q^{out}\\kappa_r^{in}},
            (ignoring constants w.r.t. node memberships)

        Returns:
            _type_: _description_
        """
        pass

    def update_params(self, init=False, learning_rate=0.2, planted_p=0.5):
        """Given marginals, update parameters suitably

        Args:
            messages (_type_): _description_
        """
        if init:
            self.planted_p = planted_p
        if not self.frozen:
            # first init of parameters given initial groups if init=True, else use provided marginals
            self.update_alpha(
                init=init, learning_rate=learning_rate, use_all_marg=self.alpha_use_all
            )
            if self.verbose:
                print(self._alpha)
                print("\tUpdated alpha")
            self.update_pi(init=init, learning_rate=learning_rate)
            if self.verbose:
                print(self._pi)
                print("\tUpdated pi")
            if self.deg_corr:
                self.update_lambda(init=init, learning_rate=learning_rate)
                if self.verbose:
                    print(self._lam)
                    print("\tUpdated lambda")
            else:
                # NB only implemented for binary case
                self.update_beta(init=init, learning_rate=learning_rate)
                # NB only implemented for binary case
                if self.verbose:
                    print(self._beta.transpose(2, 0, 1))
                    print("\tUpdated beta")
                if not self.directed:
                    assert np.all(
                        np.abs(self._beta.transpose(1, 0, 2) - self._beta) < 1e-10
                    )
            self.update_meta_params(init=init, learning_rate=learning_rate)
            if self.verbose:
                # print(self.jit_model._meta_params)
                print("\tUpdated meta")
        self.calc_log_meta_lkl()

    def set_node_marg(self, values):
        self.node_marg = values

    def set_twopoint_time_marg(self, values):
        self.twopoint_time_marg = values

    def set_twopoint_edge_marg(self, values):
        self.twopoint_edge_marg = {
            t: np.zeros((e, self.Q, self.Q)) for t, e in enumerate(self.E)
        }
        e_idxs = np.concatenate([[0], np.cumsum(self.E)]).astype(int)
        # convert from flat structure used in BP to sep by T for utility
        for t in range(self.T):
            self.twopoint_edge_marg[t] = values[e_idxs[t] : e_idxs[t + 1], :, :]

    def zero_diff(self):
        self.diff = 0.0

    def set_Z_by_MAP(self):
        self.Z = np.argmax(self.node_marg, axis=-1)
        self.Z[~self._pres_nodes] = -1

    def set_alpha(self, params: np.ndarray):
        self._alpha = params

    def set_pi(self, params: np.ndarray):
        self._pi = params

    def set_beta(self, params: np.ndarray):
        self._beta = params

    def set_lambda(self, params: np.ndarray):
        self._lam = params

    def set_meta_params(self, params: List[np.ndarray]):  # type: ignore
        assert len(params) == len(self._meta_params)
        for s, param in enumerate(params):  # type: ignore
            self._meta_params[s] = params[s]  # type: ignore

        self.nz_idxs = {t: {"i_idxs": None, "j_idxs": None} for t in range(self.T)}
        for t in range(self.T):
            i_idxs, j_idxs = np.argwhere(self.A[t]).T
            self.nz_idxs[t]["i_idxs"] = i_idxs
            self.nz_idxs[t]["j_idxs"] = j_idxs

    def log_meta_pois_lkl(self, k: np.ndarray, lam: np.ndarray):
        # k_it,lam_qt -> e^-lam_qt * lam_qt^k_it / k_it!
        # in log -> -lam_qt + k_it*log(lam_qt) - log(k_it!)
        # log(n!) = gammaln(n+1)
        return (
            -lam.T
            + np.einsum(
                "it,qt->itq",
                k,
                np.log(
                    lam,
                    where=lam > 0.0,
                    out=np.log(TOL) * np.ones_like(lam, dtype=float),
                    # out=np.zeros_like(lam, dtype=float),
                ),
            )
            - gammaln(k + 1)[:, :, np.newaxis]
        )

    def calc_log_meta_lkl(self):
        self.log_meta_lkl = np.zeros((self.N, self.T, self.Q))
        # NB meta_lkl will be nan when a node is missing
        for s, mt in enumerate(self.meta_types):
            # print(f"Updating params for {mt} dist")
            if mt == "poisson":
                # print("In Poisson")
                pois_params = self._meta_params[s]  # shape (Q x T x 1)
                # recall X[s] has shape (N x T x Ds), w Ds = 1 here
                # pois_lkl(k;lam) = e^-lam lam^k / k!
                self.log_meta_lkl += self.log_meta_pois_lkl(
                    self.X[s][:, :, 0], pois_params[:, :, 0]
                )
                if self.verbose:
                    print("\tUpdated Poisson lkl contribution")
            elif mt == "indep bernoulli":
                # print("In IB")
                ib_params = self._meta_params[s]  # shape (Q x T x L)
                # recall X[s] has shape (N x T x Ds), w Ds = L here
                self.log_meta_lkl += (
                    np.sum(
                        np.log(
                            ib_params,
                            where=ib_params > 0.0,
                            out=np.log(TOL) * np.ones_like(ib_params),
                            # out=np.zeros_like(ib_params),
                        )[np.newaxis, ...]
                        * self.X[s][:, np.newaxis, :, :],
                        axis=-1,
                    )
                    + np.sum(
                        np.log(
                            1 - ib_params,
                            where=1 - ib_params > 0.0,
                            # out=10 * np.log(TOL) * np.ones_like(ib_params),
                            out=np.log(TOL) * np.ones_like(ib_params),
                        )[np.newaxis, ...]
                        * (1 - self.X[s][:, np.newaxis, :, :]),
                        axis=-1,
                    )
                ).transpose(0, 2, 1)
                if self.verbose:
                    print("\tUpdated IB lkl contribution")
            elif mt == "categorical":
                cat_params = self._meta_params[s]
                log_cat_params = np.log(
                    cat_params,
                    where=cat_params > 0.0,
                    out=np.log(TOL) * np.ones_like(cat_params)
                    if not self.frozen
                    else -np.inf * np.ones_like(cat_params),
                    # out=np.zeros_like(cat_params),
                )[np.newaxis, ...]
                assert np.all(self.X[s].sum(axis=-1) == 1)  # REMOVE
                self.log_meta_lkl += np.sum(
                    np.multiply(
                        log_cat_params,
                        self.X[s][:, np.newaxis, :, :],
                        where=~np.isinf(log_cat_params)
                        | (self.X[s][:, np.newaxis, :, :] != 0),
                        out=np.zeros((self.N, self.Q, self.T, cat_params.shape[-1])),
                    ),
                    axis=-1,
                ).transpose(
                    0, 2, 1
                )  # should only have single nonzero X, one-hot encoding for category, so this should be sufficient
                if self.verbose:
                    print("\tUpdated categorical lkl contribution")
            elif mt == "multinomial":
                multi_params = self._meta_params[s]  # shape (Q,T,L)
                # TODO: stop recalculating xsums for multinomial each time
                xsums = self.X[s].sum(axis=-1, keepdims=True)
                self.log_meta_lkl += (
                    gammaln(xsums + 1)
                    - gammaln(self.X[s] + 1).sum(axis=-1, keepdims=True)
                    + np.sum(
                        self.X[s][:, :, np.newaxis, :]
                        * np.log(
                            multi_params,
                            where=multi_params > 0.0,
                            out=np.zeros_like(multi_params, dtype=float),
                        ).transpose(1, 0, 2)[np.newaxis, ...],
                        axis=-1,
                    )
                )
                if self.verbose:
                    print("\tUpdated multinomial lkl contribution")
            else:
                raise NotImplementedError(
                    "Yet to implement metadata distribution of given type \nOptions are 'poisson', 'indep bernoulli', 'categorical' or 'multinomial'."
                )
        # enforce all vals +ve prior to taking power
        # self.meta_lkl[self.meta_lkl < TOL] = TOL
        # self.meta_lkl[self.meta_lkl < 0.0] = 0.0
        self.log_meta_lkl = self.log_meta_lkl * self.tuning_param
        # self.meta_lkl[self.meta_lkl < TOL] = TOL
        # self.meta_lkl[self.meta_lkl > 1 - TOL] = 1 - TOL

    def update_alpha(self, init=False, learning_rate=0.2, use_all_marg=False):
        if init:
            if self.non_informative_init:
                self._alpha = np.ones(self.Q) / self.Q
            else:
                # case of no marginals / true partition provided to calculate most likely params
                planted_alpha = np.array(
                    [(self.Z == q).sum() / self._tot_N_pres for q in range(self.Q)]
                )
                if planted_alpha.sum() > 0:
                    planted_alpha /= planted_alpha.sum()
                self._alpha = (
                    self.planted_p * planted_alpha
                    + (1 - self.planted_p) * np.ones(self.Q) / self.Q
                )
                if self._alpha.sum() > 0:
                    self._alpha /= self._alpha.sum()
                # self._alpha[self._alpha < TOL] = TOL
                self._alpha /= self._alpha.sum()
                # _alpha[_alpha > 1 - TOL] = 1 - TOL
        else:
            # if DC, seems like should multiply marg by degree prior to sum - unseen for directed case but can calculate
            if use_all_marg:
                # in style of M&M, count contributions from all node marginals
                tmp = np.nansum(self.node_marg, axis=(0, 1))
                tmp /= self._tot_N_pres
            else:
                # only count contribs from nodes wout previous states, i.e. first timestep and nodes previously missing, as eqns suggest
                tmp = np.nansum(self.node_marg[:, 0, :], axis=0)
                tmp += np.nansum(
                    self.node_marg[:, 1:, :][~self._pres_trans], axis=(0, 1)
                )

            if tmp.sum() > 0:
                tmp /= tmp.sum()
            else:
                tmp = np.ones(self.Q) / self.Q
            # tmp[tmp < TOL] = TOL
            # tmp /= tmp.sum()
            # tmp[tmp > 1 - TOL] = 1 - TOL

            tmp = learning_rate * tmp + (1 - learning_rate) * self._alpha
            tmp_diff = np.abs(tmp - self._alpha).max()
            if np.isnan(tmp_diff):
                raise RuntimeError("Problem updating alpha")
            if self.verbose:
                print("Alpha diff:", np.round_(tmp_diff, 3))
            self.diff = max(tmp_diff, self.diff)
            self._alpha = tmp

    def update_pi(self, init=False, learning_rate=0.2):
        qqprime_trans = np.zeros((self.Q, self.Q))
        if init:
            if self.non_informative_init:
                qqprime_trans = np.ones((self.Q, self.Q))
            else:
                qqprime_trans = np.array(
                    [
                        [
                            [
                                ((self.Z[:, t] == q) * (self.Z[:, t + 1] == r)).sum()
                                for t in range(self.T - 1)
                            ]
                            for r in range(self.Q)
                        ]
                        for q in range(self.Q)
                    ],
                    dtype=float,
                )
                qqprime_trans /= qqprime_trans.sum(axis=1, keepdims=True)
                unif_trans = np.ones((self.Q, self.Q)) / self.Q
                qqprime_trans = (
                    self.planted_p * qqprime_trans + (1 - self.planted_p) * unif_trans
                )
        else:
            qqprime_trans = np.nansum(self.twopoint_time_marg, axis=(0, 1))
            # qqprime_trans /= np.expand_dims(
            #     node_marg[:, :-1, :].sum(axis=0).sum(axis=0), 1
            # )  # need to do sums twice as numba axis argument
            # # only takes integers (rather than axis=(0,1) as
            # # we would want) - can't use this as node_marg sums could
            # # be tiny / zero
            # below is unnecessary as can enforce normalisation directly
            # - just introduces instability
            # tot_marg = node_marg[:, :-1, :].sum(axis=0).sum(axis=0)
            # print("tot marg sums:", tot_marg)
            # for q in range(Q):
            #     if tot_marg[q] > TOL:
            #         qqprime_trans[q, :] = qqprime_trans[q, :] / tot_marg[q]
            #     else:
            #         raise RuntimeError("Problem with node marginals")
            # qqprime_trans[q, :] = TOL
            # correct_pi()
        qqprime_trans /= qqprime_trans.sum(axis=1, keepdims=True)
        # qqprime_trans[qqprime_trans < TOL] = TOL
        # qqprime_trans /= qqprime_trans.sum(axis=1)[:, np.newaxis]
        if not init:
            tmp = learning_rate * qqprime_trans + (1 - learning_rate) * self._pi
            tmp_diff = np.abs(tmp - self._pi).max()
            if np.isnan(tmp_diff):
                raise RuntimeError("Problem updating pi")
            if self.verbose:
                print("Pi diff:", np.round_(tmp_diff, 3))
            self.diff = max(tmp_diff, self.diff)
            self._pi = tmp
        else:
            self._pi = qqprime_trans

    def update_lambda(self, init=False, learning_rate=0.2):
        lam_num = np.zeros((self.Q, self.Q, self.T))
        lam_den = np.zeros((self.Q, self.Q, self.T))
        if init:
            # UNFINISHED - either need to pass full A (inefficient
            # unless this works well with sparse also), or idx between
            # nodes and edges so can find idxs where z_i=q etc
            if self.non_informative_init:
                lam_num = np.array(
                    [
                        [
                            [self.A[t].mean() for t in range(self.T)]
                            for r in range(self.Q)
                        ]
                        for q in range(self.Q)
                    ]
                )
                lam_den = np.ones((self.Q, self.Q, self.T))
            else:
                lam_num = np.array(
                    [
                        [
                            [
                                self.A[t][
                                    np.ix_(self.Z[:, t] == q, self.Z[:, t] == r)
                                ].sum()
                                for t in range(self.T)
                            ]
                            for r in range(self.Q)
                        ]
                        for q in range(self.Q)
                    ]
                )

                lam_den = np.einsum(
                    "qt,rt->qrt", self._kappa[:, :, 1], self._kappa[:, :, 0]
                )
                # lam_num[lam_num < TOL] = TOL
                # lam_den[lam_den < TOL] = 1.0
                if not self.directed:
                    lam_num = (lam_num + lam_num.transpose(1, 0, 2)) / 2
                    lam_den = (lam_den + lam_den.transpose(1, 0, 2)) / 2
                # lam_num[lam_num < TOL] = TOL
                # lam_den[lam_den < TOL] = 1.0
        else:
            lam_num = np.dstack(
                [
                    np.nansum(
                        self.twopoint_edge_marg[t]
                        * self._edge_vals[t][:, np.newaxis, np.newaxis],
                        axis=0,
                    )
                    for t in range(self.T)
                ]
            )
            # enforce uniformity for identifiability
            diag_vals = np.stack(
                [np.diag(lam_num[:, :, t]) for t in range(self.T)], axis=-1
            ).sum(axis=-1)
            [np.fill_diagonal(lam_num[:, :, t], diag_vals) for t in range(self.T)]
            if not self.directed:
                lam_num = (lam_num + lam_num.transpose(1, 0, 2)) / 2
            # lam_num[lam_num < TOL] = TOL
            # NB einsums fail for missing data
            # lam_den_out = np.einsum("itq,it->qt", self.node_marg, self.degs[:, :, 1])
            # lam_den_in = np.einsum("itq,it->qt", self.node_marg, self.degs[:, :, 0])
            lam_den_out = np.nansum(
                self.node_marg * self.degs[:, :, 1][..., np.newaxis], axis=0
            ).T
            lam_den_in = np.nansum(
                self.node_marg * self.degs[:, :, 0][..., np.newaxis], axis=0
            ).T
            lam_den = np.einsum("qt,rt->qrt", lam_den_out, lam_den_in)
            # enforce uniformity for identifiability
            diag_vals = np.stack(
                [np.diag(lam_den[:, :, t]) for t in range(self.T)], axis=-1
            ).sum(axis=-1)
            [np.fill_diagonal(lam_den[:, :, t], diag_vals) for t in range(self.T)]
            if not self.directed:
                lam_den = (lam_den + lam_den.transpose(1, 0, 2)) / 2
            # lam_num[lam_num < TOL] = TOL
            # lam_den[lam_den < TOL] = 1.0
        # NB use relative rather than absolute difference here as lam could be large
        tmp = np.divide(
            lam_num, lam_den, where=lam_den > 0, out=np.zeros_like(lam_num, dtype=float)
        )
        if not init:
            tmp = learning_rate * tmp + (1 - learning_rate) * self._lam
            tmp_diff = np.divide(
                np.abs(tmp - self._lam),
                self._lam,
                where=self._lam > 0.0,
                out=np.zeros_like(self._lam, dtype=float),
            ).max()
            if np.isnan(tmp_diff):
                raise RuntimeError("Problem updating lambda")
            if self.verbose:
                print("Lambda diff:", np.round_(tmp_diff, 3))
            self.diff = max(tmp_diff, self.diff)
            self._lam = tmp
        else:
            if not self.non_informative_init:
                unif_lam_num = np.array(
                    [
                        [
                            [self.A[t].mean() for t in range(self.T)]
                            for r in range(self.Q)
                        ]
                        for q in range(self.Q)
                    ]
                )
                tmp = self.planted_p * tmp + (1 - self.planted_p) * unif_lam_num
            self._lam = tmp

    def update_beta(self, init: bool = False, learning_rate=0.2):
        beta_num = np.zeros((self.Q, self.Q, self.T))
        beta_den = np.ones((self.Q, self.Q, self.T))
        if init:
            if self.non_informative_init:
                # assign as near uniform - just assume edges twice as likely in comms as out,
                # and that all groups have same average out-degree at each timestep
                Ns = self._pres_nodes.sum(axis=0)
                av_degs = self.degs.sum(axis=0)[:, 1] / Ns
                # beta_in = 2*beta_out
                # N*(beta_in + (Q - 1)*beta_out) = av_degs
                # = (Q + 1)*beta_out*N
                beta_out = av_degs / (Ns * (self.Q + 1))
                beta_in = 2 * beta_out
                beta_num = np.stack(
                    [
                        (beta_in[t] - beta_out[t]) * np.eye(self.Q)
                        + beta_out[t] * np.ones((self.Q, self.Q))
                        for t in range(self.T)
                    ],
                    axis=-1,
                )
            else:
                beta_num = np.array(
                    [
                        [
                            [
                                self.A[t][
                                    np.ix_(self.Z[:, t] == q, self.Z[:, t] == r)
                                ].sum()
                                for t in range(self.T)
                            ]
                            for r in range(self.Q)
                        ]
                        for q in range(self.Q)
                    ]
                )
                # enforce uniformity for identifiability
                diag_vals = np.stack(
                    [np.diag(beta_num[:, :, t]) for t in range(self.T)], axis=-1
                ).sum(axis=-1)
                [np.fill_diagonal(beta_num[:, :, t], diag_vals) for t in range(self.T)]
                beta_den = np.einsum("qt,rt->qrt", self._n_qt, self._n_qt)
                # beta_num[beta_num < TOL] = TOL
                # beta_den[beta_den < TOL] = 1.0
                if not self.directed:
                    beta_num = (beta_num + beta_num.transpose(1, 0, 2)) / 2
                    beta_den = (beta_den + beta_den.transpose(1, 0, 2)) / 2
                # beta_num[beta_num < TOL] = TOL
                # beta_den[beta_den < TOL] = 1.0
                # enforce uniformity for identifiability
                diag_vals = np.stack(
                    [np.diag(beta_den[:, :, t]) for t in range(self.T)], axis=-1
                ).sum(axis=-1)
                [np.fill_diagonal(beta_den[:, :, t], diag_vals) for t in range(self.T)]

        else:
            beta_num = np.dstack(
                [np.nansum(self.twopoint_edge_marg[t], axis=0) for t in range(self.T)]
            )
            # enforce uniformity for identifiability
            diag_vals = np.stack(
                [np.diag(beta_num[:, :, t]) for t in range(self.T)], axis=-1
            ).sum(axis=-1)
            [np.fill_diagonal(beta_num[:, :, t], diag_vals) for t in range(self.T)]
            if not self.directed:
                beta_num = (beta_num + beta_num.transpose(1, 0, 2)) / 2
            # beta_num[beta_num < TOL] = TOL
            beta_den = np.nansum(self.node_marg, axis=0).T
            beta_den = np.einsum("qt,rt->qrt", beta_den, beta_den)
            # beta_den[beta_den < TOL] = 1.0
            # enforce uniformity for identifiability
            diag_vals = np.stack(
                [np.diag(beta_den[:, :, t]) for t in range(self.T)], axis=-1
            ).sum(axis=-1)
            [np.fill_diagonal(beta_den[:, :, t], diag_vals) for t in range(self.T)]
            if not self.directed:
                beta_den = (beta_den + beta_den.transpose(1, 0, 2)) / 2

        # correct for numerical stability
        tmp = np.divide(
            beta_num,
            beta_den,
            where=beta_den > 0,
            out=np.zeros_like(beta_num, dtype=float),
        )
        # tmp[tmp < TOL] = TOL
        # tmp[tmp > 1 - TOL] = 1 - TOL

        if not init:
            tmp = learning_rate * tmp + (1 - learning_rate) * self._beta
            tmp_diff = np.abs(tmp - self._beta).max()
            if np.isnan(tmp_diff):
                raise RuntimeError("Problem updating beta")
            if self.verbose:
                print("Beta diff:", np.round_(tmp_diff, 3))
            self.diff = max(tmp_diff, self.diff)
            self._beta = tmp
        else:
            if not self.non_informative_init:
                Ns = self._pres_nodes.sum(axis=0)
                av_degs = self.degs.sum(axis=0)[:, 1] / Ns
                # beta_in = 2*beta_out
                # N*(beta_in + (Q - 1)*beta_out) = av_degs
                # = (Q + 1)*beta_out*N
                beta_out = av_degs / (Ns * (self.Q + 1))
                beta_in = 2 * beta_out
                beta_num = np.stack(
                    [
                        (beta_in[t] - beta_out[t]) * np.eye(self.Q)
                        + beta_out[t] * np.ones((self.Q, self.Q))
                        for t in range(self.T)
                    ],
                    axis=-1,
                )
                tmp = self.planted_p * tmp + (1 - self.planted_p) * beta_num
            self._beta = tmp

    def update_meta_params(self, init=False, learning_rate=0.2):
        # NB can't internally parallelise as need to aggregate
        # on diff, but would need to write entirely within this
        # fn to do so (less clear) - marginal benefit anyway
        # as typically S << N, T
        for s in range(len(self.meta_types)):
            # print(f"Updating params for {mt} dist")
            if self.meta_types[s] == "poisson":
                # print("In Poisson")
                self.update_poisson_meta(init, s, learning_rate=learning_rate)
                if self.verbose:
                    print("\tUpdated Poisson")
            elif self.meta_types[s] == "indep bernoulli":
                # print("In IB")
                self.update_indep_bern_meta(init, s, learning_rate=learning_rate)
                if self.verbose:
                    print("\tUpdated IB")
            elif self.meta_types[s] == "categorical":
                # print("In categorical")
                self.update_cat_meta(init, s, learning_rate=learning_rate)
                if self.verbose:
                    print("\tUpdated categorical")
            elif self.meta_types[s] == "multinomial":
                # print("In multinomial")
                self.update_multi_meta(init, s, learning_rate=learning_rate)
                if self.verbose:
                    print("\tUpdated multinomial")

            else:
                raise NotImplementedError(
                    "Yet to implement metadata distribution of given type \nOptions are 'poisson', 'indep bernoulli', 'categorical' or 'multinomial'"
                )  # NB can't use string formatting for print in numba

    def update_poisson_meta(self, init, s, learning_rate=0.2):
        xi = np.ones((self.Q, self.T, 1))
        zeta = np.zeros((self.Q, self.T, 1))
        if init:
            if self.non_informative_init:
                xi[:, :, 0] = np.ones((self.Q, self.T), dtype=float)
                zeta[:, :, :] = np.nanmean(self.X[s], axis=0, keepdims=True)
            else:
                xi[:, :, 0] = np.array(
                    [
                        [(self.Z[:, t] == q).sum() for t in range(self.T)]
                        for q in range(self.Q)
                    ]
                )
                zeta = np.array(
                    [
                        [
                            [
                                np.nansum(self.X[s][self.Z[:, t] == q, t, 0])
                                if not np.all(self.X[s][self.Z[:, t] == q, t, 0].mask)
                                else 0.0  # handle case of init group containing only missing nodes at t
                            ]
                            for t in range(self.T)
                        ]
                        for q in range(self.Q)
                    ]
                )
            # xi[xi < TOL] = 1.0
            # zeta[zeta < TOL] = TOL
        else:
            xi[:, :, 0] = np.nansum(self.node_marg, axis=0).T
            # zeta = np.einsum("itq,itd->qtd", self.node_marg, self.X[s]) # can't use einsum if X[s] contains nans, i.e. missing nodes
            zeta = np.nansum(
                self.node_marg[..., np.newaxis] * self.X[s][:, :, np.newaxis, :], axis=0
            ).data.transpose(1, 0, 2)
            # xi[xi < TOL] = 1.0
            # zeta[zeta < TOL] = TOL
        # NB again use relative error here as could be large
        tmp = np.divide(zeta, xi, where=xi > 0, out=np.zeros_like(zeta, dtype=float))
        # tmp[tmp < TOL] = TOL
        if not init:
            tmp = learning_rate * tmp + (1 - learning_rate) * self._meta_params[s]
            tmp_diff = np.divide(
                np.abs(tmp - self._meta_params[s]),
                self._meta_params[s],
                where=self._meta_params[s] > 0.0,
                out=np.zeros_like(self._meta_params[s], dtype=float),
            ).max()
            if np.isnan(tmp_diff):
                raise RuntimeError("Problem updating poisson params")
            if self.verbose:
                print("Poisson diff: ", np.round_(tmp_diff, 3))
            self.diff = max(tmp_diff, self.diff)
            self._meta_params[s] = tmp
        else:
            if np.isnan(tmp).sum() > 0:
                print("tmp pois params:")
                print(tmp)
                raise RuntimeError("Problem updating poisson params")
            if not self.non_informative_init:
                unif_zeta = np.nanmean(self.X[s], axis=0, keepdims=True)
                tmp = self.planted_p * tmp + (1 - self.planted_p) * unif_zeta
            self._meta_params[s] = tmp

    def update_indep_bern_meta(self, init, s, learning_rate=0.2):
        xi = np.ones((self.Q, self.T, 1))
        L = self.X[s].shape[-1]
        rho = np.zeros((self.Q, self.T, L))
        if init:
            if self.non_informative_init:
                xi = np.ones_like(xi, dtype=float)
                rho[:, :, :] = np.nanmean(self.X[s], axis=0, keepdims=True)
            else:
                xi[:, :, 0] = np.array(
                    [
                        [(self.Z[:, t] == q).sum() for t in range(self.T)]
                        for q in range(self.Q)
                    ]
                )
                rho = np.array(
                    [
                        [
                            np.nansum(self.X[s][self.Z[:, t] == q, t, :], axis=0)
                            if not np.all(self.X[s][self.Z[:, t] == q, t, :].mask)
                            else np.zeros(L)
                            for t in range(self.T)
                        ]
                        for q in range(self.Q)
                    ]
                )
            # xi[xi < TOL] = 1.0
            # rho[rho < TOL] = TOL
        else:
            xi[:, :, 0] = np.nansum(self.node_marg, axis=0).T
            # rho = np.einsum("itq,itl->qtl", self.node_marg, self.X[s]) # again can't use einsum w nans in X[s]
            rho = np.nansum(
                self.node_marg[..., np.newaxis] * self.X[s][:, :, np.newaxis, :], axis=0
            ).data.transpose(1, 0, 2)
            # xi[xi < TOL] = 1.0
            # rho[rho < TOL] = TOL
        tmp = np.divide(rho, xi, where=xi > 0, out=np.zeros_like(rho, dtype=float))
        # tmp[tmp < TOL] = TOL
        # tmp[tmp > 1 - TOL] = 1 - TOL
        if not init:
            tmp = learning_rate * tmp + (1 - learning_rate) * self._meta_params[s]
            tmp_diff = np.abs(tmp - self._meta_params[s]).max()
            if np.isnan(tmp_diff):
                raise RuntimeError("Problem updating IB params")
            if self.verbose:
                print("IB diff: ", np.round_(tmp_diff, 3))
            self.diff = max(tmp_diff, self.diff)
            self._meta_params[s] = tmp
        else:
            if not self.non_informative_init:
                unif_rho = np.nanmean(self.X[s], axis=0, keepdims=True)
                tmp = self.planted_p * tmp + (1 - self.planted_p) * unif_rho
            self._meta_params[s] = tmp

    def update_cat_meta(self, init, s, learning_rate=0.2):
        L = self.X[s].shape[-1]
        rho = np.zeros((self.Q, self.T, L))
        if init:
            if self.non_informative_init:
                rho[:, :, :] = np.nanmean(self.X[s], axis=0, keepdims=True)
            else:
                rho = np.array(
                    [
                        [
                            np.nansum(self.X[s][self.Z[:, t] == q, t, :], axis=0)
                            if not np.all(self.X[s][self.Z[:, t] == q, t, :].mask)
                            else np.zeros(L)
                            for t in range(self.T)
                        ]
                        for q in range(self.Q)
                    ]
                )
            # rho[rho < TOL] = TOL
        else:
            rho = np.nansum(
                self.node_marg[..., np.newaxis] * self.X[s][:, :, np.newaxis, :], axis=0
            ).data.transpose(1, 0, 2)
            # rho[rho < TOL] = TOL
        xi = rho.sum(axis=-1, keepdims=True)
        tmp = np.divide(rho, xi, where=xi > 0, out=np.zeros_like(rho, dtype=float))
        # tmp[tmp < TOL] = TOL
        # tmp /= tmp.sum(axis=-1, keepdims=True)
        if not init:
            tmp = learning_rate * tmp + (1 - learning_rate) * self._meta_params[s]
            tmp_diff = np.abs(tmp - self._meta_params[s]).max()
            if np.isnan(tmp_diff):
                raise RuntimeError("Problem updating cat params")
            if self.verbose:
                print("Cat diff: ", np.round_(tmp_diff, 3))
            self.diff = max(tmp_diff, self.diff)
            self._meta_params[s] = tmp
        else:
            if not self.non_informative_init:
                unif_rho = np.nanmean(self.X[s], axis=0, keepdims=True)
                tmp = self.planted_p * tmp + (1 - self.planted_p) * unif_rho
            self._meta_params[s] = tmp

    def update_multi_meta(self, init, s, learning_rate=0.2):
        xi = np.ones((self.Q, self.T, 1))
        L = self.X[s].shape[-1]
        rho = np.zeros((self.Q, self.T, L))
        xsums = self.X[s].sum(
            axis=-1, keepdims=True
        )  # shape (N,T,1), OK for not using nansum as should only have either all or no meta missing for i,t
        if init:
            if self.non_informative_init:
                xi[:, :, :] = np.nansum(xsums, axis=0, keepdims=True)
                rho[:, :, :] = np.nansum(self.X[s], axis=0, keepdims=True)
            else:
                xi[:, :, 0] = np.array(
                    [
                        [
                            np.nansum(xsums[self.Z[:, t] == q, t, :])
                            for t in range(self.T)
                        ]
                        for q in range(self.Q)
                    ]
                )
                rho = np.array(
                    [
                        [
                            np.nansum(self.X[s][self.Z[:, t] == q, t, :], axis=0)
                            if not np.all(self.X[s][self.Z[:, t] == q, t, :].mask)
                            else np.zeros(L)
                            for t in range(self.T)
                        ]
                        for q in range(self.Q)
                    ]
                )
            # xi[xi < TOL] = 1.0
            # rho[rho < TOL] = TOL
        else:
            xi = np.nansum(
                self.node_marg[..., np.newaxis] * xsums[:, :, np.newaxis, :], axis=0
            ).data.transpose(1, 0, 2)
            # rho = np.einsum("itq,itl->qtl", self.node_marg, self.X[s]) # again can't use einsum w nans in X[s]
            rho = np.nansum(
                self.node_marg[..., np.newaxis] * self.X[s][:, :, np.newaxis, :], axis=0
            ).data.transpose(1, 0, 2)
            # xi[xi < TOL] = 1.0
            # rho[rho < TOL] = TOL
        tmp = np.divide(rho, xi, where=xi > 0, out=np.zeros_like(rho, dtype=float))
        # tmp[tmp < TOL] = TOL
        # tmp[tmp > 1 - TOL] = 1 - TOL
        if not init:
            tmp = learning_rate * tmp + (1 - learning_rate) * self._meta_params[s]
            tmp_diff = np.abs(tmp - self._meta_params[s]).max()
            if np.isnan(tmp_diff):
                raise RuntimeError("Problem updating multi params")
            if self.verbose:
                print("Multi diff: ", np.round_(tmp_diff, 3))
            self.diff = max(tmp_diff, self.diff)
            self._meta_params[s] = tmp
        else:
            if not self.non_informative_init:
                unif_xi = np.nansum(xsums, axis=0, keepdims=True)
                unif_rho = np.zeros((self.Q, self.T, L))
                unif_rho[:, :, :] = np.nansum(self.X[s], axis=0, keepdims=True)
                tmp = self.planted_p * tmp + (1 - self.planted_p) * unif_rho
            self._meta_params[s] = tmp

    def set_params(self, true_params, freeze=True):
        self.frozen = freeze
        self._alpha = true_params["alpha"]
        self._beta = true_params.get("beta", None)
        self._lam = true_params.get("lam", None)
        self._pi = true_params["pi"]
        self._meta_params = true_params["meta_params"]
