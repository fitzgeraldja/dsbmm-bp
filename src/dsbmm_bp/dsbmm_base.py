from typing import List, Union

import csr
import numpy as np
from numba import set_num_threads
from numba.typed import List as nbList
from scipy.sparse import csr_matrix


class DSBMMTemplate:
    """Pure Python base class wrapper around jit models to allow optional/keyword arguments"""

    def __init__(
        self,
        # data=None,
        A: Union[np.ndarray, List[csr_matrix]] = None,
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
    ):
        """Initialise the class - need to reimplement for sparse / not sparse versions

        Args:
            A (list[scipy.sparse.csr_matrix], or np.ndarray, optional):
                                                        (Sparse) adjacency matrices at each timestep,
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
        raise NotImplementedError
        # if n_threads is not None:
        #     set_num_threads(n_threads)

        # # if data is not None:
        # #     self.A = data["A"]
        # #     self.X = data["X"]
        # #     self.Z = data.get("Z", None)
        # # else:
        # # assume A passed as list of scipy sparse matrices (CSR format)
        # tmp = nbList()
        # for A_t in A:
        #     tmp.append(csr.CSR.from_scipy(A_t))
        # A = tmp
        # self.A = A
        # self.tuning_param = tuning_param
        # self.jit_model = None  # DSBMMTemplateBase(
        # #     A,
        # #     X,
        # #     Z,
        # #     Q,
        # #     deg_corr,
        # #     directed,
        # #     use_meta,
        # #     meta_types,
        # #     tuning_param,
        # #     verbose,
        # # )
        # self.directed = directed
        # self.verbose = verbose

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
        """Compute group in- and out-degrees for current node memberships"""
        return self.jit_model.compute_group_degs()

    def compute_log_likelihood(self):
        """Compute log likelihood of model for given memberships

            In DC case this corresponds to usual DSBMM with exception of each timelice now has log lkl
                \\sum_{q,r=1}^Q m_{qr} \\log\frac{m_{qr}}{\\kappa_q^{out}\\kappa_r^{in}},
            (ignoring constants w.r.t. node memberships)

        Returns:
            _type_: _description_
        """
        return self.jit_model.compute_log_likelihood()

    def update_params(self, init=False, learning_rate=0.2):
        """Given marginals, update parameters suitably

        Args:
            init (bool, optional): _description_. Defaults to False.
            learning_rate (float, optional): _description_. Defaults to 0.2.
        """
        # first init of parameters given initial groups if init=True, else use provided marginals
        self.jit_model.update_params(init, learning_rate)

    def set_node_marg(self, values):
        self.jit_model.node_marg = values

    def set_twopoint_time_marg(self, values):
        self.jit_model.twopoint_time_marg = values

    def set_twopoint_edge_marg(self, values):
        self.jit_model.twopoint_edge_marg = values

    def update_alpha(self, init=False):
        self.jit_model.update_alpha(init)

    def update_pi(
        self,
        init=False,
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

    def set_params(self, params, freeze=True):
        if freeze:
            self.frozen = True
            self.jit_model.frozen = self.frozen
        try:
            self.jit_model.set_alpha(params["alpha"])
            self.jit_model.set_pi(params["pi"])
            if self.deg_corr:
                self.jit_model.set_lambda(params["lambda"])
            else:
                self.jit_model.set_beta(params["beta"])
            self.jit_model.set_meta_params(params["meta_params"])
        except KeyError:
            raise ValueError(
                "To set params, must pass dict with keys 'alpha', 'pi', 'beta' (NDC) or 'lambda' (DC), and 'meta_params'"
            )
