# type: ignore
# Defines base class for BP
# TODO: convert other classes to inheritance
# NB numba classes DO NOT support inheritance, so can only apply to wrappers - full separate
# code needed for numba classes

from .dsbmm_base import DSBMMTemplate


class BPTemplate:
    """Pure Python wrapper of BP numba class to allow optional/keyword arguments"""

    def __init__(self, dsbmm: DSBMMTemplate):
        self.model = dsbmm
        self.jit_model = None  # BPTemplateBase(dsbmm.jit_model)
        # self.jit_model.get_neighbours()

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
    # @property
    # def meta_prob(self):
    #     return self.model.meta_lkl

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
