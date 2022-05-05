# numba reimplementation of all methods for BP class that reasonably stand to gain from doing so (would allow parallelisation + GPU usage)
# - simply prepend method name with nb_
from numba import njit


@njit(parallel=True)
def nb_init_msgs():
    pass


@njit(parallel=True)
def nb_forward_temp_msg_term():
    pass


@njit(parallel=True)
def nb_backward_temp_msg_term():
    pass


@njit(parallel=True)
def nb_spatial_msg_term_small_deg():
    pass


@njit(parallel=True)
def nb_spatial_msg_term_large_deg():
    pass


@njit(parallel=True)
def nb_init_h():
    pass


@njit(parallel=True)
def nb_update_h():
    pass


@njit(parallel=True)
def nb_update_node_marg():
    pass


@njit(parallel=True)
def nb_compute_free_energy():
    pass


@njit(parallel=True)
def nb_update_twopoint_marginals():
    pass


@njit(parallel=True)
def nb_update_twopoint_temp_marg():
    pass


@njit(parallel=True)
def nb_update_twopoint_spatial_marg():
    pass

