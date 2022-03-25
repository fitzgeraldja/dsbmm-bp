# numba reimplementation of all methods for BP class that reasonably stand to gain from doing so (would allow parallelisation + GPU usage)
# - simply prepend method name with nb_
from numba import njit, prange


@njit(parallel=True)
def nb_update_messages():
    pass


@njit(parallel=True)
def nb_update_spatial_message():
    pass


@njit(parallel=True)
def nb_cavity_spatial_message():
    pass


@njit(parallel=True)
def nb_forward_temp_msg_term():
    pass


@njit(parallel=True)
def nb_backward_temp_msg_term():
    pass


@njit(parallel=True)
def nb_update_temporal_messages():
    pass


@njit(parallel=True)
def nb_update_forward_temporal_message():
    pass


@njit(parallel=True)
def nb_update_backward_temporal_message():
    pass


@njit(parallel=True)
def nb_spatial_message_term():
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
def nb_update_twopoint_marginals():
    pass
