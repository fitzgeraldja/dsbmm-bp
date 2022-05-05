# numba reimplementation of all methods for DSBMM class that reasonably stand to gain from doing so
# - simply prepend method name with nb_

from numba import njit
import numpy as np

from utils import nb_poisson_lkl_int, nb_ib_lkl

import yaml

with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

TOL = config["tol"]


@njit(parallel=True)
def nb_calc_meta_lkl(
    N, T, Q, meta_types, _meta_params, tuning_param, _pres_nodes, verbose
):
    meta_lkl = np.ones((N, T, Q))
    for s, mt in enumerate(meta_types):
        # print(f"Updating params for {mt} dist")
        if mt == "poisson":
            # print("In Poisson")
            pois_params = _meta_params[s]  # shape (Q x T x 1)
            # recall X[s] has shape (N x T x Ds), w Ds = 1 here
            for t in range(T):
                for i in range(N):
                    if _pres_nodes[i, t]:
                        for q in range(Q):
                            # potentially could speed up further but
                            # loops still v efficient in numba
                            meta_lkl[i, t, q] *= nb_poisson_lkl_int(
                                X[s][i, t, 0], pois_params[q, t, 0]
                            )
            if verbose:
                print("\tUpdated Poisson lkl contribution")
        elif mt == "indep bernoulli":
            # print("In IB")
            ib_params = _meta_params[s]  # shape (Q x T x L)
            # recall X[s] has shape (N x T x Ds), w Ds = L here
            for t in range(T):
                for i in range(N):
                    if _pres_nodes[i, t]:
                        for q in range(Q):
                            meta_lkl[i, t, q] *= nb_ib_lkl(
                                X[s][i, t, :], ib_params[q, t, :]
                            )
            if verbose:
                print("\tUpdated IB lkl contribution")
        else:
            raise NotImplementedError(
                "Yet to implement metadata distribution of given type \nOptions are 'poisson' or 'indep bernoulli'"
            )  # NB can't use string formatting for print in numba
    for i in range(N):
        for t in range(T):
            if _pres_nodes[i, t]:
                for q in range(Q):
                    meta_lkl[i, t, q] = meta_lkl[i, t, q] ** tuning_param
                    if meta_lkl[i, t, q] < TOL:
                        meta_lkl[i, t, q] = TOL
                    elif meta_lkl[i, t, q] > 1 - TOL:
                        meta_lkl[i, t, q] = 1 - TOL
    return meta_lkl


@njit(parallel=True)
def nb_update_alpha():
    pass


@njit(parallel=True)
def nb_update_pi():
    pass


@njit(parallel=True)
def nb_update_lambda():
    pass


@njit(parallel=True)
def nb_update_beta():
    pass


@njit(parallel=True)
def nb_update_meta():
    pass


@njit(parallel=True)
def nb_update_poisson_meta():
    pass

@njit(parallel=True)
def nb_update_ib_meta():
    pass

