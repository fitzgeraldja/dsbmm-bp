import numpy as np
from numba import njit


@njit
def xlogx(x):
    if x == 0.0:
        return 0.0
    else:
        return x * np.log(x)


@njit
def pi_contrib(
    old_q: int, new_r: int, t: int, T: int, trans_counts: np.ndarray, q_tots: np.ndarray
):
    """Compute the change in likelihood from the move q->r of a node at time t

    :param old_q: old group
    :type old_q: int
    :param new_r: new group
    :type new_r: int
    :param t: timestep
    :type t: int
    :param T: Final timestep
    :type T: int
    :param trans_counts: (old) counts of transitions between groups
    :type trans_counts: np.ndarray, shape (Q,Q)
    :param q_tots: (old) total number of nodes present in each group
    :type q_tots: np.ndarray, shape (Q,)
    """
    count_term = 0.0
    Q = q_tots.shape[0]
    if t > 0:
        if q_tots[old_q] != 1:
            count_term += (trans_counts[old_q, old_q] - 1) * np.log(
                q_tots[old_q] - 1
            ) - trans_counts[old_q, old_q] * np.log(q_tots[old_q])

        if q_tots[new_r] != 0:
            count_term += (trans_counts[new_r, new_r] + 1) * np.log(
                q_tots[new_r] + 1
            ) - trans_counts[new_r, new_r] * np.log(q_tots[new_r])
    if t < T - 1:
        if q_tots[old_q] != 1:
            for qprime in range(Q):
                count_term += (trans_counts[old_q, qprime] - 1) * np.log(
                    q_tots[old_q] - 1
                ) - trans_counts[old_q, qprime] * np.log(q_tots[old_q])

        if q_tots[new_r] != 0:
            for qprime in range(Q):
                count_term += (trans_counts[new_r, qprime] + 1) * np.log(
                    q_tots[new_r] + 1
                ) - trans_counts[new_r, qprime] * np.log(q_tots[new_r])

    trans_term = 0.0
    if t > 0:
        for qprime in range(Q):
            trans_term += xlogx(trans_counts[qprime, old_q] - 1) - xlogx(
                trans_counts[qprime, old_q]
            )
            trans_term += xlogx(trans_counts[qprime, new_r] + 1) - xlogx(
                trans_counts[qprime, new_r]
            )
    if t < T - 1:
        for qprime in range(Q):
            trans_term += xlogx(trans_counts[old_q, qprime] - 1) - xlogx(
                trans_counts[old_q, qprime]
            )
            trans_term += xlogx(trans_counts[new_r, qprime] + 1) - xlogx(
                trans_counts[new_r, qprime]
            )
    return trans_term - count_term


@njit
def edge_contrib(
    old_q: int,
    new_r: int,
    m_t: np.ndarray,
    kappa_t: np.ndarray,
    k_it: np.ndarray,
    u_it: int,
):
    """Calculate DC likelihood change on change from q->r, as in Karrer & Newman (2011)

    :param old_q: old group
    :type old_q: int
    :param new_r: new group
    :type new_r: int
    :param m_t: (old) count of edges between blocks at timestep where change occurs
    :type m_t: np.ndarray, shape (Q,Q)
    :param kappa_t: (old) total degree of groups at timestep where change occurs
    :type kappa_t: np.ndarray, shape (Q,)
    :param k_it: (old) number of edges from i to nodes in group q at t
    :type k_it: np.ndarray, shape (Q,)
    :param u_it: number of self-edges of i at t
    :type u_it: int
    """
    res = 0.0
    k_i = k_it.sum()
    Q = kappa_t.shape[0]
    for s in range(Q):
        if s != old_q and s != new_r:
            res += 2 * (
                xlogx(m_t[old_q, s] - k_it[s])
                - xlogx(m_t[old_q, s])
                + xlogx(m_t[new_r, s] + k_it[s])
                - xlogx(m_t[new_r, s])
            )
    res += 2 * (
        xlogx(m_t[old_q, new_r] + k_it[old_q] - k_it[new_r])
        - xlogx(m_t[old_q, new_r])
        - xlogx(kappa_t[old_q] - k_i)
        + xlogx[kappa_t[old_q]]
        - xlogx(kappa_t[new_r] + k_i)
        + xlogx(kappa_t[new_r])
    )
    res += (
        xlogx(m_t[old_q, old_q] - 2 * (k_it[old_q] + u_it))
        - xlogx(m_t[old_q, old_q])
        + xlogx(m_t[new_r, new_r] + 2 * (k_it[new_r] + u_it))
        - xlogx(m_t[new_r, new_r])
    )
    return res


@njit
def poisson_contrib(
    old_q: int,
    new_r: int,
    X_t: np.ndarray,
    n_t: np.ndarray,
    x_it: float,
):
    """Contribution from Poisson metadata

    :param old_q: old group
    :type old_q: int
    :param new_r: new group
    :type new_r: int
    :param X_t: sum of metadata assigned to groups at t
    :type X_t: np.ndarray, shape (Q,)
    :param n_t: total number of nodes present in each group at t
    :type n_t: np.ndarray, shape (Q,)
    :param x_it: value of metadata for node changing groups
    :type x_it: float
    """
    res = 0.0
    res += (
        xlogx(X_t[old_q] - x_it)
        - xlogx(X_t[old_q])
        + xlogx(X_t[new_r] + x_it)
        - xlogx(X_t[new_r])
    )
    if n_t[old_q] != 1:
        res += X_t[old_q] * (
            np.log(n_t[old_q]) - np.log(n_t[old_q] - 1)
        ) + x_it * np.log(n_t[old_q] - 1)
    if n_t[new_r] != 0:
        res += X_t[new_r] * (
            np.log(n_t[new_r]) - np.log(n_t[new_r] + 1)
        ) - x_it * np.log(n_t[new_r] + 1)
    return res


@njit
def indep_bern_contrib(
    old_q: int,
    new_r: int,
    X_t: np.ndarray,
    n_t: np.ndarray,
    x_it: np.ndarray,
):
    """Contribution from independent Bernoulli or categorical metadata

    :param old_q: old group
    :type old_q: int
    :param new_r: new group
    :type new_r: int
    :param X_t: sum of metadata within each category, assigned to groups at t
    :type X_t: np.ndarray, shape (Q,L)
    :param n_t: total number of nodes present in each group at t
    :type n_t: np.ndarray, shape (Q,)
    :param x_it: value of metadata for node changing groups for each category
    :type x_it: np.ndarray, shape (L,)
    """
    res = 0.0
    for l in range(x_it.shape[0]):
        # TODO: make greedy IB contrib log safe
        res += (
            (n_t[new_r] + 1) * np.log(1 - (X_t[new_r, l] + x_it[l]) / (n_t[new_r] + 1))
            + (X_t[new_r, l] + x_it[l])
            * np.log(
                (X_t[new_r, l] + x_it[l]) / (n_t[new_r] + 1 - X_t[new_r, l] - x_it[l])
            )
            + (n_t[old_q] - 1)
            * np.log(1 - (X_t[old_q, l] - x_it[l]) / (n_t[old_q] - 1))
            + (X_t[old_q, l] - x_it[l])
            * np.log(
                (X_t[old_q, l] - x_it[l]) / (n_t[old_q] - 1 - X_t[old_q, l] + x_it[l])
            )
        )
        res -= (
            n_t[new_r] * np.log(1 - X_t[new_r, l] / n_t[new_r])
            + n_t[old_q] * np.log(1 - X_t[old_q, l] / n_t[old_q])
            + X_t[new_r, l] * np.log(X_t[new_r, l] / (n_t[new_r] - X_t[new_r, l]))
            + X_t[old_q, l] * np.log(X_t[old_q, l] / (n_t[old_q] - X_t[old_q, l]))
        )
    return res


# TODO: derive greedy contribution for multinomial metadata
