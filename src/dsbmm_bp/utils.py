from numba import njit, prange
import numpy as np


@njit(cache=True)
def numba_ix(arr, rows, cols):
    """
    Numba compatible implementation of arr[np.ix_(rows, cols)] for 2D arrays.
    :param arr: 2D array to be indexed
    :param rows: Row indices
    :param cols: Column indices
    :return: 2D array with the given rows and columns of the input array
    
    From https://github.com/numba/numba/issues/5894
    """
    one_d_index = np.zeros(len(rows) * len(cols), dtype=np.int32)
    for i, r in enumerate(rows):
        start = i * len(cols)
        one_d_index[start : start + len(cols)] = cols + arr.shape[1] * r

    arr_1d = arr.reshape((arr.shape[0] * arr.shape[1], 1))
    slice_1d = np.take(arr_1d, one_d_index)
    return slice_1d.reshape((len(rows), len(cols)))


@njit(cache=True)
def nb_poisson_lkl(k, mu):
    def _logpmf(self, k, mu):
        # log lkl is k*log(mu)-log(k!)-mu
        # can use that gamma(k)=(k-1)! for k +ve integer
        # and approx for log(gamma(k)) below
        Pk = k * np.log(mu) - gammaln_nb_p(k + 1) - mu
        return Pk

    # p(k) = exp(-mu)*mu^k/k!
    return np.exp(_logpmf(k, mu))


def nb_ib_lkl():
    pass


@njit(fastmath=True, error_model="numpy", parallel=True)
def gammaln_nb_p(z):
    """Numerical Recipes 6.1
    Code from https://stackoverflow.com/questions/55048299/why-is-this-log-gamma-numba-function-slower-than-scipy-for-large-arrays-but-fas"""
    # Don't use global variables.. (They only can be changed if you recompile the function)
    coefs = np.array(
        [
            57.1562356658629235,
            -59.5979603554754912,
            14.1360979747417471,
            -0.491913816097620199,
            0.339946499848118887e-4,
            0.465236289270485756e-4,
            -0.983744753048795646e-4,
            0.158088703224912494e-3,
            -0.210264441724104883e-3,
            0.217439618115212643e-3,
            -0.164318106536763890e-3,
            0.844182239838527433e-4,
            -0.261908384015814087e-4,
            0.368991826595316234e-5,
        ]
    )

    out = np.empty(z.shape[0])

    for i in prange(z.shape[0]):
        y = z[i]
        tmp = z[i] + 5.24218750000000000
        tmp = (z[i] + 0.5) * np.log(tmp) - tmp
        ser = 0.999999999999997092

        n = coefs.shape[0]
        for j in range(n):
            y = y + 1.0
            ser = ser + coefs[j] / y

        out[i] = tmp + np.log(2.5066282746310005 * ser / z[i])
    return out
