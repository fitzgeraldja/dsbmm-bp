import simulation
import data_processer
import base
import numpy as np
from numba.typed import List


if __name__ == "__main__":
    # Simulate data
    N = 200
    Q = 10
    T = 10
    Z_1 = np.random.randint(0, Q, size=(N,))
    Z_1 = np.sort(Z_1)
    meta_types = List()
    meta_types.append("poisson")
    meta_types.append("indep bernoulli")
    L = 5
    meta_params = [np.random.randint(5, 15, size=(1, Q, T)), np.random.rand(L, Q, T)]
    data = simulation.sample_dynsbm_meta(
        Z_1=Z_1, Q=Q, T=T, meta_types=meta_types, meta_params=meta_params
    )
    print("Successfully simulated data, now initialising model...")
    # Process
    pass  # not needed as simulating directly in form needed
    # Initialise model
    A = data["A"]
    X = data["X"]
    # X = [X["poisson"].transpose(1, 2, 0), X["indep bernoulli"].transpose(1, 2, 0)]
    X_poisson = X["poisson"].transpose(1, 2, 0)
    X_ib = X["indep bernoulli"].transpose(1, 2, 0)
    Z = data["Z"]
    dsbmm = base.DSBMMBase(
        A=A, X_poisson=X_poisson, X_ib=X_ib, Z=Z, Q=Q, meta_types=meta_types
    )  # X=X,
    print("Successfully instantiated DSBMM...")
    bp = base.BPBase(dsbmm)
    print("Successfully instantiated BP system...")
    # Apply model
