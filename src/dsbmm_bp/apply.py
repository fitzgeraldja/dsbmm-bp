import faulthandler
import simulation
import data_processer
import dsbmm
import bp
import numpy as np
from numba.typed import List

faulthandler.enable()

if __name__ == "__main__":
    ## Simulate data
    N = 20
    Q = 3
    T = 5
    p_stay = 0.8
    trans_prob = simulation.gen_trans_mat(p_stay, Q)
    Z_1 = np.random.randint(0, Q, size=(N,))
    Z_1 = np.sort(Z_1)
    meta_types = List()
    meta_types.append("poisson")
    meta_types.append("indep bernoulli")
    L = 5
    meta_params = [np.random.randint(5, 15, size=(1, Q, T)), np.random.rand(L, Q, T)]
    data = simulation.sample_dynsbm_meta(
        Z_1=Z_1,
        Q=Q,
        T=T,
        trans_prob=trans_prob,
        meta_types=meta_types,
        meta_params=meta_params,
    )
    print("Successfully simulated data, now initialising model...")

    ## Process
    pass  # not needed as simulating directly in form needed

    ## Initialise model
    A = data["A"]
    try:
        assert np.allclose(A, A.transpose(1, 0, 2))
    except:
        # symmetrise for this test case
        A = ((A + A.transpose(1, 0, 2)) > 0) * 1.0
    # print("A:")
    # print(A.flags)
    X = data["X"]
    # X = [X["poisson"].transpose(1, 2, 0), X["indep bernoulli"].transpose(1, 2, 0)]
    X_poisson = X["poisson"].transpose(1, 2, 0)
    X_ib = X["indep bernoulli"].transpose(1, 2, 0)
    # print("X_pois:")
    # print(X_poisson.flags)
    # print("X_ib:")
    # print(X_ib.flags)
    Z = data["Z"]
    dsbmm_ex = dsbmm.DSBMM(
        A=A, X_poisson=X_poisson, X_ib=X_ib, Z=Z, Q=Q, meta_types=meta_types
    )  # X=X,
    print("Successfully instantiated DSBMM...")
    bp_ex = bp.BP(dsbmm_ex)
    print("Successfully instantiated BP system...")

    ## Apply model
    print("Now applying model:")
    bp_ex.model.update_params(init=True)
    print("\tInitialised all DSBMM params!")
    bp_ex.init_messages()
    print("\tInitialised messages (random)")
    bp_ex.update_node_marg()
    print("\tInitialised corresponding node marginals")
    bp_ex.update_twopoint_marginals()
    print("\tInitialised corresponding twopoint marginals")
    bp_ex.init_h()
    print("\tInitialised corresponding external fields")
    bp_ex.update_messages()
    print("\tSuccessfully completed first update!")
    bp_ex.model.set_node_marg(bp.node_marg)
    bp_ex.mode.set_two
