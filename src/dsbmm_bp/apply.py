import faulthandler
import simulation
import data_processer
import dsbmm
import bp
from utils import nb_nmi_local
import numpy as np
from numba.typed import List

faulthandler.enable()

# TODO: don't hardcode this
MAX_MSG_ITER = 5
MSG_CONV_TOL = 1e-4
MAX_ITER = 10
CONV_TOL = 1e-4

if __name__ == "__main__":
    ## Simulate data
    N = 500
    Q = 20
    T = 5
    p_in = 0.1
    p_out = 0.01
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
        p_in=p_in,
        p_out=p_out,
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
    X_poisson = np.ascontiguousarray(X["poisson"].transpose(1, 2, 0))
    X_ib = np.ascontiguousarray(X["indep bernoulli"].transpose(1, 2, 0))
    # print("X_pois:")
    # print(X_poisson.flags)
    # print("X_ib:")
    # print(X_ib.flags)
    Z = data["Z"]
    # print("Z:")
    # print(Z.flags)
    dsbmm_ex = dsbmm.DSBMM(
        A=A, X_poisson=X_poisson, X_ib=X_ib, Z=Z, Q=Q, meta_types=meta_types
    )  # X=X,
    print("Successfully instantiated DSBMM...")
    bp_ex = bp.BP(dsbmm_ex)
    print("Successfully instantiated BP system...")

    ## Apply model
    print("Now initialising model:")
    bp_ex.model.update_params(init=True)
    print("\tInitialised all DSBMM params!")
    bp_ex.init_messages()
    print("\tInitialised messages (random) and marginals (random)")
    bp_ex.init_h()
    print("\tInitialised corresponding external fields")
    print("Done, now running updates:")
    for n_iter in range(MAX_ITER):
        print(f"\n##### At iteration {n_iter+1} #####")
        for msg_iter in range(MAX_MSG_ITER):
            print(f"Message update iter {msg_iter + 1}...")
            bp_ex.update_node_marg()
            print("\tUpdated node marginals, messages and external fields")
            msg_diff = bp_ex.jit_model.msg_diff
            print(f"\tmsg differences: {msg_diff:.4f}")
            if msg_diff < MSG_CONV_TOL:
                break
            bp_ex.zero_diff()
        bp_ex.update_twopoint_marginals()
        print("Initialised corresponding twopoint marginals")
        bp_ex.model.set_node_marg(bp_ex.jit_model.node_marg)
        bp_ex.model.set_twopoint_edge_marg(bp_ex.jit_model.twopoint_e_marg)
        bp_ex.model.set_twopoint_time_marg(bp_ex.jit_model.twopoint_t_marg)
        print("\tPassed marginals to DSBMM")
        bp_ex.model.update_params()
        print("\tUpdated DSBMM params given marginals")
        diff = bp_ex.model.jit_model.diff
        print(f"\tSuccessfully completed update! Diff = {diff:.4f}")
        if diff < CONV_TOL:
            print("~~~~~~ CONVERGED ~~~~~~")
            break
        bp_ex.model.zero_diff()
    bp_ex.model.set_Z_by_MAP()
    # print("NMIs: ", nb_nmi_local(bp_ex.model.Z, Z))
    print("Overlaps:", (bp_ex.model.Z == Z).sum(axis=0) / Z.shape[0])
