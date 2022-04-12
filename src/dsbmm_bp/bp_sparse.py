# TODO: implement BP again but using sparse formulation so can fully take advantage of scaling

# Immediate way of doing so is to create an E x 4 matrix much as before, i.e. i,j,t, val rows
# edge list, as don't actually need any matrix operations (can implement directly - faster
# in numba and more clear anyway)
# But things to think about are
# - Calculating degrees
# - Storing lookup idxs (i.e. so when looking for j\to i for j a nbr of i, can find adj of j
# and know pos of i)
# The latter of these is more complex but could do via e.g. NT x max_deg len array, where then
# row i*t contains all edgelist indices corresponding to i at t, as then could immediately find
# adj of j - would then need to construct an inverse lookup matrix of same size where i*t row
# contains the position of i in the row corresponding to the node in that place in the original
# lookup mat - e.g. if orig[i*t,0] = j_idx s.t. edgelist[j_idx] = i, j, t, val, then want
# inv[i*t,0] = i_idx_for_j s.t. edgelist[i_idx_for_j] = j, i, t, val

# However, can also just take advantage of existing CSR implementations, along with again first
# identifying the index of i in row of j for faster lookup

# Actually the original implementation was already memory efficient for edge parameters - key
# change is just storage of A, and handling all uses of this within (should be doable)

# NB CSR implementation used only allows consideration of matrices w up to 2^31 ~ 2 B
# nonzero entries - this should be sufficient for most problems considered, and would likely
# already suggest batching (currently unimplemented)

# ADVANTAGES OF CSR:
# - Efficient arithmetic operations between matrices (addition / multiplication etc)
# - Efficient row slicing
# - Fast matrix vector products
# DISADVANTAGES OF CSR:
# - Slow column slicing operations (consider CSC instead - easy to obtain as transpose)
# - Changes to sparsity structure are expensive (suitable alts like LIL or DOK currently
#   unavailable in suitable numba form I believe) - this is not a concern herein as all
#   sparse matrices (i.e. just A) are left unchanged throughout the process

# Assumption is that only N is large, so N^2 terms dominate - not using sparse formulations
# for other params - possible problems could particularly emerge from next largest
# variables:
# - X (size S x N x T x Ds) which could be large if any metadata types are very large
#   (though unlikely in this case to be particularly sparse unless categorical, and
#   in this case if num categories ~ N it is unlikely to be hugely useful in this
#   framework anyway), or T is large (long time series - should further aggregate)
# - twopoint_edge_marg (size N x T x Q x Q)  which could be large if T and/or Q is large
#   (assumption is that Q ~ log(N) which holds in general for most real data)
# No immediately obvious way of sparsifying these, with the exception of X for categorical data
# with many categories (currently one-hot encoded (OHE)) - plan to deal with in future

# TODO: Allow X to be ordinal categorical encoding rather than OHE for memory efficiency
# TODO: think about how could allow batching

