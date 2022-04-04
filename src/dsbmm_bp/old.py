# def update_messages(self,):
#     # assume want to visit messages in (almost) random order - here only random over edge locations, then use
#     # i vals as choice of loc to update temporal messages
#     np.random.shuffle(self._edge_locs)
#     for i, j, t in self._edge_locs:
#         self.update_spatial_message(i, j, t)
#     print("\tUpdated spatial messages")
#     np.random.shuffle(self._trans_locs)
#     for i, t in self._trans_locs:
#         self.update_temporal_messages(i, t)
#     print("\tUpdated temporal messages")

# def update_spatial_message(self, i, j, t):
#     # TODO: remove, just calc messages while calc node marginals, as have all
#     # necessary info there
#     j_idx = self.nbrs[t][i] == j
#     # TODO: make separate large deg version using logs
#     # backhere - this is problem (cavity term going to zero for large deg)
#     cavity_term = self.cavity_spatial_message(i, j, t)

#     marg = self.meta_prob(i, t) * np.exp(-1.0 * self._h[:, t]) * cavity_term
#     # TODO: include h neg update here (remove old h)
#     try:
#         if self._pres_trans[i, t]:
#             # i pres at t and t+1, can include backward temp msg term
#             marg *= self.backward_temp_msg_term(i, t)
#     except:
#         # t must be T-1, so no t+1, that's fine - doing this way means don't have to check t val for every edge
#         try:
#             assert t == self.T - 1
#         except:
#             print("i,j,t:", i, j, t)
#             print("meta:", self.meta_prob(i, t))
#             print("external:", np.exp(-1.0 * self._h[:, t]))
#             print("cavity:", self.cavity_spatial_message(i, j, t))
#             print("pres t:", self._pres_trans[i, t])
#             print("backward:", self.backward_temp_msg_term(i, t))
#             raise RuntimeError("Problem updating spatial message, backward term")
#     try:
#         if self._pres_trans[i, t - 1]:
#             # i pres at t-1 and t, can include forward temp msg term
#             marg *= self.forward_temp_msg_term(i, t)
#     except:
#         # t must be 0, so no t-1, that's fine
#         try:
#             assert t == 0
#         except:
#             print("i,j,t:", i, j, t)
#             print("meta:", self.meta_prob(i, t))
#             print("external:", np.exp(-1.0 * self._h[:, t]))
#             print("cavity:", self.cavity_spatial_message(i, j, t))
#             print("pres t - 1:", self._pres_trans[i, t - 1])
#             print("forward:", self.forward_temp_msg_term(i, t))
#             raise RuntimeError("Problem updating spatial message, forward term")
#     marg[marg < TOL] = TOL
#     marg /= marg.sum()
#     self._psi_e[t][i][j_idx] = marg
#     # TODO: then include h pos update here (add new h)

# def cavity_spatial_message(self, i, j, t):
#     # sum_r(p_rq^t *
#     # self._psi_e[t][k][i_idx (=self.nbrs[t][k]==i)][r]
#     # for k in self.nbrs[t][i]!=j (= self.nbrs[t][i][j_idx]))
#     nbrs = np.array([k for k in self.nbrs[t][i] if k != j])
#     # sum_terms = np.array(
#     #     [
#     #         self.block_edge_prob[:, :, t].T
#     #         @ self._psi_e[t][k][self.nbrs[t][k] == i]
#     #         for k in nbrs
#     #     ]
#     # )  # |N_i| - 1 x Q
#     # return np.prod(sum_terms, axis=0)
#     msg = np.ones((self.Q,))
#     if len(nbrs) > 0:
#         beta = self.block_edge_prob[:, :, t]
#         for k in nbrs:
#             if len(self.nbrs[t][k] > 0):
#                 idx = self.nbrs[t][k] == i
#                 if idx.sum() == 0:
#                     print("Fault:", k, t)
#             else:
#                 print("Fault:", k, t)
#             ktoi_msgs = self._psi_e[t][k][idx, :].reshape(
#                 -1
#             )  # for whatever reason this stays 2d, so need to flatten first
#             # print("jtoi_msg:", jtoi_msgs.shape)
#             tmp = np.ascontiguousarray(beta.T) @ np.ascontiguousarray(ktoi_msgs)
#             # print("summed:", tmp.shape)
#             msg *= tmp
#     else:
#         print("Fault:", i, t)
#     return msg


# def update_temporal_messages(self, i, t):
#     # know that i present at t and t+1, so want to update forward from t,
#     # backward from t+1
#     self.update_backward_temporal_message(i, t)
#     self.update_forward_temporal_message(i, t)

# def update_forward_temporal_message(self, i, t):
#     # size N x T - 1 x Q, w itq entry corresponding to temporal message from i at t to
#     # i at t+1 (i.e. covering t=0 to T-2)
#     try:
#         nbrs = self.nbrs[t][i]
#         if len(nbrs) > 0:
#             if len(nbrs) < LARGE_DEG_THR:
#                 marg = (
#                     self.meta_prob(i, t)
#                     * self.forward_temp_msg_term(i, t)
#                     * np.exp(-1.0 * self._h[:, t])
#                     * self.spatial_msg_term_small_deg(i, t, nbrs)
#                 )
#                 marg[marg < TOL] = TOL
#                 marg /= marg.sum()
#                 self._psi_t[i, t, :, 1] = marg
#             else:
#                 spatial_msg_term, max_log_msg = self.spatial_msg_term_large_deg(
#                     i, t, nbrs
#                 )
#                 logmarg = np.log(self.meta_prob(i, t))
#                 logmarg += np.log(self.forward_temp_msg_term(i, t))
#                 logmarg -= self._h[:, t]
#                 logmarg += spatial_msg_term - max_log_msg
#                 marg = np.exp(logmarg)
#                 marg[marg < TOL] = TOL
#                 marg /= marg.sum()
#                 self._psi_t[i, t, :, 1] = marg
#         else:
#             print("Fault:", i, t)
#             raise RuntimeError("Problem with adj for given i")

#     except:
#         # at t=0, no forward message
#         try:
#             assert t == 0
#         except:
#             print("Meta:", self.meta_prob(i, t))
#             print("Backward:", self.forward_temp_msg_term(i, t))
#             print("External:", np.exp(-1.0 * self._h[:, t]))
#             # print("Spatial:", self.spatial_msg_term(i, t))
#             raise RuntimeError("Problem with updating forward msg term")

# def update_backward_temporal_message(self, i, t):
#     # size N x T - 1 x Q, w itq entry corresponding to temporal message from i at t+1
#     # to i at t (i.e. covering t=0 to T-2 again)
#     try:
#         nbrs = self.nbrs[t][i]
#         if len(nbrs) > 0:
#             if len(nbrs) < LARGE_DEG_THR:
#                 marg = (
#                     self.meta_prob(i, t + 1)
#                     * self.backward_temp_msg_term(i, t + 1)
#                     * np.exp(-1.0 * self._h[:, t + 1])
#                     * self.spatial_msg_term_small_deg(i, t + 1, nbrs)
#                 )
#                 marg[marg < TOL] = TOL
#                 marg /= marg.sum()
#                 self._psi_t[i, t, :, 0] = marg
#             else:
#                 spatial_msg_term, max_log_msg = self.spatial_msg_term_large_deg(
#                     i, t + 1, nbrs
#                 )
#                 logmarg = np.log(self.meta_prob(i, t + 1))
#                 logmarg += np.log(self.backward_temp_msg_term(i, t + 1))
#                 logmarg -= self._h[:, t + 1]
#                 logmarg += spatial_msg_term - max_log_msg
#                 marg = np.exp(logmarg)
#                 marg[marg < TOL] = TOL
#                 marg /= marg.sum()
#                 self._psi_t[i, t, :, 0] = marg
#         else:
#             print("Fault:", i, t)
#             raise RuntimeError("Problem w adj for i")
#     except:
#         # at t=T, no backward msg
#         try:
#             assert t == self.T - 1
#         except:
#             print("Meta:", self.meta_prob(i, t + 1))
#             print("Backward:", self.backward_temp_msg_term(i, t + 1))
#             print("External:", np.exp(-1.0 * self._h[:, t + 1]))
#             # print("Spatial:", self.spatial_msg_term(i, t + 1))
#             raise RuntimeError("Problem with updating backward msg term")

