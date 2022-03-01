from numba import jit, njit, prange, int32, float32, float64
from numba.typed import List as NumbaList
from numba.experimental import jitclass
import numpy as np 
from utils import numba_ix

base_spec = [
    ('A',float32[:]), # an array field 
    ('X',float32[:]), # an array field 
    ('Z',int32[:]), # an integer array field 
]

# this decorator ensures types of each base field, and means all methods are compiled into nopython fns
# further types are inferred from type annotations
@jitclass(base_spec) 
class DSBMMBase:
    A: np.ndarray # assume N x N x T array s.t. (i,j,t)th position confers information about connection from i to j at time t
    X: np.ndarray 
    Z: np.ndarray  
    T: int 
    N: int
    E: int
    Q: int
    deg_corr: bool
    degs: np.ndarray
    kappa: np.ndarray
    edgemat: np.ndarray
    deg_entropy: float
    omega: np.ndarray # block edge probs
    
    def __init__(self, data: dict, Q=None, deg_corr=False):
        self.A = data['A']
        self.X = data['X']
        self.Z = data.get('Z',None) 
        self.N = self.A.shape[0]
        self.E = np.array(list(map(np.count_nonzero,self.A.transpose(2,0,1))))
        self.T = self.A.shape[-1]
        self.Q = Q if Q is not None else len(set(self.Z))
        self.deg_corr = deg_corr
        self.degs = self.compute_degs()
        self.kappa = self.compute_group_degs()
        self.edgemat = self.compute_block_edgemat()
        
        self.deg_entropy = -self.degs*np.log(self.degs).sum() 
        
    
    @property
    def num_nodes(self):
        return self.N    
    
    @property
    def num_groups(self):
        return self.Q 
    
    @property
    def num_timesteps(self):
        return self.T     
    
    @property
    def get_deg_entropy(self):
        return self.deg_entropy
    
    @property
    def num_edges(self): 
        # return number of edges in each slice - important as expect to affect BP computational complexity linearly
        return self.E
    
    def get_degree(self,i,t): 
        return self.degs[i,t,:] 
    
    def get_degree_vec(self,t): 
        return self.degs[:,t,:] 
    
    def get_groups(self,t): 
        return self.Z[:,t] 
    
    def get_entropy(self):
        pass
    
    def compute_degs(self,A=None):
        """Compute in-out degree matrix from given temporal adjacency mat

        Args:
            A (_type_): _description_

        Returns:
            _type_: _description_
        """
        if A is None:
            A = self.A
        return np.dstack((A.sum(axis=1),A.sum(axis=0)))
    
    def compute_group_degs(self):
        """Compute group in- and out-degrees for current node memberships
        """
        kappa = np.stack([
                np.array([self.degs[self.Z[:,t]==q,t,:].sum(axis=0) for t in range(self.T)])
                for q in range(self.Q)
            ])
        return kappa
        
    def compute_block_edgemat(self):
        """Compute number of edges between each pair of blocks

        Returns:
            _type_: _description_
        """
        
        edgemat = np.array([[[numba_ix(self.A[:,:,t],self.Z[:,t]==q,self.Z[:,t]==r).sum() for t in range(self.T)] 
                                  for r in range(self.Q)]
                                            for q in range(self.Q)])
            
        # numpy impl
        # self.edgemat = np.array([[[self.A[np.ix_(self.Z[:,t]==q,self.Z[:,t]==r),t].sum() for t in range(self.T)] 
        #                           for r in range(self.Q)]
        #                          for q in range(self.Q)])
        
        return edgemat
    
    
    def compute_log_likelihood(self): 
        """Compute log likelihood of model for given memberships 

            In DC case this corresponds to usual DSBMM with exception of each timelice now has log lkl
                \sum_{q,r=1}^Q m_{qr} \log\frac{m_{qr}}{\kappa_q^{out}\kappa_r^{in}},
            (ignoring constants w.r.t. node memberships) 
            
        Returns:
            _type_: _description_
        """
        pass
    
    def update_params(self,messages):
        """Given messages, update parameters suitably - not actually needed as all internal to BP class

        Args:
            messages (_type_): _description_
        """
        pass
    
        
    
    







@jitclass
class BPBase:
    N: int 
    Q: int
    beta: float # temperature 
    deg_corr: bool 
    A: np.ndarray 
    c_qrt: np.ndarray # N*p_qrt
    n_qt: np.ndarray # prior prob
    psi: np.ndarray # messages 
    h: np.ndarray # external field for each group
    nu: np.ndarray # marginal group probabilities for each node - NB this also referred to as eta or other terms elsewhere

    
    def __init__(self,data):
        # start with given membership and corresponding messages, iterate until reach fixed point
        # given messages at fixed point, can update parameters to their most likely values - these emerge naturally
        # by requiring that the fixed point provides stationary free energy w.r.t. parameters
        pass
    
    @property
    def n_timesteps(self):
        return self.T
    
    @property
    def beta(self): 
        # temperature for Boltzmann dist 
        # looks like this enters eqns via external field, i.e. e^-beta*h_q/N rather than e^-h_q/N
        # and also then in derivatives, hence possible need to have fns calculating vals of derivs  - need to be careful
        return self.beta
    
    @beta.setter
    def set_beta(self,value):
        self.beta = value
    
    def fit(self,data):
        pass 
    
    def init_messages(self,mode='random'):
        if mode=='random': 
            # initialise by random messages  
            pass 
        elif mode=='partial': 
            # initialise by partly planted partition - others left random 
            pass
        elif mode=='planted': 
            # initialise by given partition 
            pass
             
        pass
    
    def update_messages(self,):
        pass 
    
    def correct_messages(self): 
        # make sure messages sum to one over groups, and are nonzero for numerical stability
        pass
    
    def collect_messages(self): 
        pass 
    
    def store_messages(self,):
        pass 
    
    def learning_step(self,):
        # this should fix normalisation of expected marginals so sum to one - might not due to learning rate. Unsure if necessary
        pass  
    
    def init_h(self):
        pass
    
    def update_h(self,node_idx,):
        # double di = adj_list_ptr_->at(node_idx).size();

        # if (mode == -1) {
        #     for (unsigned int q1 = 0; q1 < Q_; ++q1) {
        #         for (unsigned int q2 = 0; q2 < Q_; ++q2) {
        #             if (deg_corr_flag_ == 0) {
        #                 h_[q1] -= cab_[q2][q1] * real_psi_[node_idx][q2];
        #             } else if (deg_corr_flag_ == 1 || deg_corr_flag_ == 2) {
        #                 h_[q1] -= di * cab_[q2][q1] * real_psi_[node_idx][q2];
        #             }
        #         }
        #     }
        # } else if (mode == +1) {
        #     for (unsigned int q1 = 0; q1 < Q_; ++q1) {
        #         for (unsigned int q2 = 0; q2 < Q_; ++q2) {
        #             if (deg_corr_flag_ == 0) {
        #                 h_[q1] += cab_[q2][q1] * real_psi_[node_idx][q2];
        #             } else if (deg_corr_flag_ == 1 || deg_corr_flag_ == 2) {
        #                 h_[q1] += di * cab_[q2][q1] * real_psi_[node_idx][q2];
        #             }
        #         }
        #     }
        # }
        pass
    
    def convergence(self):
        pass 
    
    def compute_free_energy(self):
        pass 
    
    def compute_entropy(self):
        pass 
    
    def compute_overlap(self):
        pass 
    
    def get_marginal_entropy(self):
        pass
    