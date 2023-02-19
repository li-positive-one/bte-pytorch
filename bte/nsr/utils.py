import torch
import random
import numpy as np
from matplotlib import pyplot as plt

# def plot_solution(d):
#     fig=plt.figure()
#     plt.plot(solver.dis.density().cpu()[0,:,0],label='rho')
#     plt.plot(solver.dis.velocity().cpu()[0,:,0],label='ux')
#     plt.plot(solver.dis.velocity().cpu()[0,:,1],label='uy')
#     plt.plot(solver.dis.temperature().cpu()[0,:,0],label='T')
#     plt.legend()
#     plt.savefig("DVM-t0.png")
def fvmlinspace(vmin,vmax,nv):
    dv=(vmax-vmin)/nv
    return torch.linspace(vmin+dv/2,vmax-dv/2,nv)

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def orthonormalize(vectors):    
    """    
        Orthonormalizes the vectors using gram schmidt procedure.    
    
        Parameters:    
            vectors: torch tensor, size (dimension, n_vectors)    
                    they must be linearly independant    
        Returns:    
            orthonormalized_vectors: torch tensor, size (dimension, n_vectors)    
    """    
    assert (vectors.size(1) <= vectors.size(0)), 'number of vectors must be smaller or equal to the dimension'    
    orthonormalized_vectors = torch.zeros_like(vectors)    
    orthonormalized_vectors[:, 0] = vectors[:, 0] / torch.norm(vectors[:, 0], p=2)    
    
    for i in range(1, orthonormalized_vectors.size(1)):    
        vector = vectors[:, i]    
        V = orthonormalized_vectors[:, :i]    
        PV_vector= torch.mv(V, torch.mv(V.t(), vector))    
        orthonormalized_vectors[:, i] = (vector - PV_vector) / torch.norm(vector - PV_vector, p=2)    
    
    return orthonormalized_vectors