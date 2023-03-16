import torch
import numpy as np


def fvmlinspace_t(vmin,vmax,nv):
    dv=(vmax-vmin)/nv
    return torch.linspace(vmin+dv/2,vmax-dv/2,nv)

def fvmlinspace(vmin,vmax,nv):
    dv=(vmax-vmin)/nv
    return np.linspace(vmin+dv/2,vmax-dv/2,nv)