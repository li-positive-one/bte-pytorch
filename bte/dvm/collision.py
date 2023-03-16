from typing import Tuple
import torch
from torch.fft import fftn,ifftn,fftshift
import numpy as np
import torch.nn as nn
import math
from scipy import special
from bte.dvm.distribution import DVDis,DVDisMeta_Grid
from bte.dvm.ops import collision_fft,init_kernel_mode_vector

def get_collision(v_meta:DVDisMeta_Grid,quad_num:int=8,omega:float=0.81,M:int=5):
    u=v_meta.v_dims[0].v
    v=v_meta.v_dims[1].v
    w=v_meta.v_dims[2].v
    umax=u.max().item()
    umin=u.min().item()
    unum=u.numel()
    vmax=v.max().item()
    vmin=v.min().item()
    vnum=v.numel()
    wmax=w.max().item()
    wmin=w.min().item()
    wnum=w.numel()    
    return init_kernel_mode_vector(umax,umin,unum,vmax,vmin,vnum,wmax,wmin,wnum,quad_num,omega=omega,M=M)

def get_vshape(v_meta:DVDisMeta_Grid)->Tuple[int,int,int]:
    u=v_meta.v_dims[0].v
    v=v_meta.v_dims[1].v
    w=v_meta.v_dims[2].v
    unum=u.numel()
    vnum=v.numel()
    wnum=w.numel()  
    return unum,vnum,wnum

class collisioner(nn.Module):
    r"""Class with kernel initialized for doing binary collision
    """
    def __init__(self,v_meta,quad_num,omega=0.81,M=5,device="cpu",dtype=torch.float32):  
        super().__init__()
        phi, psi, phipsi = get_collision(v_meta,quad_num,omega,M)
        self.register_buffer("phi",phi.to(device=device,dtype=dtype))
        self.register_buffer("psi",psi.to(device=device,dtype=dtype))
        self.register_buffer("phipsi",phipsi.to(device=device,dtype=dtype))
        self.f_shape = get_vshape(v_meta)

    def do_collision(self, f:torch.Tensor, kn_bzm=1.0):
        f1=f.reshape(f.shape[:-1]+self.f_shape)
        if isinstance(kn_bzm,torch.Tensor):
            kn_bzm=kn_bzm.reshape_as(f1)
        Q=collision_fft(f1,kn_bzm,self.phi,self.psi,self.phipsi)
        Q=Q.reshape_as(f)
        return Q

class collisioner_bgk(nn.Module):
    """Class with kernel initialized for doing binary collision
    """

    def __init__(self,v_meta):
        super().__init__()
        self.v_meta=v_meta

    def do_collision(self, f:torch.Tensor, kn_bzm=1.0):
        F=DVDis(self.v_meta,f)
        return (F.maxwellian()-f)/kn_bzm