import numpy as np
import math

def NDmsh(vmin,vmax,nv):
    """generate Nd mesh.

    Args:
        vmin (number or tuple of numbers): _description_
        vmax (number or tuple of numbers): _description_
        nv (int or tuple of ints): _description_
    """
    def vmsh(vmin,vmax,nv):
        assert vmax>vmin
        assert nv>0
        dv=(vmax-vmin)/nv
        v=np.linspace(vmin+dv/2,vmax-dv/2,nv)
        w=(vmax-vmin)/nv*np.ones_like(v)
        return v,w        
    if isinstance(vmin,(int,float)) and isinstance(vmax,(int,float)) and isinstance(nv,int):
        v,w=vmsh(vmin,vmax,nv)
        return v,w,(v,),(w,)
    else:
        assert len(vmin)==len(vmax)==len(nv), "vmin,vmax,nv must be the same length"
        vL,wL=list(zip(*[vmsh(vmini,vmaxi,nvi) for vmini,vmaxi,nvi in zip(vmin,vmax,nv)]))
        v=np.meshgrid(*vL,indexing='ij')
        v=np.stack([vi.flatten() for vi in v],axis=-1)
        w=np.meshgrid(*wL,indexing='ij')
        w=np.stack([wi.flatten() for wi in w],axis=-1)
        w=np.multiply.reduce(w,axis=-1)[...,None]
        return v,w,vL,wL

def maxwellianND(v:np.array, rho:np.array, u:np.array, T:np.array):
    """generate ND maxwellian VDF

    Args:
        v (np.array): [Nv,D] array
        rho (np.array): [N,1] array
        u (np.array): [N,D] array
        T (np.array): [N,1] array

    Returns:
        np.array: [N,Nv] array
    """
    return (rho / np.sqrt(2 * math.pi * T)**v.shape[-1]) * np.exp(-((u[..., None, :] - v) ** 2).sum(axis=-1) / (2 * T))

def get_vmsh(config):
    nv=config["vmesh"]["nv"]
    vmax=config["vmesh"]["vmax"]
    vmin=config["vmesh"]["vmin"]
    v,w,vL,wL=NDmsh(vmin,vmax,nv)
    return v,w,vL,wL
