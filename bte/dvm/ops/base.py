import math
import torch


def maxwellian(v: torch.Tensor, rho: torch.Tensor, u: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """generate the maxwellian distribution for given rho,u,T

    Args:
        v (torch.Tensor): shape [Nv,D]
        rho (torch.Tensor):  shape [...,1]
        u (torch.Tensor):  shape [...,D]
        T (torch.Tensor):  shape [...,1]
    Returns:
        torch.Tensor: shape [...,Nv]
    """
    return (rho / torch.sqrt(2 * math.pi * T)**v.shape[-1]) * torch.exp(-((u[..., None, :] - v) ** 2).sum(dim=-1) / (2 * T))

def fsum(f,w):
    return f @ w

def _m0(f,v,w):
    return fsum(f,w)

def _m1(f,v,w):
    return torch.cat(
        [fsum(f * v[..., i],w) for i in range(v.shape[-1])], dim=-1
    )
def _m2(f,v,w):
    return torch.cat(
        [fsum(f * (v**2)[..., i],w) for i in range(v.shape[-1])], dim=-1
    )

eps=1e-6
def rho_u_theta(f,v,w):
    m0,m1,m2=_m0(f,v,w),_m1(f,v,w),_m2(f,v,w)
    density = torch.maximum(m0,eps*torch.ones_like(m0))
    veloc = m1/m0
    v2 = (veloc ** 2).sum(dim=-1, keepdim=True)
    temperature = torch.maximum((m2.sum(dim=-1, keepdim=True)/m0-v2),eps*torch.ones_like(m0))/v.shape[-1]
    return density,veloc,temperature


def fvmlinspace(vmin, vmax, nv):
    dv = (vmax-vmin)/nv
    return torch.linspace(vmin+dv/2, vmax-dv/2, nv)


def NDmsh(vmin, vmax, nv):
    def vmsh(vmin, vmax, nv):
        assert vmax > vmin
        assert nv > 0
        dv = (vmax-vmin)/nv
        v = torch.linspace(vmin+dv/2, vmax-dv/2, nv)
        w = (vmax-vmin)/nv*torch.ones_like(v)
        return v, w
    if isinstance(vmin, (int, float)) and isinstance(vmax, (int, float)) and isinstance(nv, int):
        v, w = vmsh(vmin, vmax, nv)
        v=v[...,None]
        w=w[...,None]
        return v, w, (v,), (w,)
    else:
        assert len(vmin) == len(vmax) == len(
            nv), "vmin,vmax,nv must be the same length"
        vL, wL = list(zip(*[vmsh(vmini, vmaxi, nvi)
                      for vmini, vmaxi, nvi in zip(vmin, vmax, nv)]))
        v = torch.meshgrid(*vL, indexing='ij')
        v = torch.stack([vi.flatten() for vi in v], axis=-1)
        w = torch.meshgrid(*wL, indexing='ij')
        w = torch.stack([wi.flatten() for wi in w], axis=-1)
        w = torch.prod(w,dim=-1,keepdims=True)
        return v, w, vL, wL
