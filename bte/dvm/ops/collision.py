import torch
from torch.fft import fftn,ifftn,fftshift
import numpy as np
import math
from scipy import special
from ...utils.gas import get_potential

def lgwt(N:int,a:float,b:float):
    """Gauss-Legendre quadrature

    Args:
        N (int): points
        a (float): left
        b (float): right

    Returns:
        Tuple[torch.Tensor,torch.Tensor]: quadrature points, weight
    """
    x,w=np.polynomial.legendre.leggauss(N)
    x = 0.5*(x + 1)*(b - a) + a
    w = w * 0.5 * (b-a)
    return torch.from_numpy(x),torch.from_numpy(w)

def init_kernel_mode_vector(vmin, vmax, nv, quad_num:int=64, omega:float=0.81,M:int=5,dtype=torch.float64):
    """Get the collision kernel.

    reference: http://dx.doi.org/10.1016/j.jcp.2013.05.003

    Args:
        quad_num (int): _description_
        omega (float, optional): _description_. Defaults to 0.81.
        M (int, optional): _description_. Defaults to 5.
        dtype (_type_, optional): _description_. Defaults to torch.float64.

    Returns:
        _type_: _description_
    """
    PI=math.pi
    umax,vmax,wmax=vmax
    umin,vmin,wmin=vmin
    unum,vnum,wnum=nv

    alpha=get_potential(omega)
    du,dv,dw=(umax-umin)/(unum-1),(vmax-vmin)/(vnum-1),(wmax-wmin)/(wnum-1)
    supp=math.sqrt(2.0)*2.0*max(umax,vmax,wmax)/(3.0+math.sqrt(2.0))

    fre_vx=torch.linspace(-PI/du,(unum/2-1.0)*2.0*PI/unum/du,unum)
    fre_vy=torch.linspace(-PI/dv,(vnum/2-1.0)*2.0*PI/vnum/dv,vnum)
    fre_vz=torch.linspace(-PI/dw,(wnum/2-1.0)*2.0*PI/wnum/dw,wnum)

    abscissa, gweight=lgwt(quad_num, 0.0, supp)
    theta=PI/M*torch.arange(1,M-1+1)
    theta2=PI/M*torch.arange(1,M+1)

    s=((fre_vx[:,None,None]*torch.sin(theta)[None,:,None]*torch.cos(theta2)[None,None,:])[:,None,None,:,:]+
        (fre_vy[:,None,None]*torch.sin(theta)[None,:,None]*torch.sin(theta2)[None,None,:])[None,:,None,:,:]+
        (fre_vz[:,None,None]*torch.cos(theta)[None,:,None])[None,None,:,:,:]
    ) 

    int_temp= (2 * gweight[...,None,None,None,None,None] * 
                torch.cos( s[None,...] * abscissa[...,None,None,None,None,None] ) * 
                (abscissa[...,None,None,None,None,None]**alpha)).sum(dim=0)
    phi2 = int_temp*torch.sin(theta[None,None,None,:,None]) 

    s=((fre_vx*fre_vx)[:,None,None,None,None]+
        (fre_vy*fre_vy)[None,:,None,None,None]+
        (fre_vz*fre_vz)[None,None,:,None,None] - s*s)
    
    psi2 = torch.zeros((unum,vnum,wnum,M-1,M),dtype=dtype)
    
    so=s.clone()
    s=s.abs()
    s = torch.sqrt(s)
    bel = supp*s
    bessel = torch.from_numpy(special.jv(1,bel.numpy()))

    psi2=2.0*PI*supp*bessel/s
    psi2[so<=0] = PI*supp*supp
    phipsi2=(phi2*psi2).sum(dim=(-1,-2))

    return phi2.to(dtype=dtype),psi2.to(dtype=dtype),phipsi2.to(dtype=dtype)


def collision_fsm(f, kn_bzm, phi, psi, phipsi)->torch.Tensor:
    oshape=f.shape
    unum,vnum,wnum=phi.shape[:3]
    f=f.reshape(oshape[:-1]+(unum,vnum,wnum))
    return collision_fft(f, kn_bzm, phi, psi,phipsi).reshape(oshape)

def collision_fft(f_spec, kn_bzm, phi, psi, phipsi)->torch.Tensor:
    unum,vnum,wnum=phi.shape[:3]
    ifft3d=lambda x:ifftn(x,dim=(-3,-2,-1),norm="forward")
    fft3d=lambda x:fftn(x,dim=(-3,-2,-1),norm="backward")

    f_spec=ifft3d(f_spec)
    f_spec=f_spec/(unum*vnum*wnum)
    
    f_spec=fftshift(f_spec,dim=(-3,-2,-1))
    f_temp=0
    M=phi.shape[-1]
    for i in range(1,M-1+1):
        for j in range(1,M+1):
            fc1=f_spec*phi[:,:,:,i-1,j-1]
            fc2=f_spec*psi[:,:,:,i-1,j-1]
            fc11=fft3d(fc1)
            fc22=fft3d(fc2)
            f_temp=f_temp+fc11*fc22
    fc1=f_spec*phipsi
    fc2=f_spec
    fc11=fft3d(fc1)
    fc22=fft3d(fc2)
    f_temp=f_temp-fc11*fc22
    Q = 4.0*np.pi**2/kn_bzm/M**2*f_temp.real
    return Q

def collision_fft_fg(f_spec, g_spec, kn_bzm, phi, psi, phipsi)->torch.Tensor:
    unum,vnum,wnum=phi.shape[:3]
    ifft3d=lambda x:ifftn(x,dim=(-3,-2,-1),norm="forward")
    fft3d=lambda x:fftn(x,dim=(-3,-2,-1),norm="backward")

    f_spec=ifft3d(f_spec)
    f_spec=f_spec/(unum*vnum*wnum)
    
    g_spec=ifft3d(g_spec)
    g_spec=g_spec/(unum*vnum*wnum)

    f_spec=fftshift(f_spec,dim=(-3,-2,-1))
    g_spec=fftshift(g_spec,dim=(-3,-2,-1))
    f_temp=0
    M=phi.shape[-1]
    for i in range(1,M-1+1):
        for j in range(1,M+1):
            fc1=f_spec*phi[:,:,:,i-1,j-1]
            fc2=g_spec*psi[:,:,:,i-1,j-1]
            fc11=fft3d(fc1)
            fc22=fft3d(fc2)
            f_temp=f_temp+fc11*fc22
    fc1=f_spec*phipsi
    fc2=g_spec
    fc11=fft3d(fc1)
    fc22=fft3d(fc2)
    f_temp=f_temp-fc11*fc22
    Q = 4.0*np.pi**2/kn_bzm/M**2*f_temp.real
    return Q