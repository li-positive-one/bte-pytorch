import torch
import torch.nn as nn
import math
from functorch import grad,jacfwd,vmap,jacrev
from bte.nsr.mesh import get_vmsh
from bte.nsr.block import MultiResFCSeq
import logging
logger = logging.getLogger(__name__)

def maxwellian(v, rho, u, T):
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

eps=1e-3
def rho_u_theta(f,v,w):
    m0,m1,m2=_m0(f,v,w),_m1(f,v,w),_m2(f,v,w)
    density = torch.maximum(m0,eps*torch.ones_like(m0))
    veloc = m1/m0
    v2 = (veloc ** 2).sum(dim=-1, keepdim=True)
    temperature = torch.maximum((m2.sum(dim=-1, keepdim=True)/m0-v2),eps*torch.ones_like(m0))/v.shape[-1]
    return density,veloc,temperature

def _m012(f,v,w):
    m0,m1,m2=_m0(f,v,w),_m1(f,v,w),_m2(f,v,w)
    return m0,m1,m2

class SplitNet(nn.Module):
    def __init__(self, neurons, VDIS, multires, xdim=1):
        super().__init__()
        self.D=VDIS.shape[-1]
        self.register_buffer("VDIS",VDIS)
        self.net_eq = MultiResFCSeq([xdim+1,]+neurons+[2 + self.D],multires=multires)
        self.net_neq = MultiResFCSeq([xdim+1,]+neurons+[VDIS.shape[0]],multires=multires)
        
    def forward(self, x):
        www=self.net_eq(x)
        rho,u,theta=www[...,0:1],www[...,1:4],www[...,4:5]
        rho=torch.exp(-rho)
        theta=torch.exp(-theta)
        x1=maxwellian(self.VDIS,rho,u,theta)
        x2=self.net_neq(x)
        y=x1*(x1+0.01*x2)
        return y
 

class NoSplitNet(nn.Module):
    def __init__(self, neurons, VDIS):
        super().__init__()
        self.D=VDIS.shape[-1]
        self.register_buffer("VDIS",VDIS)
        self.backbone = MultiResFCSeq([2,]+neurons+[VDIS.shape[0],])
        
    def forward(self, x):
        xout = self.backbone(x)
        return xout

class LossCompose(nn.Module):
    def __init__(self, n_class, eta=1e-3):
        super().__init__()
        self.n_class=n_class
        self.w=nn.parameter.Parameter(torch.ones(n_class))
        self.register_buffer("eta",torch.Tensor([eta]))

    def forward(self, loss):
        assert loss.shape[-1]==self.n_class
        ww=self.eta**2+self.w**2
        Loss=1/2/ww*loss+torch.log(1+ww)
        return Loss.sum()/self.n_class


def parser_loss(config, VDIS, WDIS):
    if config["network"]["lossfunc"]=="SmoothL1":
        lossf =torch.nn.SmoothL1Loss(reduction='none', beta=1.0)
    else:
        lossf =torch.nn.MSELoss(reduction='none')

    criterion=lambda x,y:lossf(x,y).mean(dim=0)
    criterion_norm=lambda x:lossf(x,torch.zeros_like(x)).mean(dim=0)
    def prim_norm(f):
        m1,m2,m3=_m012(f,VDIS,WDIS)
        return torch.cat([criterion_norm(m1),criterion_norm(m2),criterion_norm(m3)],dim=-1)
    return criterion, criterion_norm, prim_norm

class Problem(nn.Module):
    def __init__(self, config, model, LC) -> None:
        super().__init__()
        self.model=model
        self.LC=LC

    def forward(self,*args,**kargs):
        pass
