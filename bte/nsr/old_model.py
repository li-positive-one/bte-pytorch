import torch
import torch.nn as nn
import math
from functorch import grad,jacfwd,vmap,jacrev
from bte.nsr.mesh import get_vmsh
from bte.nsr.block import MultiResFCSeq


def maxwellian(v, rho, u, T):
    return (rho / torch.sqrt(2 * math.pi * T)**v.shape[-1]) * torch.exp(-((u[..., None, :] - v) ** 2).sum(dim=-1) / (2 * T))
        
def fsum(f,w):
    return (f*w).sum(dim=-1,keepdim=True)

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

def rho_u_theta(f,v,w):
    m0,m1,m2=_m0(f,v,w),_m1(f,v,w),_m2(f,v,w)
    density = m0
    veloc = m1/m0
    v2 = (veloc ** 2).sum(dim=-1, keepdim=True)
    temperature = (m2.sum(dim=-1, keepdim=True)/m0-v2)/v.shape[-1]
    return density,veloc,temperature

def _m012(f,v,w):
    m0,m1,m2=_m0(f,v,w),_m1(f,v,w),_m2(f,v,w)
    return m0,m1,m2



sqrt6=math.sqrt(6)
class ResBlock(nn.Module):
    def __init__(self, hidden, activation=torch.sin):
        super().__init__()
        self.fc1=nn.Linear(hidden,hidden)
        self.act=activation

    def forward(self, x):
        x=self.act(self.fc1(x))
        return x

class FCSeq(nn.Module):
    def __init__(self, in_channel,out_channel,hidden):
        super().__init__()
        self.layer1=nn.Linear(in_channel,hidden)
        self.layer2=ResBlock(hidden)
        self.layer3=ResBlock(hidden)
        self.layer4=ResBlock(hidden)
        self.layer5=ResBlock(hidden)
        self.layer6=nn.Linear(hidden,out_channel)
        self.bn1=nn.Identity()
        self.bn2=nn.Identity()
        self.bn3=nn.Identity()
        self.bn4=nn.Identity()
    
    def forward(self, x):
        x=self.bn1(self.layer1(x))
        x=torch.sin(x)
        x=self.bn2(self.layer2(x))
        x=self.bn3(self.layer3(x))
        x=self.bn4(self.layer4(x))
        x=self.bn4(self.layer5(x))
        x=self.layer6(x)
        return x
    
class SingleRes(nn.Module):
    def __init__(self, in_channel, out_channel,hidden):
        super().__init__()
        self.net1=FCSeq(in_channel, out_channel,hidden)
    def forward(self, x):
        y=self.net1(x)
        return y
    
class MultiRes(nn.Module):
    def __init__(self, in_channel, out_channel,hidden):
        super().__init__()
        self.net1=FCSeq(3*in_channel, out_channel,hidden)
    def forward(self, x):
        xs=torch.cat([x,4*x,16*x],dim=-1)
        x1=self.net1(xs)
        y=x1
        return y

NetBlock=MultiRes

class SplitNet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden,VDIS):
        super().__init__()
        self.net_eq=NetBlock(in_channel, 5, hidden)
        self.net_neq=NetBlock(in_channel, out_channel, hidden)
        self.VDIS=VDIS
    def forward(self, x):
        www=self.net_eq(x)
        rho,u,theta=www[...,0:1],www[...,1:4],www[...,4:5]
        rho=torch.exp(-rho)
        theta=torch.exp(-theta)
        x1=maxwellian(self.VDIS,rho,u,theta)
        x2=self.net_neq(x)
        y=x1*(x1+0.01*x2)
        return y
