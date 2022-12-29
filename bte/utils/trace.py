import torch

def trace2d(T:torch.Tensor):
    assert T.shape[-1]==T.shape[-2]
    sum=0
    for i in range(T.shape[-1]):
        sum=sum+T[...,i,i]
    return sum[...,None,None]

def tensor2d_tosymmetry(T:torch.Tensor):
    assert T.shape[-1]==T.shape[-2]
    Tt=T.transpose(-1,-2)
    return (T+Tt)/2

def tensor2d_totrace0(T:torch.Tensor,need_symmetrize=True):
    if need_symmetrize:
        T=tensor2d_tosymmetry(T)
    delta=torch.eye(T.shape[-1])
    TracePart=1/(T.shape[-1])*torch.einsum("...kk,ij->...ij",T,delta)
    return T-TracePart

def tensor3d_tosymmetry(T:torch.Tensor):
    assert T.shape[-1]==T.shape[-2]==T.shape[-3]
    sum=T
    einsums=["...ijk->...ikj","...ijk->...jik","...ijk->...jki","...ijk->...kij","...ijk->...kji"]
    for i in range(5):
        Tx=torch.einsum(einsums[i],T)
        sum=sum+Tx
    return sum/6

def tensor3d_totrace0(T:torch.Tensor,need_symmetrize=True):
    if need_symmetrize:
        T=tensor3d_tosymmetry(T)
    delta=torch.eye(T.shape[-1])
    P1=torch.einsum("...ill,jk->...ijk",T,delta)
    P2=torch.einsum("...jll,ik->...ijk",T,delta)
    P3=torch.einsum("...kll,ij->...ijk",T,delta)
    TracePart=1/(T.shape[-1]+2)*(P1+P2+P3)
    return T-TracePart