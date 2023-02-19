import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from torch.fft import fftn, ifftn, fftshift
import math
import scipy
######################
## 正交化函数
######################
def orthonormalize(vectors):
    """
    Orthonormalizes the vectors using gram schmidt procedure.

    Parameters:
        vectors: torch tensor, size (dimension, n_vectors)
                they must be linearly independant
    Returns:
        orthonormalized_vectors: torch tensor, size (dimension, n_vectors)
    """
    assert vectors.size(1) <= vectors.size(
        0
    ), "number of vectors must be smaller or equal to the dimension"
    orthonormalized_vectors = torch.zeros_like(vectors)
    orthonormalized_vectors[:, 0] = vectors[:, 0] / torch.norm(vectors[:, 0], p=2)

    for i in range(1, orthonormalized_vectors.size(1)):
        vector = vectors[:, i]
        V = orthonormalized_vectors[:, :i]
        PV_vector = torch.mv(V, torch.mv(V.t(), vector))
        orthonormalized_vectors[:, i] = (vector - PV_vector) / torch.norm(
            vector - PV_vector, p=2
        )

    return orthonormalized_vectors


class Enforcer(nn.Module):
    def __init__(self, VDIS, WDIS):
        super().__init__()
        C = (
            torch.concat([torch.ones_like(VDIS[:, 0:1]), VDIS, VDIS**2], dim=-1)
            * WDIS
        )
        C = C.mT
        CCT = torch.linalg.inv(C @ C.T) @ C
        self.register_buffer("C", C)
        self.register_buffer("CCT", CCT)

    def forward(self, f):
        return f - f @ self.C.T @ self.CCT


######################
## 定义非对称碰撞的双线性函数
######################

def collision_fft_fg(f_spec, g_spec, kn_bzm, phi, psi, phipsi) -> torch.Tensor:
    unum, vnum, wnum = phi.shape[:3]
    ifft3d = lambda x: ifftn(x, dim=(-3, -2, -1), norm="forward")
    fft3d = lambda x: fftn(x, dim=(-3, -2, -1), norm="backward")

    f_spec = ifft3d(f_spec)
    g_spec = ifft3d(g_spec)
    f_spec = f_spec / (unum * vnum * wnum)
    g_spec = g_spec / (unum * vnum * wnum)

    f_spec = fftshift(f_spec, dim=(-3, -2, -1))
    g_spec = fftshift(g_spec, dim=(-3, -2, -1))
    f_temp = 0
    M = phi.shape[-1]
    for i in range(1, M - 1 + 1):
        for j in range(1, M + 1):
            fc1 = f_spec * phi[:, :, :, i - 1, j - 1]
            fc2 = g_spec * psi[:, :, :, i - 1, j - 1]
            fc11 = fft3d(fc1)
            fc22 = fft3d(fc2)
            f_temp = f_temp + fc11 * fc22
    fc1 = f_spec * phipsi
    fc2 = g_spec
    fc11 = fft3d(fc1)
    fc22 = fft3d(fc2)
    f_temp = f_temp - fc11 * fc22
    Q = 4.0 * np.pi**2 / kn_bzm / M**2 * f_temp.real
    return Q

from bte.utils.gas import *
from bte.dvm.collision import *
from bte.dvm import solver as bgk_solver

# phi,psi,phipsi=init_kernel_mode_vector(10,-10,N,10,-10,N,10,-10,N,5)

def get_new_kernel(f_bases, f_bases2, nx, ny, nz, kn_bzm, phi, psi, phipsi):
    k = f_bases.shape[1]
    print(k)
    T = torch.zeros((k, k, k), device=f_bases.device)
    for i in range(k):
        for j in range(k):
            f1 = f_bases[:, i].reshape((nx, ny, nz))
            f2 = f_bases[:, j].reshape((nx, ny, nz))
            f3 = collision_fft_fg(f1, f2, kn_bzm, phi, psi, phipsi).reshape(
                (nx * ny * nz, 1)
            )
            coef = f_bases2.T @ f3
            T[i, j, :] = coef[..., 0]
    return T

class ReductionCollision(nn.Module):
    def __init__(self,F,G,K,Ortho=False):
        super().__init__()
        self.register_buffer("F",F)
        self.register_buffer("K",K)
        self.register_buffer("G",G)
        self.Ortho=Ortho
    def do_collision(self, f, kn_bzm=1.0):
        ff=f@self.F
        Q=torch.einsum("...i,...j,ijk->...k", ff, ff, self.K)
        Qr=Q@self.G.T
        return Qr

# def get_reduced_kernel(config, traindata, Rank):
#     vmin = config.vmesh.vmin
#     vmax = config.vmesh.vmax
#     nv = config.vmesh.nv
#     omega = config.omega
#     kn = config.Kn
#     alpha = get_potential(omega)
#     mu_ref = get_mu(alpha,omega,kn)
#     solver = bgk_solver.BGKSolver(-0.5,0.5,1,vmin,vmax,nv,device='cpu',BC_type='periodic',bgk_simple_kn="mu_ref")
#     solver.set_collisioner("FFT")
#     solver.cuda()
#     Qdata = solver.coller.do_collision(traindata.cuda(), 1.0).cpu()
#     U, S, Vh = torch.linalg.svd(traindata.cuda(), full_matrices=False)
#     U_2, S_2, Vh_2 = torch.linalg.svd(Qdata.cuda(), full_matrices=False)

#     nvprod=math.prod(nv)
#     nvx,nvy,nvz=nv
#     v=solver.dis.v_meta.v.float()
#     w=solver.dis.v_meta.w.float()
#     VDIS=v
#     WDIS=w

#     c_rho=torch.ones((1,nvprod)).cuda()
#     c_veloc=VDIS.mT.cuda()#.sum(dim=0,keepdim=True)
#     c_energy=(VDIS.mT.cuda()**2).sum(dim=0,keepdim=True)
#     c_feature=torch.cat([c_rho,c_veloc,c_energy])

#     VhS=Vh*S[...,None]
#     CC=VhS
#     CC=torch.cat([c_feature/CC.shape[-1],CC])
#     VhSC=orthonormalize(CC.T).T
#     f_bases=VhSC[:Rank].T.cuda()

#     Vh2S=Vh_2*S_2[...,None]
#     CC2=Vh2S
#     CC2=torch.cat([c_feature/CC2.shape[-1],CC2])
#     Vh2SC=orthonormalize(CC2.T).T
#     f_bases2=Vh2SC[3:3+Rank].T.cuda()

#     phi,psi,phipsi=init_kernel_mode_vector(vmax[0],vmin[0],nvx,vmax[1],vmin[1],nvy,vmax[2],vmin[2],nvz,5)
#     nK=get_new_kernel(f_bases,f_bases2, nvx,nvy,nvz, 1.0, phi.float().cuda(), psi.float().cuda(), phipsi.float().cuda())
#     RC=ReductionCollision(f_bases, f_bases2, nK, True).cuda()
#     return RC, f_bases, f_bases2


def get_reduced_kernel(config, traindata, Rank):
    vmin = config.vmesh.vmin
    vmax = config.vmesh.vmax
    nv = config.vmesh.nv
    omega = config.omega
    kn = config.Kn
    alpha = get_potential(omega)
    mu_ref = get_mu(alpha,omega,kn)
    solver = bgk_solver.BGKSolver(-0.5,0.5,1,vmin,vmax,nv,device='cpu',BC_type='periodic',bgk_simple_kn="mu_ref")
    solver.set_collisioner("FFT")
    solver.cuda()
    Qdata = solver.coller.do_collision(traindata.cuda(), 1.0).cpu()
    U, S, Vh = torch.linalg.svd(traindata.cuda(), full_matrices=False)
    U_2, S_2, Vh_2 = torch.linalg.svd(Qdata.cuda(), full_matrices=False)

    nvprod=math.prod(nv)
    nvx,nvy,nvz=nv
    v=solver.dis.v_meta.v.float()
    w=solver.dis.v_meta.w.float()
    VDIS=v
    WDIS=w

    c_rho=torch.ones((1,nvprod)).cuda()
    c_veloc=VDIS.mT.cuda()#.sum(dim=0,keepdim=True)
    c_energy=(VDIS.mT.cuda()**2).sum(dim=0,keepdim=True)
    c_feature=torch.cat([c_rho,c_veloc,c_energy])

    CC_Feat = c_feature.cpu().numpy().T
    CC_Feat = scipy.linalg.orth(CC_Feat)
    print(CC_Feat.shape)

    # VhS=Vh*S[...,None]
    # print(VhS.shape)
    # CC = (VhS.cpu().numpy().T)[:,:Rank]
    # CC=np.concatenate([CC_Feat,CC],axis=1)
    
    # VhSC=scipy.linalg.orth(CC)
    # #VhSC=orthonormalize(CC.T).T
    # f_bases=torch.from_numpy(VhSC).cuda()


    VhS=Vh*S[...,None]
    CC=VhS
    CC=torch.cat([c_feature/CC.shape[-1],CC])
    VhSC=orthonormalize(CC.T).T
    f_bases=VhSC[:Rank].T.cuda()


    Vh2S=Vh_2*S_2[...,None]
    CC2=Vh2S.cpu().numpy().T[:,:Rank]
    #CC2=torch.cat([c_feature/CC2.shape[-1],CC2])
    CC2=CC2-CC_Feat@(CC_Feat.T@CC2)
    #Vh2SC=orthonormalize(CC2.T).T
    Vh2SC=scipy.linalg.orth(CC2)
    #f_bases2=Vh2SC[3:3+Rank].T.cuda()
    f_bases2=torch.from_numpy(Vh2SC).cuda()#[:,:Rank].cuda()
    print(f_bases.shape,f_bases2.shape)

    phi,psi,phipsi=init_kernel_mode_vector(vmax[0],vmin[0],nvx,vmax[1],vmin[1],nvy,vmax[2],vmin[2],nvz,5)
    nK=get_new_kernel(f_bases,f_bases2, nvx,nvy,nvz, 1.0, phi.float().cuda(), psi.float().cuda(), phipsi.float().cuda())
    RC=ReductionCollision(f_bases, f_bases2, nK, True).cuda()
    return RC, f_bases, f_bases2