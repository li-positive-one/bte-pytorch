import torch
import numpy as np
import math
from .block import MultiResFCSeq
import torch.nn as nn


def maxwellian_1D(v, rho, u, T):
    """
    v [nv]
    rho [nx,1]
    u [nx,1]
    T [nx,1]
    """
    # print(v.shape,rho.shape,u.shape,T.shape)
    f = rho / torch.sqrt(2 * np.pi * T) * torch.exp(-((u - v) ** 2) / (2 * T))
    return f


def maxwellian_LR(vtuple, rho, u, T):
    """
    vx [nvx]
    vy [nvy]
    rho [nx,1]
    u [nx,3]
    T [nx,1]
    """
    D = len(vtuple)
    assert D == u.shape[-1]
    if D == 2:
        vx, vy = vtuple
        f1 = maxwellian_1D(vx, rho ** (1 / 2), u[..., 0:1], T)
        f2 = maxwellian_1D(vy, rho ** (1 / 2), u[..., 1:2], T)
        return f1[..., None], f2[..., None]
    elif D == 3:
        vx, vy, vz = vtuple
        f1 = maxwellian_1D(vx, rho ** (1 / 3), u[..., 0:1], T)
        f2 = maxwellian_1D(vy, rho ** (1 / 3), u[..., 1:2], T)
        f3 = maxwellian_1D(vz, rho ** (1 / 3), u[..., 2:3], T)
        return f1[..., None], f2[..., None], f3[..., None]
    else:
        raise ValueError


def maxwellian(v, rho, u, T):
    return (rho / torch.sqrt(2 * math.pi * T) ** v.shape[-1]) * torch.exp(
        -((u[..., None, :] - v) ** 2).sum(dim=-1) / (2 * T)
    )


def fsum_LR(ftuple, wtuple):
    if len(ftuple) == 2:
        fx, fy = ftuple
        wx, wy = wtuple
        sx = torch.einsum("...ir,i->...r", fx, wx)
        sy = torch.einsum("...jr,j->...r", fy, wy)
        s = torch.einsum("...r,...r->...", sx, sy)
    elif len(ftuple) == 3:
        fx, fy, fz = ftuple
        wx, wy, wz = wtuple
        sx = torch.einsum("...ir,i->...r", fx, wx)
        sy = torch.einsum("...jr,j->...r", fy, wy)
        sz = torch.einsum("...kr,k->...r", fz, wz)
        s = torch.einsum("...r,...r,...r->...", sx, sy, sz)
    else:
        raise ValueError
    return s[..., None]


def _m0_LR(ft, vt, wt):
    return fsum_LR(ft, wt)


def _m1_LR(ft, vt, wt):
    if len(ft) == 2:
        fx, fy = ft
        vx, vy = vt
        mux = fsum_LR((fx * vx[..., None], fy), wt)
        muy = fsum_LR((fx, fy * vy[..., None]), wt)
        return torch.cat([mux, muy], dim=-1)
    elif len(ft) == 3:
        fx, fy, fz = ft
        vx, vy, vz = vt
        mux = fsum_LR((fx * vx[..., None], fy, fz), wt)
        muy = fsum_LR((fx, fy * vy[..., None], fz), wt)
        muz = fsum_LR((fx, fy, fz * vz[..., None]), wt)
        return torch.cat([mux, muy, muz], dim=-1)
    else:
        raise ValueError


def _m2_LR(ft, vt, wt):
    if len(ft) == 2:
        fx, fy = ft
        vx, vy = vt
        mu2x = fsum_LR((fx * vx[..., None] ** 2, fy), wt)
        mu2y = fsum_LR((fx, fy * vy[..., None] ** 2), wt)
        return torch.cat([mu2x, mu2y], dim=-1)
    elif len(ft) == 3:
        fx, fy, fz = ft
        vx, vy, vz = vt
        mu2x = fsum_LR((fx * vx[..., None] ** 2, fy, fz), wt)
        mu2y = fsum_LR((fx, fy * vy[..., None] ** 2, fz), wt)
        mu2z = fsum_LR((fx, fy, fz * vz[..., None] ** 2), wt)
        return torch.cat([mu2x, mu2y, mu2z], dim=-1)
    else:
        raise ValueError


epsT = 1e-2
epsR = 1e-2


def rho_u_theta_LR(ft, vt, wt):
    m0, m1, m2 = _m0_LR(ft, vt, wt), _m1_LR(ft, vt, wt), _m2_LR(ft, vt, wt)
    density = torch.max(
        m0,
        torch.ones(
            [
                1,
            ],
            device=m0.device,
        )
        * epsR,
    )
    veloc = m1 / m0
    v2 = (veloc**2).sum(dim=-1, keepdim=True)
    temperature = torch.max(
        (m2.sum(dim=-1, keepdim=True) / m0 - v2) / len(ft),
        torch.ones(
            [
                1,
            ],
            device=v2.device,
        )
        * epsT,
    )
    return density, veloc, temperature


def _m012_LR(f, v, w):
    m0, m1, m2 = _m0_LR(f, v, w), _m1_LR(f, v, w), _m2_LR(f, v, w)
    return m0, m1, m2

class SplitNet2D(nn.Module):
    def __init__(self, neurons, rank, VT, xdim=1):
        super().__init__()
        self.D = len(VT)
        self.VT = VT
        self.nx = VT[0].shape[0]
        self.ny = VT[1].shape[0]

        self.backbone = MultiResFCSeq(
            [
                xdim+1,
            ]
            + neurons
        )
        self.headx = nn.Linear(neurons[-1], rank * self.nx)
        self.heady = nn.Linear(neurons[-1], rank * self.ny)
        self.headp = nn.Linear(neurons[-1], 2+self.D)
        self.rank = rank

    def forward(self, x):
        xout = torch.sin(self.backbone(x))

        www = self.headp(xout)
        rho, u, theta = (
            www[..., 0:1],
            www[..., 1 : 1 + self.D],
            www[..., 1 + self.D : 2 + self.D],
        )
        rho = torch.exp(-rho)
        theta = torch.exp(-theta)
        fmx, fmy = maxwellian_LR(self.VT, rho, u, theta)

        f2x = self.headx(xout)
        f2y = self.heady(xout)

        f2x = f2x.reshape(f2x.shape[:-1] + (self.nx, self.rank))
        f2x = (0.01**0.33) * fmx * f2x

        f2y = f2y.reshape(f2y.shape[:-1] + (self.ny, self.rank))
        f2y = (0.01**0.33) * fmy * f2y

        yx, yy = torch.cat([fmx**2, f2x], dim=-1), torch.cat([fmy**2, f2y], dim=-1)

        return yx, yy

class SplitNet_2D_LRNew(nn.Module):
    def __init__(self, neurons, rank, VT, xdim=1):
        super().__init__()
        self.D = len(VT)
        self.VT = VT
        self.nx = VT[0].shape[0]
        self.ny = VT[1].shape[0]

        self.backbone = MultiResFCSeq(
            [
                xdim+1,
            ]
            + neurons
        )
        self.headx = nn.Linear(neurons[-1], rank)
        self.heady = nn.Linear(neurons[-1], rank)
        self.feat_x = nn.parameter.Parameter(
            torch.tensor(np.random.uniform(-1, 1, (self.nx, rank)), dtype=torch.float32)
        )
        self.feat_y = nn.parameter.Parameter(
            torch.tensor(np.random.uniform(-1, 1, (self.ny, rank)), dtype=torch.float32)
        )
        self.u = nn.parameter.Parameter(torch.tensor(np.zeros(2), dtype=torch.float32))
        self.theta = nn.parameter.Parameter(
            torch.tensor(np.zeros(1), dtype=torch.float32)
        )
        self.headp = nn.Linear(neurons[-1], 4)
        self.rank = rank

    def forward(self, x):
        xout = torch.sin(self.backbone(x))

        www = self.headp(xout)
        rho, u, theta = (
            www[..., 0:1],
            www[..., 1 : 1 + self.D],
            www[..., 1 + self.D : 2 + self.D],
        )
        rho = torch.exp(-rho)
        theta = torch.exp(-theta)
        fmx, fmy = maxwellian_LR(self.VT, rho, u, theta)

        f2x = self.headx(xout)
        f2y = self.heady(xout)

        f2x = f2x.reshape(f2x.shape[:-1] + (self.nx, self.rank))
        f2x = (0.01**0.33) * fmx * f2x

        f2y = f2y.reshape(f2y.shape[:-1] + (self.ny, self.rank))
        f2y = (0.01**0.33) * fmy * f2y

        yx, yy = torch.cat([fmx, f2x], dim=-1), torch.cat([fmy, f2y], dim=-1)

        return yx, yy


class SplitNet3D(nn.Module):
    def __init__(self, neurons, rank, VT, multires, xdim=1):
        super().__init__()
        self.D = len(VT)
        self.VT = VT
        self.nx = VT[0].shape[0]
        self.ny = VT[1].shape[0]
        self.nz = VT[2].shape[0]

        self.backbone = MultiResFCSeq(
            [
                xdim+1,
            ]
            + neurons,multires=multires
        )
        self.headx = nn.Linear(neurons[-1], rank * self.nx)
        self.heady = nn.Linear(neurons[-1], rank * self.ny)
        self.headz = nn.Linear(neurons[-1], rank * self.nz)
        self.headp = nn.Linear(neurons[-1], 2 + self.D)
        self.rank = rank

    def forward(self, x):
        xout = torch.sin(self.backbone(x))

        www = self.headp(xout)
        rho, u, theta = (
            www[..., 0:1],
            www[..., 1 : 1 + self.D],
            www[..., 1 + self.D : 2 + self.D],
        )
        rho = torch.exp(-rho)
        theta = torch.exp(-theta)
        fmx, fmy, fmz = maxwellian_LR(self.VT, rho, u, theta)

        f2x = self.headx(xout)
        f2y = self.heady(xout)
        f2z = self.headz(xout)

        f2x = f2x.reshape(f2x.shape[:-1] + (self.nx, self.rank))
        f2x = (0.01**0.33) * fmx * f2x

        f2y = f2y.reshape(f2y.shape[:-1] + (self.ny, self.rank))
        f2y = (0.01**0.33) * fmy * f2y

        f2z = f2z.reshape(f2z.shape[:-1] + (self.nz, self.rank))
        f2z = (0.01**0.33) * fmz * f2z

        yx, yy,yz = torch.cat([fmx, f2x], dim=-1), torch.cat([fmy, f2y], dim=-1), torch.cat([fmz, f2z], dim=-1)
        return yx, yy, yz


class SplitNet_3D_LRNew(nn.Module):
    def __init__(self, neurons, rank, VT, xdim=1):
        super().__init__()
        self.D = len(VT)
        self.VT = VT
        self.nx = VT[0].shape[0]
        self.ny = VT[1].shape[0]
        self.nz = VT[2].shape[0]

        self.backbone = MultiResFCSeq(
            [
                xdim+1,
            ]
            + neurons
        )
        self.headx = nn.Linear(neurons[-1], rank)
        self.heady = nn.Linear(neurons[-1], rank)
        self.headz = nn.Linear(neurons[-1], rank)
        self.feat_x= nn.parameter.Parameter(torch.tensor(np.random.uniform(-1, 1,(self.nx,rank)),dtype=torch.float32))
        self.feat_y= nn.parameter.Parameter(torch.tensor(np.random.uniform(-1, 1,(self.ny,rank)),dtype=torch.float32))
        self.feat_z= nn.parameter.Parameter(torch.tensor(np.random.uniform(-1, 1,(self.nz,rank)),dtype=torch.float32))
        self.u=nn.parameter.Parameter(torch.tensor(np.zeros(3),dtype=torch.float32))
        self.theta=nn.parameter.Parameter(torch.tensor(np.zeros(1),dtype=torch.float32))
        self.headp = nn.Linear(neurons[-1], 1)
        self.rank = rank

    def forward(self, x):
        xout = torch.sin(self.backbone(x))

        rho = self.headp(xout)
        rho = torch.exp(-rho)
        theta = torch.exp(self.theta)
        fmx, fmy, fmz = maxwellian_LR(self.VT, rho, self.u[None,:], theta[None,:])

        f2x = self.headx(xout)
        f2y = self.heady(xout)
        f2z = self.headz(xout)

        f2x = f2x[...,None,:]*self.feat_x
        f2x = (0.01**0.33) * fmx * f2x

        f2y = f2y[...,None,:]*self.feat_y
        f2y = (0.01**0.33) * fmy * f2y

        f2z = f2z[...,None,:]*self.feat_z
        f2z = (0.01**0.33) * fmz * f2z
      
        yx, yy,yz = torch.cat([fmx, f2x], dim=-1), torch.cat([fmy, f2y], dim=-1), torch.cat([fmz, f2z], dim=-1)
        return yx, yy, yz


class NoSplitNet2D(nn.Module):
    def __init__(self, neurons, rank, VT):
        raise NotImplemented
