from __future__ import annotations
from typing import Callable, Tuple, List
import torch
import torch.nn.functional as F
import torch.nn as nn
from math import factorial, gamma, sqrt
import math
import numpy as np
import abc
import functools
from bte.utils.indexs import *
from bte.utils.operators import make_tensor
from bte.utils.specials import eval_hermitenorm


class distributionBase(metaclass=abc.ABCMeta):
    """
    The base class for gas distribution.
    """

    @abc.abstractmethod
    def density(self):
        pass

    @abc.abstractmethod
    def velocity(self):
        pass

    @abc.abstractmethod
    def temperature(self):
        pass

    @abc.abstractmethod
    def __add__(self, anotherDis):
        pass

    @abc.abstractmethod
    def __radd__(self, anotherDis):
        pass

    @abc.abstractmethod
    def __sub__(self, anotherDis):
        pass

    @abc.abstractmethod
    def __mul__(self, multiplier):
        pass

    @abc.abstractmethod
    def __rmul__(self, multiplier):
        pass

    @abc.abstractmethod
    def __truediv__(self, divider):
        pass

class DVDisMetaBase():
    pass

class DVDisMeta(nn.Module,DVDisMetaBase):
    def __init__(self, v, v_w, gamma=5/3, *args):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("v", v)
        self.register_buffer("v2", v ** 2)
        self.register_buffer("v3", v ** 3)
        self.register_buffer("w", v_w)

class DVDisMeta_Grid(nn.Module,DVDisMetaBase):
    def __init__(self, vL: List, v_wL: List, *args):
        nn.Module.__init__(self)
        assert len(vL) == len(v_wL)
        self.v_dims = nn.ModuleList([DVDisMeta(v, v_w) for v, v_w in zip(vL, v_wL)])
        self.dim = len(vL)
        V, W = product_meshgrid(vL, v_wL)
        self.register_buffer("v", V)
        self.register_buffer("v2", V ** 2)
        self.register_buffer("v3", V ** 3)
        self.register_buffer("w", W)


class DVDis(nn.Module,distributionBase):
    """
    The gas distribution for discrete velocity method.
    """
    def __init__(self, v_meta: DVDisMetaBase, data):
        super().__init__()
        self.v_meta = v_meta
        self.ndim = v_meta.v.shape[-1]
        self.register_buffer("f", data)

    @property
    def v(self):
        return self.v_meta.v

    @property
    def v2(self):
        return self.v_meta.v2

    @property
    def v3(self):
        return self.v_meta.v3

    @property
    def w(self):
        return self.v_meta.w

    def __add__(self, anotherDis: DVDis):
        return DVDis(self.v_meta, self.f + anotherDis.f)

    def __radd__(self, anotherDis: DVDis):
        return DVDis(self.v_meta, self.f + anotherDis.f)

    def __sub__(self, anotherDis):
        return DVDis(self.v_meta, self.f - anotherDis.f)

    def __mul__(self, multiplier):
        return DVDis(self.v_meta, self.f * multiplier)

    def __rmul__(self, multiplier):
        return DVDis(self.v_meta, self.f * multiplier)

    def __truediv__(self, divider):
        return DVDis(self.v_meta, self.f / divider)

    @staticmethod
    def empty(nv: int, vmin: float, vmax: float, method="uni", device="cpu") -> DVDis:
        """An empty new distribution. with velocity grid created by method 'uni' or 'leg'.

        Args:
            nv (int): Numbers of grid
            vmin (float): vmin
            vmax (float): vmax
            method (str, optional): 'uni' for uniform_velocity, 'leg' for legendre_velocity. Defaults to 'uni'.

        Raises:
            ValueError: [description]

        Returns:
            DVDis: [description]
        """
        if method == "uni":
            v, w = uniform_velocity(nv, vmin, vmax)
        elif method == "leg":
            v, w = legendre_velocity(nv, vmin, vmax)
        else:
            raise ValueError(f"method must be uni pof leg, but is {method}.")
        v = v.to(device=device)
        w = w.to(device=device)
        meta = DVDisMeta(v, w)
        return DVDis(meta, None)

    def sum(self, f: torch.Tensor):
        return f @ self.w  # .sum(dim=(-1))
        # return torch.einsum("...ij,...ij->...", f, self.w)

    def density(self, f=None):
        """Compute the macro quantity: density
        """
        if f is None:
            f = self.f
        density = self._m0(f)
        return density

    def velocity(self, f=None):
        """Compute the macro quantity: velocity
        """
        if f is None:
            f = self.f
        velocity = self._m1(f) / self._m0(f)
        return velocity

    def temperature(self, f=None):
        """Compute the macro quantity: temperature
        """
        if f is None:
            f = self.f
        m0 = self._m0(f)
        m1 = self._m1(f)
        v = m1 / m0
        v2 = (v ** 2).sum(dim=-1, keepdim=True)
        temperature = (self._m2(f).sum(dim=-1, keepdim=True) / m0 - v2) / self.v.shape[
            -1
        ]
        return temperature

    def _m0(self, f=None):
        r"""0-th order momentum of f"""
        if f is None:
            f = self.f
        return self.sum(f)

    def _m1(self, f=None):
        r"""1st order momentum of f"""
        if f is None:
            f = self.f
        return torch.cat(
            [self.sum(f * self.v[..., i]) for i in range(self.v.shape[-1])], dim=-1
        )

    def _m2(self, f=None):
        r"""2nd order momentum of f"""
        if f is None:
            f = self.f
        return torch.cat(
            [self.sum(f * self.v2[..., i]) for i in range(self.v.shape[-1])], dim=-1
        )

    def _m3(self, f=None):
        r"""3nd order momentum of f"""
        if f is None:
            f = self.f
        return torch.cat(
            [self.sum(f * self.v3[..., i]) for i in range(self.v.shape[-1])], dim=-1
        )

    def _m(self,alpha, i, j, k, f=None):
        r"""ijk order momentum of f"""
        if f is None:
            f = self.f
        return self.sum(f 
            *torch.sum(self.v[..., :])**alpha
            *self.v[..., 0]** i
            *self.v[..., 1]** j 
            *self.v[..., 2]** k)

    def _M(self, alpha, i, j, k, f=None):
        r"""ijk order momentum of f"""
        if f is None:
            f = self.f
        velocity=self.velocity()[...,None,:]
        return self.sum(f 
                        * torch.sum((self.v[..., :]-velocity[...,:])**2,dim=-1)** alpha
                        * (self.v[..., 0]-velocity[...,0])** i
                        * (self.v[..., 1]-velocity[...,1])** j
                        * (self.v[..., 2]-velocity[...,2])** k)

    def heatflux(self, f=None):
        if f is None:
            f = self.f
        m0 = self._m0(f)
        m1 = self._m1(f)
        m2 = self._m2(f)
        m3 = self._m3(f)
        u = m1 / m0
        q = -(u ** 3) * m0 + 3 * (u ** 2) * m1 - 3 * u * m2 + m3
        return q

    def rho_u_theta(self, f=None):
        # TODO: 这里可以避免重复计算
        return self.density(f), self.velocity(f), self.temperature(f)

    def maxwellian(self, rho_u_theta=None):
        if rho_u_theta is None:
            rho_u_theta = self.rho_u_theta()
        rho, u, T = rho_u_theta
        return (rho / torch.sqrt(2 * math.pi * T) ** self.v.shape[-1]) * torch.exp(
            -((u[..., None, :] - self.v) ** 2).sum(dim=-1) / (2 * T)
        )

    def maxwellian_half(self, rho_u_theta=None, sign=0):
        maxwellian = self.maxwellian(rho_u_theta=rho_u_theta)
        if sign > 0:
            maxwellian[..., self.v[..., 0] < 0] = 0
            maxwellian[..., self.v[..., 0] == 0] = (
                maxwellian[..., self.v[..., 0] == 0] / 2
            )
        elif sign < 0:
            maxwellian[..., self.v[..., 0] > 0] = 0
            maxwellian[..., self.v[..., 0] == 0] = (
                maxwellian[..., self.v[..., 0] == 0] / 2
            )
        else:
            raise ValueError("sign should be 1 or -1")
        return maxwellian

    def one_side(self, f=None, sign=0):
        if f is None:
            f = self.f
        f = f.clone()
        if sign == 1:
            f[..., self.v[..., 0] < 0] = 0
            f[..., self.v[..., 0] == 0] = f[..., self.v[..., 0] == 0] / 2
        elif sign == -1:
            f[..., self.v[..., 0] > 0] = 0
            f[..., self.v[..., 0] == 0] = f[..., self.v[..., 0] == 0] / 2
        else:
            raise ValueError("sign should be 1 or -1")
        return f

    def reverse(self, f=None):
        if f is None:
            f = self.f
        return torch.flip(f, [-1,])

    def from_HermiteDis(self, dis) -> "DVDis":
        rho, u, theta = dis.rho_u_theta
        f = dis.coef
        v = (self.v[..., 0] - u) / torch.sqrt(theta)
        h = 0
        sqr_T = theta.sqrt()
        kernel = torch.exp(-(v ** 2) / 2) / torch.sqrt(2 * math.pi * theta)
        for i in range(f.shape[-1]):
            h = h + f[..., i : i + 1] * eval_hermitenorm(i, v) / sqr_T ** i
        # h = (eval_hermitenorm_stack(f.shape[-1],v) @ f.unsqueeze(-1)).squeeze(-1)
        self.f = h * kernel
        return self


def uniform_velocity(
    nv: int, vmin: float, vmax: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """return a discrete of velocity with uniform partition.

    v_0=vmin+0.5*(vmax-vmin)/nv
    v_{nv-1}=vmax-0.5*(vmax-vmin)/nv

    Args:
        nv (int): Numbers of grid
        vmin (float): vmin
        vmax (float): vmax

    Returns:
        Tuple[torch.Tensor,torch.Tensor]: v, weight of v
    """
    dv = 2 / nv
    v = torch.arange(start=-1 + 0.5 * dv, end=1 + 0.5 * dv, step=dv)
    w = dv * torch.ones_like(v)
    v1 = 0.5 * (v + 1.0) * (vmax - vmin) + vmin
    w1 = 0.5 * (vmax - vmin) * w
    # v1 = torch.tensor(v1, dtype=torch.get_default_dtype())
    # w1 = torch.tensor(w1, dtype=torch.get_default_dtype())
    v1 = v1.reshape([-1, 1])
    w1 = w1.reshape([-1, 1])
    return v1, w1


def legendre_velocity(
    nv: int, vmin: float, vmax: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """return a discrete of velocity with points on legendre-gauss points.

    Args:
        nv (int): Numbers of grid
        vmin (float): vmin
        vmax (float): vmax

    Returns:
        Tuple[torch.Tensor,torch.Tensor]: v, weight of v
    """
    v, w = np.polynomial.legendre.leggauss(nv)
    v1 = 0.5 * (v + 1.0) * (vmax - vmin) + vmin
    w1 = 0.5 * (vmax - vmin) * w
    v1 = torch.tensor(v1, dtype=torch.get_default_dtype())
    w1 = torch.tensor(w1, dtype=torch.get_default_dtype())
    v1 = v1.reshape([-1, 1])
    w1 = w1.reshape([-1, 1])
    return v1, w1


def uniform_velocity_2d(nv, vmin, vmax):
    v1, w1 = uniform_velocity(nv[0], vmin[0], vmax[0])
    v2, w2 = uniform_velocity(nv[1], vmin[1], vmax[1])
    grid_x, grid_y = torch.meshgrid(v1.squeeze(), v2.squeeze())
    v = torch.stack([grid_x, grid_y], dim=-1)
    w = w1.reshape([-1, 1]) * w2.reshape([1, -1])
    v = v.reshape([-1, 2])
    w = w.reshape([-1, 1])
    return v, w


def legendre_velocity_2d(nv, vmin, vmax):
    v1, w1 = legendre_velocity(nv[0], vmin[0], vmax[0])
    v2, w2 = legendre_velocity(nv[1], vmin[1], vmax[1])
    grid_x, grid_y = torch.meshgrid(v1.squeeze(), v2.squeeze())
    v = torch.stack([grid_x, grid_y], dim=-1)
    w = w1.reshape([-1, 1]) * w2.reshape([1, -1])
    v = v.reshape([-1, 2])
    w = w.reshape([-1, 1])
    return v, w


def ND_velocity(func: Callable[[int, int, int], Tuple]):
    def wrapper(nvT: Tuple, vminT: Tuple, vmaxT: Tuple):
        assert len(nvT) == len(vminT)
        assert len(nvT) == len(vmaxT)
        vL = []
        wL = []
        DIM = len(nvT)
        for d, nv, vmin, vmax in zip(range(DIM), nvT, vminT, vmaxT):
            v, w = func(nv, vmin, vmax)
            w = w.reshape([1,] * d + [-1,] + [1,] * (DIM - d - 1))
            vL.append(v)
            wL.append(w)
        grids = torch.meshgrid(*[v.squeeze() for v in vL])
        V = torch.stack(grids, dim=-1)
        print(V.shape)
        W = functools.reduce(lambda a, b: a * b, wL)
        V = V.reshape([-1, DIM])
        W = W.reshape([-1, 1])
        return V, W

    return wrapper


uniform_velocity_nd = ND_velocity(uniform_velocity)
legendre_velocity_nd = ND_velocity(legendre_velocity)


def velocity_list(func, nvT: Tuple, vminT: Tuple, vmaxT: Tuple):
    assert len(nvT) == len(vminT)
    assert len(nvT) == len(vmaxT)
    vL = []
    wL = []
    DIM = len(nvT)
    for d, nv, vmin, vmax in zip(range(DIM), nvT, vminT, vmaxT):
        v, w = func(nv, vmin, vmax)
        w = w.reshape([1,] * d + [-1,] + [1,] * (DIM - d - 1))
        vL.append(v)
        wL.append(w)
    return vL, wL


def product_meshgrid(vL: List, wL: List):
    assert len(vL) == len(wL)
    DIM = len(vL)
    grids = torch.meshgrid(*[v.squeeze() for v in vL])
    V = torch.stack(grids, dim=-1)
    print(V.shape)
    W = functools.reduce(lambda a, b: a * b, wL)
    V = V.reshape([-1, DIM])
    W = W.reshape([-1, 1])
    return V, W


class DVDis_Chu(nn.Module, distributionBase):
    def __init__(self, v_meta, data_g, data_h, ndim=3):
        self.v_meta = v_meta
        self.ndim = ndim
        self.g = DVDis(v_meta, data_g)
        self.h = DVDis(v_meta, data_h)

    def __add__(self, another):
        if isinstance(another, DVDis_Chu):
            return DVDis_Chu(
                self.v_meta,
                self.g.f + another.g.f,
                self.h.f + another.h.f,
                ndim=self.ndim,
            )
        else:
            return DVDis_Chu(
                self.v_meta, self.g.f + another, self.h.f + another, ndim=self.ndim
            )

    def __radd__(self, another):
        return DVDis_Chu(
            self.v_meta, self.g.f + another, self.h.f + another, ndim=self.ndim
        )

    def __sub__(self, another):
        return DVDis_Chu(
            self.v_meta, self.g.f - another.g.f, self.h.f - another.h.f, ndim=self.ndim
        )

    def __mul__(self, multiplier):
        return DVDis_Chu(
            self.v_meta, self.g.f * multiplier, self.h.f * multiplier, ndim=self.ndim
        )

    def __rmul__(self, multiplier):
        return DVDis_Chu(
            self.v_meta, self.g.f * multiplier, self.h.f * multiplier, ndim=self.ndim
        )

    def __truediv__(self, divider):
        return DVDis_Chu(
            self.v_meta, self.g.f / divider, self.h.f / divider, ndim=self.ndim
        )

    @property
    def v(self):
        return self.v_meta.v

    @property
    def v2(self):
        return self.v_meta.v2

    @property
    def v3(self):
        return self.v_meta.v3

    @property
    def w(self):
        return self.v_meta.w

    def density(self, g=None, h=None):
        if g is None:
            g = self.g.f
        if h is None:
            h = self.h.f
        return self.g.density(f=g)

    def velocity(self, g=None, h=None):
        if g is None:
            g = self.g.f
        if h is None:
            h = self.h.f
        return self.g.velocity(f=g)

    def temperature(self, g=None, h=None):
        if g is None:
            g = self.g.f
        if h is None:
            h = self.h.f
        temp = (
            self.g.temperature(f=g) + 2 * self.h.density(f=h) / self.g.density(f=g)
        ) / self.ndim
        return temp

    def rho_u_theta(self, g=None, h=None):
        rho = self.density()
        u = self.velocity()
        T = self.temperature()
        return rho, u, T

    def maxwellian(self, g=None, h=None, rho_u_theta=None):
        if rho_u_theta is None:
            if g is None:
                g = self.g.f
            if h is None:
                h = self.h.f
            rho, u, T = self.rho_u_theta(g=g, h=h)
        else:
            rho, u, T = rho_u_theta
        gm = self.g.maxwellian(rho_u_theta=(rho, u, T))
        hm = gm * T * (self.ndim - self.g.ndim) / 2
        return DVDis_Chu(self.v_meta, gm, hm)