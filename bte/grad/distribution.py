from __future__ import annotations
from typing import Callable, Tuple, List
import torch
import torch.nn.functional as F
import torch.nn as nn
from math import factorial, gamma, sqrt
import math
import numpy as np
import abc
from bte.utils.indexs import *
from bte.utils.operators import make_tensor

class distributionBase(metaclass=abc.ABCMeta):
    """
    The base class for gas distribution.
    """
    def __init__(self):
        pass

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


class HermitedistributionBase():
    def __init__(self):
        pass

    @abc.abstractmethod
    def down(self):
        pass

    @abc.abstractmethod
    def new(self, *args, **kw):
        pass


class HermiteDis(HermitedistributionBase):
    """
    HermiteDis: Gas distribution in Hermite form.
    """

    def __init__(self, u:torch.Tensor, theta:torch.Tensor, coef:torch.Tensor):
        self.u = u
        self.theta = theta
        self.coef = coef
        
    def new(self, u, theta, coef, *args, **kw):
        return HermiteDis(u, theta, coef)

    def __str__(self) -> str:
        return "u:"+self.u.__str__()+"\n"+"theta:"+self.theta.__str__()+"\n"+"coef:"+self.coef.__str__()

    @staticmethod
    def empty(order: int,device='cpu') -> "HermiteDis":
        u = torch.empty((0, 1),device=device)
        theta = torch.empty((0, 1),device=device)
        coef = torch.empty((0, order),device=device)
        return HermiteDis(u, theta, coef)

    def from_DVDis(self, dis: "DVDis", u=None,theta=None) -> "HermiteDis":
        if u is None:
            u = dis.velocity()
        if theta is None:
            theta = dis.temperature()
        sqr_T = torch.sqrt(theta)
        lenv = dis.v.shape[0]
        # print(u.shape,theta.shape,dis.v.shape,u.shape[:-1]+(lenv,))

        c_v = (dis.v[...,0] - u.expand(u.shape[:-1] + (lenv,))) / sqr_T.expand(
            sqr_T.shape[:-1] + (lenv,)
        )
        # print(u.shape[:-1] + (lenv,))
        # print(c_v.shape)
        M = self.ORDER
        mom = torch.empty(dis.f.shape[:-1] + (M,),device=u.device)
        for i in range(M):
            #print(dis.f.shape, eval_hermitenorm(i, c_v).shape,sqr_T.shape)
            #print(dis.sum(dis.f * eval_hermitenorm(i, c_v)).shape)
            tmp = (
                dis.sum(dis.f * eval_hermitenorm(i, c_v))
                / factorial(i)
                * sqr_T ** i
            )
            #print(tmp.shape)
            mom[..., i : i + 1] = tmp
        return HermiteDis(u, theta, mom)

    def cpu(self):
        return HermiteDis(self.u.cpu(), self.theta.cpu(), self.coef.cpu())

    def cuda(self):
        return HermiteDis(self.u.cuda(), self.theta.cuda(), self.coef.cuda())

    def to(self, args):
        return HermiteDis(self.u.to(args), self.theta.to(args), self.coef.to(args))

    @property
    def ORDER(self):
        return self.coef.shape[-1]

    @property
    def rho(self):
        return self.coef[..., 0:1]

    def to_tensor(self):
        return torch.cat([self.u, self.theta, self.coef], dim=-1)

    @property
    def moments(self):
        u, theta, coef = self.u, self.theta, self.coef
        c = u
        s2 = theta
        v0 = coef[..., 0:1]
        v1 = c * v0 + coef[..., 1:2]
        tmp1 = c * v1
        tmp2 = c * c
        tmp3 = coef[..., 2:3]
        v2 = tmp1 + tmp3 + 0.5 * (1.0 * s2 - tmp2) * v0
        return v0, v1, v2

    def density(self):
        return self.coef[...,0:1]

    def velocity(self):
        return self.u

    def temperature(self):
        return self.theta

    @property
    def rho_u_theta(self):
        m0, m1, m2 = self.moments
        v0 = m0
        v1 = m1 / m0
        v2 = 2.0 * m2 / v0 - v1 * v1
        return v0, v1, v2

    def as_tuple(self):
        return self.u, self.theta, self.coef

    def __add__(self, anotherDis):
        return HermiteDis(self.u, self.theta, self.coef + anotherDis.coef)

    def __radd__(self, anotherDis):
        return HermiteDis(self.u, self.theta, self.coef + anotherDis.coef)

    def __sub__(self, anotherDis):
        return HermiteDis(self.u, self.theta, self.coef - anotherDis.coef)

    def __mul__(self, multiplier):
        return HermiteDis(self.u, self.theta, multiplier * self.coef)

    def __rmul__(self, multiplier):
        return HermiteDis(self.u, self.theta, multiplier * self.coef)

    def __truediv__(self, divider):
        return HermiteDis(self.u, self.theta, self.coef / divider)

    def roll(self, idx):
        return HermiteDis(
            d3roll(self.u, idx), d3roll(self.theta, idx), d3roll(self.coef, idx)
        )

    def clone(self):
        return HermiteDis(self.u.clone(), self.theta.clone(), self.coef.clone())

    def mulvec(self):
        tmp = torch.zeros_like(self.coef)
        coef = self.coef
        ind = torch.arange(coef.shape[2] - 1, device=coef.device) + 1
        ind = ind.reshape([1, 1, coef.shape[2] - 1])
        tmp[..., 0:] = tmp[..., 0:] + coef[..., 0:] * self.u
        tmp[..., 1:] = tmp[..., 1:] + coef[..., 0:-1] * self.theta
        tmp[..., 0:-1] = tmp[..., 0:-1] + coef[..., 1:] * ind
        return HermiteDis(self.u, self.theta, tmp)

    def down(self):
        return HermiteDis(self.u, self.theta, self.coef[..., :-1])

    def Project_to_STD(self, STEP: int = 1):
        """
        Project_to_STD 将分布参数重设，重新投影到最佳参数上
        """
        _, u, theta = self.rho_u_theta
        dis = Project_RK4(self, u, theta, STEP=STEP)
        return dis

    def Project_2B(self, dest: "HermiteDis", STEP: int = 1) -> "HermiteDis":
        """
        Project_A2B 将分布投影到dest对应的参数上
        """
        u, theta = dest.u, dest.theta
        dis = Project_RK4(self, u, theta, STEP=STEP)
        return dis


def d3roll(tensor, idx):
    return torch.cat((tensor[..., idx:, :], tensor[..., :idx, :]), -2)


def Project_RK4_helper(dis, u0, theta0, u, theta):
    tmp = (u0 - u) * F.pad(dis[..., :-1], (1, 0)) + 0.5 * (theta0 - theta) * F.pad(
        dis[..., :-2], (2, 0)
    )
    return tmp


def Project_RK4_step(dis, u0, theta0, u, theta, h: float):
    k1 = Project_RK4_helper(dis, u0, theta0, u, theta)
    k2 = Project_RK4_helper(dis + h / 2 * k1, u0, theta0, u, theta)
    k3 = Project_RK4_helper(dis + h / 2 * k2, u0, theta0, u, theta)
    k4 = Project_RK4_helper(dis + h * k3, u0, theta0, u, theta)
    dis = dis + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * h / 6.0
    return dis


def Project_RK4(dis: HermiteDis, u, theta, STEP: int = 1):
    """Project_RK4 使用R-K4格式对分布函数进行投影

    Parameters
    ----------
    dis : torch.Tensor
        分布函数
    u : torch.Tensor
        分布中心
    theta : torch.Tensor
        分布宽度
    STEP : int, optional
        RK方法的步数, by default 1
    Returns
    -------
    torch.Tensor
        投影后的分布函数
    """
    h = 1.0 / STEP
    tmpslice = dis.coef
    u0 = dis.u
    theta0 = dis.theta
    for i in range(STEP):
        tmpslice = Project_RK4_step(tmpslice, u0, theta0, u, theta, h)
    return HermiteDis(u, theta, tmpslice)


def Project_RK4_helperT(dis, u0, theta0, u, theta):
    tmp = (u0 - u) * F.pad(dis[..., 1:], (0, 1)) + 0.5 * (theta0 - theta) * F.pad(
        dis[..., 2:], (0, 2)
    )
    return tmp


def Project_RK4_stepT(dis, u0, theta0, u, theta, h: float = 1):
    k1 = Project_RK4_helperT(dis, u0, theta0, u, theta)
    k2 = Project_RK4_helperT(dis + h / 2 * k1, u0, theta0, u, theta)
    k3 = Project_RK4_helperT(dis + h / 2 * k2, u0, theta0, u, theta)
    k4 = Project_RK4_helperT(dis + h * k3, u0, theta0, u, theta)
    dis = dis + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * h / 6.0
    return dis


def Project_RK4T(dis, u0, theta0, u, theta, STEP: int = 1):
    h = 1.0 / STEP
    tmpslice = dis
    for i in range(STEP):
        tmpslice = Project_RK4_stepT(tmpslice, u0, theta0, u, theta, h)
    return tmpslice


def eval_hermitenorm(n: int, x: torch.Tensor):
    if n < 0:
        return float("nan") * torch.ones_like(x)
    elif n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return x
    else:
        y3 = torch.zeros_like(x)
        y2 = torch.ones_like(x)
        for k in range(n, 1, -1):
            y1 = x * y2 - k * y3
            y3 = y2
            y2 = y1
        return x * y2 - y3

def eval_hermitenorm_seq(n: int, x: torch.Tensor):
    y=torch.zeros((n+1,)+x.shape)
    if n < 0:
        raise ValueError
    if n >= 0:
        y[0]=torch.ones_like(x)
    if n >= 1:
        y[1]=x
    if n >= 2:
        for j in range(2,n+1):
            y[j]=x*y[j-1]-(j-1)*y[j-2]
    return y

class DVDisMeta(nn.Module):
    def __init__(self, v, v_w, *args):
        super().__init__()
        self.register_buffer("v", v)
        self.register_buffer("v2", v ** 2)
        self.register_buffer("v3", v ** 3)
        self.register_buffer("w", v_w)

class DVDisMeta_Grid(DVDisMeta):
    def __init__(self, vL:List, v_wL:List, *args):
        nn.Module.__init__(self)
        assert len(vL)==len(v_wL)
        self.v_dims = nn.ModuleList([DVDisMeta(v,v_w) for v,v_w in zip(vL,v_wL)])
        self.dim = len(vL)
        V, W = product_meshgrid(vL,v_wL)
        self.register_buffer("v", V)
        self.register_buffer("v2", V ** 2)
        self.register_buffer("v3", V ** 3)
        self.register_buffer("w", W)


class DVDis(distributionBase):
    """
    The gas distribution for discrete velocity method.
    """

    def __init__(self, v_meta:DVDisMeta, data):
        super().__init__()
        self.v_meta = v_meta
        self.ndim = v_meta.v.shape[-1]
        self.f = data

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

    def cpu(self):
        return DVDis(self.v_meta.cpu(), self.f.cpu())

    def cuda(self):
        return DVDis(self.v_meta.cuda(), self.f.cuda())

    def to(self, args):
        return DVDis(self.v_meta.to(args), self.f.to(args))

    def __add__(self, anotherDis:DVDis):
        return DVDis(self.v_meta,self.f+anotherDis.f)

    def __radd__(self, anotherDis:DVDis):
        return DVDis(self.v_meta,self.f+anotherDis.f)

    def __sub__(self, anotherDis):
        return DVDis(self.v_meta,self.f-anotherDis.f)

    def __mul__(self, multiplier):
        return DVDis(self.v_meta,self.f*multiplier)

    def __rmul__(self, multiplier):
        return DVDis(self.v_meta,self.f*multiplier)

    def __truediv__(self, divider):
        return DVDis(self.v_meta,self.f/ divider)

    @staticmethod
    def empty(nv: int, vmin: float, vmax: float, method="uni",device='cpu') -> DVDis:
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
        v=v.to(device=device)
        w=w.to(device=device)
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

    def heatflux(self, f=None):
        if f is None:
            f = self.f
        m0 = self._m0(f)
        m1 = self._m1(f)
        m2 = self._m2(f)
        m3 = self._m3(f)
        u = m1 / m0
        q = (-(u ** 3) * m0 + 3 * (u ** 2) * m1 - 3 * u * m2 + m3)
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
        maxwellian=self.maxwellian(rho_u_theta = rho_u_theta)
        if sign > 0:
            maxwellian[..., self.v[...,0] < 0] = 0
            maxwellian[..., self.v[...,0] == 0] = maxwellian[..., self.v[...,0] == 0]/2
        elif sign < 0:
            maxwellian[..., self.v[...,0] > 0] = 0
            maxwellian[..., self.v[...,0] == 0] = maxwellian[..., self.v[...,0] == 0]/2
        else:
            raise ValueError("sign should be 1 or -1")
        return maxwellian

    def one_side(self, f=None, sign=0):
        if f is None:
            f=self.f
        f = f.clone()
        if sign == 1:
            f[..., self.v[...,0] < 0] = 0
            f[..., self.v[...,0] == 0] = f[..., self.v[...,0] == 0]/2
        elif sign == -1:
            f[..., self.v[...,0] > 0] = 0
            f[..., self.v[...,0] == 0] = f[..., self.v[...,0] == 0]/2
        else:
            raise ValueError("sign should be 1 or -1")
        return f

    def reverse(self,f=None):
        if f is None:
            f=self.f
        return torch.flip(f, [-1,])

    def from_HermiteDis(self, dis: HermiteDis) -> "DVDis":
        rho, u, theta = dis.rho_u_theta
        f = dis.coef
        v = (self.v[...,0] - u) / torch.sqrt(theta)
        h = 0
        sqr_T=theta.sqrt()
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
    #v1 = torch.tensor(v1, dtype=torch.get_default_dtype())
    #w1 = torch.tensor(w1, dtype=torch.get_default_dtype())
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

def ND_velocity(func:Callable[[int,int,int],Tuple]):
    def wrapper(nvT:Tuple, vminT:Tuple, vmaxT:Tuple):
        assert len(nvT)==len(vminT)
        assert len(nvT)==len(vmaxT)
        vL=[]
        wL=[]
        DIM=len(nvT)
        for d,nv,vmin,vmax in zip(range(DIM), nvT, vminT, vmaxT):
            v, w = func(nv, vmin, vmax)
            w=w.reshape([1,]*d+[-1,]+[1,]*(DIM-d-1))
            vL.append(v)
            wL.append(w)
        grids = torch.meshgrid(*[v.squeeze() for v in vL])
        V = torch.stack(grids, dim=-1)
        W = functools.reduce(lambda a,b:a*b,wL)
        V = V.reshape([-1, DIM])
        W = W.reshape([-1, 1])
        return V, W
    return wrapper

uniform_velocity_nd=ND_velocity(uniform_velocity)
legendre_velocity_nd=ND_velocity(legendre_velocity)

def velocity_list(func, nvT:Tuple, vminT:Tuple, vmaxT:Tuple):
    assert len(nvT)==len(vminT)
    assert len(nvT)==len(vmaxT)
    vL=[]
    wL=[]
    DIM=len(nvT)
    for d,nv,vmin,vmax in zip(range(DIM), nvT, vminT, vmaxT):
        v, w = func(nv, vmin, vmax)
        w=w.reshape([1,]*d+[-1,]+[1,]*(DIM-d-1))
        vL.append(v)
        wL.append(w) 
    return vL,wL

def product_meshgrid(vL:List, wL:List):
    assert len(vL)==len(wL)
    DIM = len(vL)
    grids = torch.meshgrid(*[v.squeeze() for v in vL])
    V = torch.stack(grids, dim=-1)
    W = functools.reduce(lambda a,b:a*b,wL)
    V = V.reshape([-1, DIM])
    W = W.reshape([-1, 1])
    return V, W

class DVDis_Chu(distributionBase):
    def __init__(self, v_meta, data_g, data_h, ndim=3):
        self.v_meta = v_meta
        self.ndim = ndim
        self.g = DVDis(v_meta, data_g)
        self.h = DVDis(v_meta, data_h)

    def __add__(self,another):
        if isinstance(another,DVDis_Chu):
            return DVDis_Chu(self.v_meta,self.g.f+another.g.f,self.h.f+another.h.f,ndim=self.ndim)
        else:
            return DVDis_Chu(self.v_meta,self.g.f+another,self.h.f+another,ndim=self.ndim)

    def __radd__(self,another):
        return DVDis_Chu(self.v_meta,self.g.f+another,self.h.f+another,ndim=self.ndim)

    def __sub__(self,another):
        return DVDis_Chu(self.v_meta,self.g.f-another.g.f,self.h.f-another.h.f,ndim=self.ndim)

    def __mul__(self,multiplier):
        return DVDis_Chu(self.v_meta,self.g.f*multiplier,self.h.f*multiplier,ndim=self.ndim)

    def __rmul__(self,multiplier):
        return DVDis_Chu(self.v_meta,self.g.f*multiplier,self.h.f*multiplier, ndim=self.ndim)

    def __truediv__(self,divider):
        return DVDis_Chu(self.v_meta,self.g.f/divider,self.h.f/divider, ndim=self.ndim)

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
        hm = gm * T *(self.ndim-self.g.ndim)/2
        return DVDis_Chu(self.v_meta, gm, hm)


class HermiteDisND(HermitedistributionBase):
    """
    HermiteDis: Gas distribution in Hermite form.
    """

    def __init__(self, u:torch.Tensor, theta:torch.Tensor, coef:torch.Tensor, indt:index_table):
        self.u = u
        self.theta = theta
        self.coef = coef
        self.indt = indt
        self.D = indt.DIM
        self.DIM = self.D
        
    def new(self,u,theta,coef,indt=None):
        if indt is None:
            indt=self.indt
        return HermiteDisND(u,theta,coef,indt)
        
    def M(self):
        return self.indt.M

    def __str__(self) -> str:
        return "u:"+self.u.__str__()+"\n"+"theta:"+self.theta.__str__()+"\n"+"coef:"+self.coef.__str__()

    @staticmethod
    def empty(order: int,dim=3, device='cpu') -> "HermiteDisND":
        indt = index_table(order,dim)
        u = torch.empty((0, dim),device=device)
        theta = torch.empty((0, 1),device=device)
        coef = torch.empty((0, len(indt.iNto1)),device=device)
        return HermiteDisND(u, theta, coef, indt)

    def from_DVDis(self, dis: "DVDis", u=None,theta=None) -> "HermiteDis":
        pass

    def cpu(self):
        return HermiteDisND(self.u.cpu(), self.theta.cpu(), self.coef.cpu(), self.indt)

    def cuda(self):
        return HermiteDisND(self.u.cuda(), self.theta.cuda(), self.coef.cuda(), self.indt)

    def to(self, args):
        return HermiteDisND(self.u.to(args), self.theta.to(args), self.coef.to(args), self.indt)

    @property
    def ORDER(self):
        return self.indt.ORDER

    @property
    def rho(self):
        return self.coef[..., 0:1]

    def to_tensor(self):
        return torch.cat([self.u, self.theta, self.coef], dim=-1)

    def density(self):
        return self.coef[...,0:1]

    def velocity(self):
        return self.u

    def temperature(self):
        return self.theta
    
    @property
    def moments(self):
        # TODO: 返回0、1、2阶矩，其中0阶矩有1个，1阶矩有3个，2阶矩有9个
        u, theta, coef = self.u, self.theta, self.coef
        v0 = coef[..., 0:1]
        f1=coef[..., 1:self.DIM+1]
        v1 = u * v0 + f1
        u2 = u * u
        v2 = (self.DIM * theta + u2.sum(dim=-1,keepdim=True))*v0+2*(u*f1).sum(dim=-1,keepdim=True)+2*(
           coef[...,self.indt.v2].sum(dim=-1,keepdim=True))
        v2 = v2 / 2
        return v0, v1, v2

    @property
    def rho_u_theta(self):
        m0, m1, m2 = self.moments
        v0 = m0
        v1 = m1 / m0
        v2 = (2.0 * m2 / v0 - (v1 * v1).sum(dim=-1,keepdim=True))/self.D
        return v0, v1, v2

    def as_tuple(self):
        return self.u, self.theta, self.coef

    def __add__(self, anotherDis):
        return HermiteDisND(self.u, self.theta, self.coef + anotherDis.coef, self.indt)

    def __radd__(self, anotherDis):
        return HermiteDisND(self.u, self.theta, self.coef + anotherDis.coef, self.indt)

    def __sub__(self, anotherDis):
        return HermiteDisND(self.u, self.theta, self.coef - anotherDis.coef, self.indt)

    def __mul__(self, multiplier):
        return HermiteDisND(self.u, self.theta, multiplier * self.coef, self.indt)

    def __rmul__(self, multiplier):
        return HermiteDisND(self.u, self.theta, multiplier * self.coef, self.indt)

    def __truediv__(self, divider):
        return HermiteDisND(self.u, self.theta, self.coef / divider, self.indt)

    def roll(self, idx):
        return HermiteDisND(
            d3roll(self.u, idx), d3roll(self.theta, idx), d3roll(self.coef, idx), self.intd
        )

    def clone(self):
        return HermiteDisND(self.u.clone(), self.theta.clone(), self.coef.clone(), self.indt)

    def down(self):
        indt=index_tables.get(self.indt.ORDER-1,self.indt.DIM)
        return HermiteDisND(self.u, self.theta, self.coef[..., :indt.len], indt)

    def Project_to_STD(self, STEP: int = 1):
        """
        Project_to_STD 将分布参数重设，重新投影到最佳参数上
        """
        _, u, theta = self.rho_u_theta
        dis = self.Project_RK4(u, theta, STEP=STEP)
        return dis

    def Project_2B(self, dest: "HermiteDisND", STEP: int = 1) -> "HermiteDis":
        """
        Project_A2B 将分布投影到dest对应的参数上
        """
        u, theta = dest.u, dest.theta
        dis = self.Project_RK4(u, theta, STEP=STEP)
        return dis

    def mulvec(self, axis = 0):
        """axis=0 for x, 1 for y, 2 for z.
        """
        tmp = torch.zeros_like(self.coef)
        
        coef = self.coef
        ind= self.indt.m1[axis]
        #ind = torch.tensor(self.indt.get_order(self.indt.u1[axis][0].tolist(),dim=axis),device=tmp.device) + 1 # 1,2,....,
        ind = ind.reshape([1, 1, -1]).to(device=coef.device)
        #print(ind.shape)
        
        tmp[..., :] = tmp[..., :] + coef[..., :] * self.u[...,axis:axis+1]
        tmp[..., self.indt.d1[axis][0]] = tmp[..., self.indt.d1[axis][0]] + coef[..., self.indt.d1[axis][1]] * self.theta
        tmp[..., self.indt.u1[axis][0]] = tmp[..., self.indt.u1[axis][0]] + coef[..., self.indt.u1[axis][1]] * ind
        
        return HermiteDisND(self.u, self.theta, tmp,self.indt)
    

    def Project_RK4_helper(self, dis, u0, theta0, u, theta):
        tmp=[torch.zeros_like(dis) for d in range(self.DIM)]
        tmpt=[torch.zeros_like(dis) for d in range(self.DIM)]
        for d in range(self.DIM):
            #print(self.indt.d2[d][0],self.indt.d2[d][1])
            #print(tmp[0].shape)
            tmp[d][...,self.indt.d1[d][0]]=dis[...,self.indt.d1[d][1]]
            tmpt[d][...,self.indt.d2[d][0]]=dis[...,self.indt.d2[d][1]]
        ud = (u0 - u)
        #print(ud.shape,tmp[0].shape,theta0.shape,theta.shape,tmpt[0].shape)
        tmp = sum([ud[...,d:d+1]*tmp[d] for d in range(self.DIM)]) + 0.5 * (theta0 - theta) * sum(tmpt)
        return tmp

    def Project_RK4_step(self, dis, u0, theta0, u, theta, h: float):
        k1 = self.Project_RK4_helper(dis, u0, theta0, u, theta)
        k2 = self.Project_RK4_helper(dis + h / 2 * k1, u0, theta0, u, theta)
        k3 = self.Project_RK4_helper(dis + h / 2 * k2, u0, theta0, u, theta)
        k4 = self.Project_RK4_helper(dis + h * k3, u0, theta0, u, theta)
        dis = dis + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * h / 6.0
        return dis

    def Project_RK4(self, u, theta, STEP: int = 1):
        """Project_RK4 使用R-K 4格式对分布函数进行投影

        Parameters
        ----------
        dis : torch.Tensor
            分布函数
        u : torch.Tensor
            分布中心
        theta : torch.Tensor
            分布宽度
        STEP : int, optional
            RK方法的步数, by default 1
        Returns
        -------
        torch.Tensor
            投影后的分布函数
        """
        h = 1.0 / STEP
        tmpslice = self.coef
        u0 = self.u
        theta0 = self.theta
        for i in range(STEP):
            tmpslice = self.Project_RK4_step(tmpslice, u0, theta0, u, theta, h)
        return HermiteDisND(u, theta, tmpslice, self.indt)