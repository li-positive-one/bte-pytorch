from functools import partial
from math import factorial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bte.dvm import distribution
from bte.dvm.distribution import DVDisMeta_Grid
from bte.dvm.distribution import DVDis, DVDisMeta
from bte.utils import IMEX_RK as IMEX_RK
from bte.utils.limiter import *
from bte.utils.weno5 import *
from bte.dvm.collision import collisioner,collisioner_bgk
from bte.utils.specials import eval_hermitenorm

EPS = 1e-8

def F_pm_2(f, vp, vm, dx, dt, limiter=limiter_minmod):
    df = f[..., 1:, :] - f[..., :-1, :]
    rp = (df[..., :-1, :]) / (df[..., 1:, :] + EPS)
    rm = (df[..., 1:, :]) / (df[..., :-1, :] + EPS)
    phi_p = limiter(rp)
    phi_m = limiter(rm)
    Fm = torch.empty_like(f)
    Fp = torch.empty_like(f)

    Fp[..., 1:-1, :] = 0.5 * phi_p * (1.0) * (df[..., 1:, :])
    Fm[..., 1:-1, :] = 0.5 * phi_m * (-1.0) * (df[..., :-1, :])

    return (
        (df[..., 1:-2, :] + Fp[..., 2:-2, :] - Fp[..., 1:-3, :]) / dx,
        (df[..., 2:-1, :] + Fm[..., 3:-1, :] - Fm[..., 2:-2, :]) / dx,
    )


def F_plm(f, vp, vm, dx, dt, limiter=limiter_minmod):
    df = f[..., 1:, :] - f[..., :-1, :]
    rp = (df[..., :-1, :]) / (df[..., 1:, :] + EPS)
    rm = (df[..., 1:, :]) / (df[..., :-1, :] + EPS)
    phi_p = limiter(rp)
    phi_m = limiter(rm)
    Fm = torch.empty_like(f)
    Fp = torch.empty_like(f)

    Fp[..., 1:-1, :] = 0.5 * phi_p * (1.0 - vp * dt / dx) * (df[..., 1:, :])
    Fm[..., 1:-1, :] = 0.5 * phi_m * (-1.0 - vm * dt / dx) * (df[..., :-1, :])

    return (
        (df[..., 1:-2, :] + Fp[..., 2:-2, :] - Fp[..., 1:-3, :]) / dx,
        (df[..., 2:-1, :] + Fm[..., 3:-1, :] - Fm[..., 2:-2, :]) / dx,
    )

def linear_reconstruction_VanLeer(fL:torch.Tensor, f:torch.Tensor, fR:torch.Tensor)->torch.Tensor:
    """linear_reconstruction using VanLeer limiter

    Args:
        fL (torch.Tensor): left cell 
        f (torch.Tensor): cell
        fR (torch.Tensor): right cell

    Returns:
        torch.Tensor: leftRec, rightRec
    """
    theta1 = f - fL
    theta2 = fR - f
    slope = (theta1.sign()+theta2.sign())*theta1.abs()*theta2.abs()/(theta1.abs()+theta2.abs()+1e-8)
    RecL = f - slope / 2 
    RecR = f + slope / 2 
    return RecL, RecR

def linear_reconstruction_MinMod(fL:torch.Tensor, f:torch.Tensor, fR:torch.Tensor)->torch.Tensor:
    """linear_reconstruction using MinMod limiter

    Args:
        fL (torch.Tensor): left cell 
        f (torch.Tensor): cell
        fR (torch.Tensor): right cell

    Returns:
        torch.Tensor: leftRec, rightRec
    """
    theta1 = f - fL
    theta2 = fR - f
    slope = (theta1.sign()+theta2.sign())/2*torch.minimum(theta1.abs(),theta2.abs())
    RecL = f - slope / 2 
    RecR = f + slope / 2 
    return RecL, RecR

def linear_reconstruction(fL, f, fR, limiter=limiter_minmod):
    theta = torch.zeros_like(f)
    theta1 = f - fL
    theta2 = fR - f
    mask = torch.abs(theta1 * theta2) > 1e-12
    theta[mask] = (theta1 / (theta2 + 1e-12))[mask]

    theta = limiter(theta)
    RecL = f - theta / 2 * theta2
    RecR = f + theta / 2 * theta2

    return RecL, RecR

def get_reflecting_bdv(dis:DVDis, BV, xi, side, no_linear=False):
    """[summary]

    Args:
        dis (DVDis): [description]
        BV ([type]): [description]
        xi ([type]): [description]
        side ([type]):side=-1 for left BV, side=1 for right BV
        no_linear (bool, optional): [description]. Defaults to False.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    if side == -1:
        fl = dis.f
        fbl_toleft = dis.one_side(fl, -1)
        fbl_toright = dis.reverse(fbl_toleft)
        fbl_maxwellian = dis.maxwellian_half(rho_u_theta=BV, sign=1)
        # fbl_maxwellian的动量应该和fbl_toleft相等。
        fbl_maxwellian_U = dis.rho_u_theta(fbl_maxwellian)
        fbl_toleft_U = dis.rho_u_theta(fbl_toleft)
        c = (
            (
                (fbl_toleft_U[0] * fbl_toleft_U[1])
                / (fbl_maxwellian_U[0] * fbl_maxwellian_U[1])
            )
            .abs()
        )
        fbl = (
            xi * c * fbl_maxwellian + (1 - xi) * fbl_toright + fbl_toleft
        )
        if no_linear:
            fbll=fbl
        else:
            fbll=2*fbl-fl
        return DVDis(dis.v_meta,fbll)
    elif side == 1:
        fr = dis.f
        fbr_toright = dis.one_side(fr, 1)
        fbr_toleft = dis.reverse(fbr_toright)
        fbr_maxwellian = dis.maxwellian_half(rho_u_theta=BV, sign=-1)
        # fbl_maxwellian的动量应该和fbl_toleft相等。
        fbr_maxwellian_U = dis.rho_u_theta(fbr_maxwellian)
        fbr_toright_U = dis.rho_u_theta(fbr_toright)
        c = (
            (
                (fbr_toright_U[0] * fbr_toright_U[1])
                / (fbr_maxwellian_U[0] * fbr_maxwellian_U[1])
            )
            .abs()
        )
        fbr = (
            xi * c * fbr_maxwellian + (1 - xi) * fbr_toleft + fbr_toright
        )
        if no_linear:
            fbrr=fbr
        else:
            fbrr=2*fbr-fr
        return DVDis(dis.v_meta,fbrr)
    else:
        raise ValueError


class Grid_onDVDis_NU(DVDis):
    """A grid class that stores the details and solution of the
    computational grid."""

    def __init__(self, xmin, xmax, nx, vmin, vmax, nv, v_discrete="uni", device="cpu"):
        self.xmin = xmin
        self.xmax = xmax
        self.vmin = vmin
        self.vmax = vmax
        self.nx, self.nv = nx, nv
        dx = float(self.xmax - self.xmin) / self.nx
        self.v_discrete=v_discrete
        x = torch.arange(
            start = self.xmin + dx / 2, end = self.xmax + dx / 2, step=dx
        )
        #self.dx = dx * torch.ones(nx,1)
        # self.dx = self.dx.to(device=device)
        # self.x = x.to(device=device)


        if isinstance(nv, int):# If single dimension
            if v_discrete == "leg":
                v, w = distribution.legendre_velocity(nv, vmin, vmax)
            elif v_discrete == "uni":
                v, w = distribution.uniform_velocity(nv, vmin, vmax)
            else:
                raise ValueError
            # v = v.to(device=device)
            # w = w.to(device=device)
            v_meta=DVDisMeta(v, w)
        else:# If multiple dimension
            if v_discrete == "leg":
                vL, wL = distribution.velocity_list(distribution.legendre_velocity,nv, vmin, vmax)
            elif v_discrete == "uni":
                vL, wL = distribution.velocity_list(distribution.uniform_velocity,nv, vmin, vmax)
            else:
                raise ValueError
            v_meta=DVDisMeta_Grid(vL, wL)
        super().__init__(v_meta, None)

        self.register_buffer("x",x)
        self.register_buffer("dx",dx * torch.ones(nx,1))

    # def cuda(self):
    #     self.v_meta=self.v_meta.cuda()
    #     self.dx=self.dx.cuda()
    #     self.x=self.x.cuda()
    #     return self



# BGKSolver with MUSL flux (一种DVM方法)
class BGKSolver(nn.Module):
    def __init__(
        self,
        xmin,
        xmax,
        nx,
        vmin,
        vmax,
        nv,
        v_discrete="uni",
        BC_type="constant",
        bgk_simple_kn="simple",
        device="cpu",
    ):
        super().__init__()
        self.kn = np.nan

        self.set_BC(BC_type)
        self.BC = BC_type
        self.set_space_order(2)
        self.set_time_stepper("bgk-RK2")
        self.bgk_simple_kn = bgk_simple_kn
        self.dis = Grid_onDVDis_NU(
            xmin, xmax, nx, vmin, vmax, nv, v_discrete=v_discrete, device=device
        )

        abs_v = torch.abs(self.dis.v)
        self.register_buffer("_v_p", 0.5 * (self.dis.v + abs_v))
        self.register_buffer("_v_m", 0.5 * (self.dis.v - abs_v))


    @property
    def f(self):
        return self.dis.f

    @f.setter
    def f(self, value):
        self.dis.f = value

    def set_order(self, space_order, time_order, time_stiff=False):
        if space_order == "plm" and (time_order != 1 or time_stiff):
            raise ValueError("plm only suit with RK1")
        self.set_space_order(space_order)
        self.set_time_order(time_order, time_stiff)

    def set_space_order(self, order):
        if order == 1:
            self._grad_vf = self._grad_vf_constant
        elif order == 2:
            self._grad_vf = self._grad_vf_2
        elif order == 5:
            self._grad_vf = self._grad_vf_weno
        elif order == "plm":
            self._grad_vf = self._grad_vf_plm
        else:
            raise ValueError

    def set_time_stepper(self,method:"str"):
        """set time stepper

        Args:
            method (str): can be "bgk-RK1","bgk-RK2","bgk-RK3","bgk-RKs1","bgk-RKs2"
            "Euler", "IMEX-1", "IMEX-2", "IMEX"
        """
        bgk_RK={
            "bgk-RK1":IMEX_RK.IMEX_RK1(),
            "bgk-RK2":IMEX_RK.IMEX_RK2(),
            "bgk-RK3":IMEX_RK.IMEX_RK3(),
            "bgk-RK1s":IMEX_RK.IMEX_RK1_stiff(),
            "bgk-RK2s":IMEX_RK.IMEX_RK2_stiff()
        }
        if method in ["bgk-RK1","bgk-RK2","bgk-RK3","bgk-RK1s","bgk-RK2s"]:
            self.step=self.step_rk
            self.RK=bgk_RK[method]
        elif method=="Euler":
            self.step=self.step_euler
        elif method=="IMEX":
            self.step=self.step_imex
        elif method=="IMEX-1":
            self.step=self.step_imex1
        else:
            raise ValueError

    def set_collisioner(self,collision_type:str,**args):
        """set collision type

        Args:
            collision_type (str): "BGK" or "FFT"

        Raises:
            ValueError: _description_
        """
        if collision_type=="FFT":
            self.collision_type="FFT"
            self.coller=collisioner(self.dis.v_meta, quad_num=64, **args)
        elif collision_type=="BGK":
            self.collision_type="BGK"
            self.coller=collisioner_bgk(self.dis.v_meta,**args)
        else:
            raise ValueError

    def set_initial(self, kn, f0):
        if isinstance(kn, torch.Tensor):
            self.kn = kn
        else:
            self.kn = kn * torch.ones(f0.shape[:-1]+(1,),device=f0.device).expand_as(f0)
        self.dis.f = f0.clone()
        self.left_BC = self.dis.f[..., 0:1, :].clone()
        self.right_BC = self.dis.f[..., -1:, :].clone()

    def set_BC(self, BC):
        self.BC = BC
        if BC == "periodic":
            self.apply_BC = self.padding_periodic
        elif BC == "cauchy":
            self.apply_BC = self.padding_replicate
        elif BC == "reflecting":
            #self.apply_BC = self.padding_reflecting_linear
            self.apply_BC = self.padding_replicate
            #self.xi = 1.0
        elif BC == "constant":
            self.apply_BC = self.padding_constant
        elif BC == "evaporation":
            self.apply_BC = self.padding_evaporation
        elif BC == "evaporation2":
            self.apply_BC = self.padding_evaporation2
        else:
            raise NotImplementedError

    def set_BC_Value(self, LV, RV):
        self.LV = LV  # LV=[rho,u,theta] #shape [[B,1,1],[B,1,1],[B,1,1]]
        self.RV = RV  # RV=[rho,u,theta] #shape [[B,1,1],[B,1,1],[B,1,1]]

    def padding_periodic(self, f):
        f = f.transpose(-1, -2)
        return F.pad(
            f, pad=(self.padding_order, self.padding_order), mode="circular"
        ).transpose(-1, -2)

    def padding_replicate(self, f):
        f = f.transpose(-1, -2)
        return F.pad(
            f, pad=(self.padding_order, self.padding_order), mode="replicate"
        ).transpose(-1, -2)

    def padding_constant(self, f):
        f = f.transpose(-1, -2)
        fbc = F.pad(
            f, pad=(self.padding_order, self.padding_order), mode="constant"
        ).transpose(-1, -2)
        fbc[..., 0 : self.padding_order, :] = self.left_BC
        fbc[..., -self.padding_order :, :] = self.right_BC
        return fbc

    def get_reflecting_bdv(self, fb, side):
        return get_reflecting_bdv(DVDis(self.dis.v_meta,fb),
                                (self.LV if side<0 else self.RV),
                                self.xi,side,no_linear=True).f
        #print("falg")
        # side=-1 for left boundary, side=+1 for right boundary
        # if side == -1:
        #     fl = fb
        #     fbl_toleft = self.dis.one_side(fl, -1)
        #     fbl_toright = self.dis.reverse(fbl_toleft)
        #     fbl_maxwellian = self.dis.maxwellian_half(rho_u_theta=self.LV, sign=1)
        #     # fbl_maxwellian的动量应该和fbl_toleft相等。
        #     # print(fbl_maxwellian.shape,fbl_toleft.shape)
        #     fbl_maxwellian_U = self.dis.rho_u_theta(fbl_maxwellian)
        #     fbl_toleft_U = self.dis.rho_u_theta(fbl_toleft)
        #     c = (
        #         (
        #             (fbl_toleft_U[0] * fbl_toleft_U[1])
        #             / (fbl_maxwellian_U[0] * fbl_maxwellian_U[1])
        #         )
        #         .abs()
        #     )
        #     fbl = (
        #         self.xi * c * fbl_maxwellian + (1 - self.xi) * fbl_toright + fbl_toleft
        #     )
        #     return fbl
        # elif side == 1:
        #     fr = fb
        #     fbr_toright = self.dis.one_side(fr, 1)
        #     fbr_toleft = self.dis.reverse(fbr_toright)
        #     fbr_maxwellian = self.dis.maxwellian_half(rho_u_theta=self.RV, sign=-1)
        #     # fbl_maxwellian的动量应该和fbl_toleft相等。
        #     fbr_maxwellian_U = self.dis.rho_u_theta(fbr_maxwellian)
        #     fbr_toright_U = self.dis.rho_u_theta(fbr_toright)
        #     c = (
        #         (
        #             (fbr_toright_U[0] * fbr_toright_U[1])
        #             / (fbr_maxwellian_U[0] * fbr_maxwellian_U[1])
        #         )
        #         .abs()
        #     )
        #     fbr = (
        #         self.xi * c * fbr_maxwellian + (1 - self.xi) * fbr_toleft + fbr_toright
        #     )
        #     return fbr
        # else:
        #     raise ValueError

    def getTau(self,U):
        if self.bgk_simple_kn=="simple":
            tau= self.kn 
        elif self.bgk_simple_kn=="simple2":
            tau= self.kn / U[0] 
        elif self.bgk_simple_kn=="mu_ref":
            tau=  self.mu_ref*2/(U[2]*2)**(1-self.omega)/U[0]
        else:
            raise ValueError
        return tau

    def update(self, qP, qN):
        if self.BC == "reflecting":
            fl = qN[..., 1:2, :]
            fbl = self.get_reflecting_bdv(fl, -1)
            qP[..., 0:1, :] = fbl

            fr = qP[..., -2:-1, :]
            fbr = self.get_reflecting_bdv(fr, 1)
            qN[..., -1:, :] = fbr
        return qP, qN

    def padding_reflecting_linear(self, f):
        if self._grad_vf == self._grad_vf_constant:
            return self.padding_replicate(f)
        # f in shape [B,W,V]
        fbl = f[:, 0:1, :]
        fblr = f[:, 1:2, :]
        fbr = f[:, -1:, :]
        fbrl = f[:, -2:-1, :]

        fblp = [fbl + (i + 1) * (fbl - fblr) for i in range(self.padding_order)]
        fbrp = [fbr + (i + 1) * (fbr - fbrl) for i in range(self.padding_order)]

        fbc = torch.cat(fblp[::-1] + [f,] + fbrp, dim=-2)

        return fbc

    def padding_reflecting(self, f):
        # f in shape [B,W,V]
        fbl = f[:, 0:1, :]
        fbr = f[:, -1:, :]

        # calculate fbl
        fl = f[:, 0:1, :]
        fbl = self.get_reflecting_bdv(fl, -1)

        # calculate fbl
        fr = f[:, -1:, :]
        fbr = self.get_reflecting_bdv(fr, 1)

        fbc = torch.cat(
            [fbl,] * self.padding_order + [f,] + [fbr,] * self.padding_order, dim=1
        )
        return fbc

    def padding_evaporation(self,f):
        fbl = f[:, 0:1, :]
        fblr = f[:, 1:2, :]
        fbr = f[:, -1:, :]
        fbrl = f[:, -2:-1, :]

        fblp_gas = [fbl + (i + 1) * (fbl - fblr) for i in range(self.padding_order)]
        fbrp_gas = [fbr + (i + 1) * (fbr - fbrl) for i in range(self.padding_order)]

        fbl_vapor=self.dis.maxwellian(rho_u_theta=self.LV)
        fbr_vapor=self.dis.maxwellian(rho_u_theta=self.RV)

        fblp=[g*(self.dis.v[...,0]<0)+fbl_vapor*(self.dis.v[...,0]>=0) for g in fblp_gas]
        fbrp=[g*(self.dis.v[...,0]>0)+fbr_vapor*(self.dis.v[...,0]<=0) for g in fbrp_gas]

        fbc = torch.cat(fblp[::-1] + [f,] + fbrp, dim=1)
        return fbc

    def padding_evaporation2(self,f):
        fbl = f[:, 0:1, :]
        fblr = f[:, 1:2, :]
        fbr = f[:, -1:, :]
        fbrl = f[:, -2:-1, :]

        fblp_gas = [fbl + (i + 1) * (fbl - fblr) for i in range(self.padding_order)]
        fbrp_gas = [fbr + (i + 1) * (fbr - fbrl) for i in range(self.padding_order)]

        fbl_vapor=self.dis.maxwellian(rho_u_theta=self.LV)
        fbr_vapor=self.dis.maxwellian(rho_u_theta=self.RV)

        fblp=[g*(self.dis.v[...,0]<0)+fbl_vapor*(self.dis.v[...,0]>0) for g in fblp_gas]
        fbrp=[g*(self.dis.v[...,0]>0)+fbr_vapor*(self.dis.v[...,0]<0) for g in fbrp_gas]

        fbc = torch.cat(fblp + [f,] + fbrp, dim=1)
        return fbc

    def upwind(self, fl, fr, direction=0):
        return self._v_p[..., direction] * fl + self._v_m[..., direction] * fr

    def _grad_vf_constant(self, f, dt):
        self.padding_order = 1
        fbc = self.apply_BC(f)
        qP = fbc  # j+1/2
        qN = fbc  # j-1/2
        qP, qN = self.update(qP, qN)
        fluxL = self.upwind(qP[..., 0:-2, :], qN[..., 1:-1, :])
        fluxR = self.upwind(qP[..., 1:-1, :], qN[..., 2:, :])
        return 1 / self.dis.dx * (fluxR - fluxL)

    def _grad_vf_2(self, f, dt):
        self.padding_order = 2
        fbc = self.apply_BC(f)
        qN, qP = linear_reconstruction_VanLeer(
            fbc[..., 0:-2, :], fbc[..., 1:-1, :], fbc[..., 2:, :]
        )
        qP, qN = self.update(qP, qN)
        fluxL = self.upwind(qP[..., 0:-2, :], qN[..., 1:-1, :])
        fluxR = self.upwind(qP[..., 1:-1, :], qN[..., 2:, :])
        return 1 / self.dis.dx * (fluxR - fluxL)

    def _grad_vf_weno(self, f, dt):
        self.padding_order = 3
        fbc = self.apply_BC(f)
        qN, qP = weno5_LR(
            fbc[..., :-4, :],
            fbc[..., 1:-3, :],
            fbc[..., 2:-2, :],
            fbc[..., 3:-1, :],
            fbc[..., 4:, :],
        )
        qP, qN = self.update(qP, qN)
        fluxL = self.upwind(qP[..., 0:-2, :], qN[..., 1:-1, :])
        fluxR = self.upwind(qP[..., 1:-1, :], qN[..., 2:, :])
        return 1 / self.dis.dx * (fluxR - fluxL)

    def _grad_vf_plm(self, f, dt):
        self.padding_order = 2
        fbc = self.apply_BC(f)
        Fp, Fm = F_plm(fbc, self._v_p, self._v_m, self.dis.dx, dt)
        return self._v_p * Fp + self._v_m * Fm

    # def half_maxwellian(self, arg, sign):
    #     # 生成以arg为参数的一个maxwellian，并截断保留一半
    #     rho, u, T = arg
    #     maxwellian = (rho / torch.sqrt(2 * pi * T))[..., None] * torch.exp(
    #         -((u[..., None] - self.v) ** 2) / (2 * T)[..., None]
    #     )
    #     # B,W,V
    #     if sign > 0:
    #         maxwellian[..., self.dis.v[...,0] < 0] = 0
    #     elif sign < 0:
    #         maxwellian[..., self.dis.v[...,0] > 0] = 0
    #     else:
    #         raise ValueError("sign should be 1 or -1")
    #     return maxwellian

    # def cut_distribution(self, f, sign):
    #     f = f.clone()
    #     if sign == 1:
    #         f[..., self.v < 0] = 0
    #     elif sign == -1:
    #         f[..., self.v > 0] = 0
    #     else:
    #         raise ValueError("sign should be 1 or -1")
    #     return f

    # def reverse_distribution(self, f):
    #     return torch.flip(f, [-1,])


    def step_rk(self, dt):
        f = self.f
        Flux = partial(self._grad_vf, dt=dt)
        ff = lambda x: -Flux(x)
        getTau=lambda U: self.getTau(U)
        f = self.RK(ff, f, dt, getTau, self.dis.rho_u_theta, self.dis.maxwellian)
        self.f = f

    def step_imex(self, dt):
        f = self.f
        flux = self._grad_vf(f,dt)
        ftmp = f - dt*flux
        Unew=self.dis.rho_u_theta(ftmp)
        Tau=self.getTau(Unew)
        Mnew=self.dis.maxwellian(Unew)
        #print("Unew",Unew[0][0,:,0],Unew[1][0,:,0],Unew[2][0,:,0])
        expt=torch.exp(-dt/Tau)
        #print(expt.flatten()[0],dt,Tau.flatten()[0])
        Q=self.coller.do_collision(f,self.kn)
        QU=self.dis.rho_u_theta(Q)
        #print("QU",QU[0][0,:,0],QU[1][0,:,0],QU[2][0,:,0])
        fnew=(ftmp+dt*((1-expt)/Tau*Mnew+Q*expt))/(1+dt/Tau*(1-expt))
        self.f=fnew
        return fnew

    def solve_to(self, tmax, max_dt=None):
        t = 0
        if max_dt is None:
            max_dt = 0.45 *  self.dis.dx / self.dis.vmax[0]
        while t < tmax:
            dt = min(tmax - t, max_dt)
            self.step(dt)  # 使用算法迭代一步
            t += dt

    def step_euler(self, dt):
        f = self.f
        flux = self._grad_vf(f,dt)
        ftmp = f - dt*flux
        Unew = self.dis.rho_u_theta(ftmp)
        # if self.bgk_simple_kn:
        #     Tau = self.kn
        # else:
        #     Tau = self.kn / Unew[0] 
        Tau=self.getTau(Unew)
        Q = self.coller.do_collision(f, Tau)
        self.Q=Q
        fnew=ftmp+dt*Q
        self.f=fnew
        return fnew

    def get_gas_macro(self):
        """返回(rho, m, E)

        Returns:
            torch.array: (rho, m, E)
        """
        rho = self.density()
        u = self.velocity()
        T = self.temperature()
        m = rho * u
        E = 0.5 * (u ** 2 + T) * rho
        return torch.vstack((rho, m, E))

    def get_hermite_moments(self, M_order):
        u, sqr_T = self.velocity(), torch.sqrt(self.temperature())
        c_v = (self.v[None, None, :] - u[:, :, None]) / sqr_T[:, :, None]
        mom = torch.zeros(u.shape + (M_order + 2,))
        mom[..., 0] = u
        mom[..., 1] = sqr_T ** 2
        for i in range(M_order):
            mom[..., i + 2] = (
                self.sum(self.f * eval_hermitenorm(i, c_v)) / factorial(i) * sqr_T ** i
            )
        return mom

    def get_f(self):
        return self.f.clone()

    def get_entropy(self):
        return self.sum(self.f * torch.log(self.f))