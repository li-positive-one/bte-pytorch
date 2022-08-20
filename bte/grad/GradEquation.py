import torch
import torch.nn as nn

from scipy.special import roots_hermitenorm
from bte.grad.distribution import HermiteDisND

from bte.utils.weno5 import *
from bte.utils.limiter import *
from bte.grad.distribution import HermiteDis
from bte.grad.distribution import HermitedistributionBase as HDISBASE
from bte.utils.indexs import *
from bte.grad.collision_grad import collision_f, load_collision_kernel
Tensor = torch.Tensor


def MaxRootOfHermitePolynomial(N: int) -> float:
    return roots_hermitenorm(N)[0][-1]


def simpson(l, m, r):
    return (l + 4 * m + r) / 6


def linear_reconstruction(
    dis: HDISBASE, disL: HDISBASE, disR: HDISBASE, limiter=limiter_mc
):
    theta = torch.zeros_like(dis.coef)
    theta1 = dis.coef - disL.coef
    theta2 = disR.coef - dis.coef
    mask = torch.abs(theta1 * theta2) > 1e-12
    theta[mask] = (theta1 / (theta2 + 1e-12))[mask]

    theta = limiter(theta)
    RecL = dis.coef - theta / 2 * theta2
    RecR = dis.coef + theta / 2 * theta2

    return dis.new(dis.u, dis.theta, RecL), dis.new(dis.u, dis.theta, RecR)


class Equation_HermiteBased(nn.Module):
    def __init__(self, M=5, bdc="circular"):
        super().__init__()
        self.M = M
        self.MRoH = MaxRootOfHermitePolynomial(self.M)
        self.veloc = self.Grad_Vec_LLF
        self.flux_solver = "LLF"
        self.closure = self.Grad_Closure
        self.Reconstruction = self.Reconstruction_linear
        self.ode_step = self.Heun
        self.PROJ_STEPS = 1
        self.bdc = bdc
        self.split_order = 1
        self.collision = self.collision_BGK
        self.bgk_simple_kn ="simple2"
        self.max_dt = 1.0
        self.hme = False
        self.nrxx = False
        self.CFL = 0.45
        self.chi = 0
        self.padding_order = 2
        self.no_closure=False
        
    def set_collision(self, colltype:str):
        assert colltype=="BGK" or colltype=="Maxwellian"
        if colltype=="BGK":
            self.collision=self.collision_BGK
        elif colltype=="Maxwellian":
            self.set_QKernel()
            self.collision=self.collision_Kernel
        else:
            pass
        return 

    def set_QKernel(self):
        self.register_buffer("QKernel",load_collision_kernel(order=self.M))

    def __str__(self) -> str:
        base_str = super().__str__()
        infomation = f"""
====
Equation_HermiteBased
----
flux_solver:{self.flux_solver}
split_order:{self.split_order}
PROJ_STEPS:{self.PROJ_STEPS}
bgk_simple_kn:{self.bgk_simple_kn}
===="""
        return base_str + "\n" + infomation

    def set_order(self, space_order, time_order):
        self.set_space_order(space_order)
        self.set_time_order(time_order)

    def set_space_order(self, order):
        if order == 1:
            self.Reconstruction = self.Reconstruction_constant
            self.padding_order=1
        elif order == 2:
            self.Reconstruction = self.Reconstruction_linear
            self.padding_order=2
        else:
            raise ValueError

    def set_time_order(self, order):
        if order == 1:
            self.ode_step = self.Euler
        elif order == 2:
            self.ode_step = self.Heun
        else:
            raise ValueError

    def set_initial(self, kn: torch.Tensor, f0: HDISBASE, dx: float):
        # assert kn.shape[-1] == 1
        # assert f0.coef.shape[-1] == self.M
        # assert kn.shape == f0.u.shape
        self.kn = kn
        self.f0 = f0
        self.dx = dx
        if self.bdc=="constant":
            self.left_BC=f0.new(f0.u[...,0:1,:],f0.theta[...,0:1,:],f0.coef[...,0:1,:])
            self.right_BC=f0.new(f0.u[...,-1:,:],f0.theta[...,-1:,:],f0.coef[...,-1:,:])

    def Grad_Vec(self, dis: HDISBASE, Kn=None):
        u0 = dis.u
        s0 = torch.sqrt(dis.theta)
        return u0 + self.MRoH * s0, u0 - self.MRoH * s0

    def Grad_Vec_LLF(self, dis: HDISBASE, Kn=None):
        u0 = dis.u
        s0 = torch.sqrt(dis.theta)
        return torch.maximum(
            torch.abs(u0 + self.MRoH * s0), torch.abs(u0 - self.MRoH * s0)
        )

    def Grad_Closure(self, dis: HDISBASE, Kn=None):
        if isinstance(dis,HermiteDisND):
            dl=index_tables.get(dis.indt.ORDER+1,dis.indt.DIM).len-dis.indt.len
            ex = torch.zeros(dis.u.shape[:-1]+(dl,), device=dis.coef.device)
        else:
            ex = torch.zeros_like(dis.u)

        return ex

    def NRxx_Closure(self, dis: HDISBASE, Kn: Tensor) -> Tensor:
        disM = dis
        disL = disM.roll(-1)
        disR = disM.roll(1)
        disLP = disL.Project_2B(disM, STEP=self.PROJ_STEPS)
        disRP = disR.Project_2B(disM, STEP=self.PROJ_STEPS)
        theta = disM.theta
        closure = (
            -theta * Kn * (disRP.coef[..., -1:] - disLP.coef[..., -1:]) / (2 * self.dx)
        )
        return closure

    def getTau(self, dis, Kn):
        if self.bgk_simple_kn=="simple":
            tau = self.kn 
        elif self.bgk_simple_kn=="simple2":
            tau = Kn / dis.density()
        elif self.bgk_simple_kn=="mu_ref":
            tau =  self.mu_ref*2/(dis.theta*2)**(1-self.omega)/dis.density()
        else:
            raise ValueError
        return tau

    def collision_BGK(self, dis: HDISBASE, dt, Kn):
        # if torch.tensor(dt).numel() == 1:
        #     dt = torch.tensor(dt, device=dis.u.device).repeat(dis.u.shape[0])
        dt = dt.reshape((dis.u.shape[0], 1, 1))
        # if self.bgk_simple_kn:
        #     ceo = Kn
        # else:
        #     #ceo = Kn / dis.coef[..., 0:1]
        #     ceo = self.mu_ref*2/(dis.theta*2)**(1-self.omega)/dis.density()
        ceo=self.getTau(dis,Kn)
        tmp = dis.coef.clone()
        tmp[..., 1:] = dis.coef[..., 1:] * torch.exp(-dt / ceo)
        return dis.new(dis.u, dis.theta, tmp)

    def collision_Kernel(self, dis:HDISBASE, dt, Kn):
        dt = dt.reshape((dis.u.shape[0], 1, 1))
        if self.bgk_simple_kn:
            ceo = Kn
        else:
            ceo = Kn / dis.coef[..., 0:1]
        tmp = dis.coef.clone()
        Q = collision_f(self.QKernel, tmp)
        tmp = tmp + dt / ceo * Q
        return dis.new(dis.u, dis.theta, tmp)

    def get_tstep_cell(self, dis: HDISBASE):
        """
        因为一般是在碰撞后求解下一步的时间步长，所以直接读取，
        若非如此，需要 _,c,theta=get_pcs(dis)
        """
        c, theta = dis.u[..., 0], dis.theta[..., 0]
        return torch.abs(c) + self.MRoH * torch.sqrt(theta)

    def get_tstep(self, dislist: HDISBASE, dx):
        z = self.get_tstep_cell(dislist)
        tstep = self.CFL * dx / z.max(dim=1)[0]
        if self.nrxx:
            tstep = self.CFL * dx * dx / z.max(dim=1)[0]
        return tstep

    def LLF_helper(
        self,
        disL: HDISBASE,
        disR: HDISBASE,
        fluxL: HDISBASE,
        fluxR: HDISBASE,
        lam,
    ):
        #print(fluxL.coef.shape,lam.shape)
        tmp = 0.5 * (fluxL.coef + fluxR.coef) + 0.5 * lam * (disL.coef - disR.coef)
        return disL.new(disL.u, disL.theta, tmp)

    def HLL_helper(
        self,
        disL: HDISBASE,
        disR: HDISBASE,
        fluxL: HDISBASE,
        fluxR: HDISBASE,
        lambda_L,
        lambda_R,
    ) -> HDISBASE:
        k0 = (torch.abs(lambda_L) - torch.abs(lambda_R)) / (lambda_L - lambda_R)
        k1 = (torch.abs(lambda_L) * lambda_R - torch.abs(lambda_R) * lambda_L) / (
            lambda_L - lambda_R
        )
        tmp = (
            0.5 * (fluxL.coef + fluxR.coef)
            + 0.5 * k0 * (fluxL.coef - fluxR.coef)
            - 0.5 * k1 * (disL.coef - disR.coef)
        )
        return disL.new(disL.u, disL.theta, tmp)

    def flux_helper(
        self,
        disL: HDISBASE,
        disR: HDISBASE,
        fluxL: HDISBASE,
        fluxR: HDISBASE,
        Kn,
        disLO: HDISBASE,
        disRO: HDISBASE,
        lam=0,
    ) -> HDISBASE:
        if self.flux_solver == "HLL":
            vLmax, vLmin = self.veloc(disLO, Kn)
            vRmax, vRmin = self.veloc(disRO, Kn)

            lambda_min = torch.minimum(vLmin, vRmin)
            lambda_max = torch.maximum(vLmax, vRmax)

            R_Flux = self.HLL_helper(disL, disR, fluxL, fluxR, lambda_min, lambda_max)
        elif self.flux_solver == "LF":
            lam = lam.reshape((disL.u.shape[0], 1, 1))
            R_Flux = self.LLF_helper(disL, disR, fluxL, fluxR, lam)
        elif self.flux_solver == "LLF":
            vLmax = self.veloc(disLO, Kn)
            vRmax = self.veloc(disRO, Kn)
            lambda_max = torch.max(vLmax, vRmax)[...,0:1]
            R_Flux = self.LLF_helper(disL, disR, fluxL, fluxR, lambda_max)
        else:
            raise NotImplementedError

        return R_Flux

    def get_flux_hme_helper(self, disL: HDISBASE, disR: HDISBASE):
        f_l = disL.coef[..., -1]
        f_r = disR.coef[..., -1]
        f_l1 = disL.coef[..., -2]
        f_r1 = disR.coef[..., -2]
        rho_l = disL.coef[..., 0]
        rho_r = disR.coef[..., 0]
        diff_u = disR.u[..., 0] - disL.u[..., 0]
        diff_theta = disR.theta[..., 0] - disL.theta[..., 0]
        tmp = -(
            simpson(f_l, (f_l / rho_l + f_r / rho_r) * (rho_l + rho_r) / 4, f_r)
            * diff_u
            + 0.5
            * simpson(f_l1, (f_l1 / rho_l + f_r1 / rho_r) * (rho_l + rho_r) / 4, f_r1)
            * diff_theta
        )
        return tmp * (self.M)

    def get_flux_hme(self, dis: HDISBASE, dx, Kn):
        disM = dis
        u, theta, coefM = dis.as_tuple()
        disL = disM.roll(-1)
        disR = disM.roll(1)
        tmp_L = torch.zeros_like(disM.coef)
        tmp_R = torch.zeros_like(disM.coef)

        coef0 = torch.zeros_like(disM.coef)

        tmp_L[..., -1] = self.get_flux_hme_helper(disL, disM)
        tmp_R[..., -1] = self.get_flux_hme_helper(disM, disR)

        dis_zero = (u, theta, coef0)
        fluxLC = self.flux_helper(
            dis_zero, dis_zero, (u, theta, -tmp_L), dis_zero, Kn, disL, disM
        )
        fluxRC = self.flux_helper(
            dis_zero, dis_zero, dis_zero, (u, theta, tmp_R), Kn, disM, disR
        )
        sum_flux = fluxRC - fluxLC

        return sum_flux

    def padding_periodic_dis(self, f: HDISBASE):
        disP = f.new(
            self.padding_periodic_single(f.u),
            self.padding_periodic_single(f.theta),
            self.padding_periodic_single(f.coef),
        )
        return disP

    def padding_replicate_dis(self, f: HDISBASE):
        disP = f.new(
            self.padding_replicate_single(f.u),
            self.padding_replicate_single(f.theta),
            self.padding_replicate_single(f.coef),
        )
        return disP

    def padding_constant_dis(self, f: HDISBASE):
        u = f.u.transpose(-1, -2)
        ubc = F.pad(
            u, pad=(self.padding_order, self.padding_order), mode="constant"
        ).transpose(-1, -2)
        ubc[..., 0 : self.padding_order, :] = self.left_BC.u
        ubc[..., -self.padding_order :, :] = self.right_BC.u

        T = f.theta.transpose(-1, -2)
        Tbc = F.pad(
            T, pad=(self.padding_order, self.padding_order), mode="constant"
        ).transpose(-1, -2)
        Tbc[..., 0 : self.padding_order, :] = self.left_BC.theta
        Tbc[..., -self.padding_order :, :] = self.right_BC.theta

        coef = f.coef.transpose(-1, -2)
        coefbc = F.pad(
            coef, pad=(self.padding_order, self.padding_order), mode="constant"
        ).transpose(-1, -2)
        coefbc[..., 0 : self.padding_order, :] = self.left_BC.coef
        coefbc[..., -self.padding_order :, :] = self.right_BC.coef

        disP = f.new(
            ubc,
            Tbc,
            coefbc
        )
        
        return disP

    def padding_reflect_dis(self, f: HDISBASE):
        disP = f.new(
            self.padding_reflect_single_u(f.u),
            self.padding_reflect_single(f.theta),
            self.padding_reflect_single(f.coef),
        )
        return disP

    def padding_periodic_single(self, f: torch.Tensor):
        f = f.transpose(-1, -2)
        return F.pad(
            f, pad=(self.padding_order, self.padding_order), mode="circular"
        ).transpose(-1, -2)

    def padding_replicate_single(self, f: torch.Tensor):
        f = f.transpose(-1, -2)
        return F.pad(
            f, pad=(self.padding_order, self.padding_order), mode="replicate"
        ).transpose(-1, -2)

    def padding_reflect_single(self, f: torch.Tensor):
        fp = torch.zeros(
            *f.shape[0:-2],
            f.shape[-2] + 2 * self.padding_order,
            f.shape[-1],
            device=f.device,
        )
        fp[..., self.padding_order : -self.padding_order, :] = f
        for i in range(self.padding_order):
            fp[..., self.padding_order - i - 1, 0::2] = fp[
                ..., self.padding_order + i, 0::2
            ]
            fp[..., -self.padding_order + i, 0::2] = fp[
                ..., -self.padding_order - i - 1, 0::2
            ]
            fp[..., self.padding_order - i - 1, 1::2] = -fp[
                ..., self.padding_order + i, 1::2
            ]
            fp[..., -self.padding_order + i, 1::2] = -fp[
                ..., -self.padding_order - i - 1, 1::2
            ]
        return fp

    def padding_reflect_single_u(self, f: torch.Tensor):
        fp = torch.zeros(
            *f.shape[0:-2],
            f.shape[-2] + 2 * self.padding_order,
            f.shape[-1],
            device=f.device,
        )
        fp[..., self.padding_order : -self.padding_order, :] = f
        for i in range(self.padding_order):
            fp[..., self.padding_order - i - 1, 0::2] = -fp[
                ..., self.padding_order + i, 0::2
            ]
            fp[..., -self.padding_order + i, 0::2] = -fp[
                ..., -self.padding_order - i - 1, 0::2
            ]
        return fp

    def padding_constant_single(self, f: torch.Tensor):
        f = f.transpose(-1, -2)
        fbc = F.pad(
            f, pad=(self.padding_order, self.padding_order), mode="constant"
        ).transpose(-1, -2)
        fbc[..., 0 : self.padding_order, :] = self.left_BC
        fbc[..., -self.padding_order :, :] = self.right_BC
        return fbc

    def padding_dis(self, f: torch.Tensor):
        if self.bdc == "circular":
            return self.padding_periodic_dis(f)
        elif self.bdc == "constant":
            return self.padding_constant_dis(f)
        elif self.bdc == "cauchy":
            return self.padding_replicate_dis(f)
        elif self.bdc == "maxwellian":
            return self.padding_reflect_dis(f)
        else:
            return

    def Reconstruction_constant(self, dis: HDISBASE):
        #print("dis.u.shape:", dis.u.shape)
        disL = dis.new(
            dis.u[..., :-2, :], dis.theta[..., :-2, :], dis.coef[..., :-2, :]
        )
        disM = dis.new(
            dis.u[..., 1:-1, :], dis.theta[..., 1:-1, :], dis.coef[..., 1:-1, :]
        )
        disR = dis.new(
            dis.u[..., 2:, :], dis.theta[..., 2:, :], dis.coef[..., 2:, :]
        )
        disLexP = disL.Project_2B(disM, STEP=self.PROJ_STEPS)
        disRexP = disR.Project_2B(disM, STEP=self.PROJ_STEPS)
        return disLexP, disM, disM, disRexP, disL, disR

    def Reconstruction_linear(self, dis: HDISBASE):
        disLL = dis.new(
            dis.u[..., :-4, :], dis.theta[..., :-4, :], dis.coef[..., :-4, :]
        ).clone()
        disL = dis.new(
            dis.u[..., 1:-3, :], dis.theta[..., 1:-3, :], dis.coef[..., 1:-3, :]
        ).clone()
        disM = dis.new(
            dis.u[..., 2:-2, :], dis.theta[..., 2:-2, :], dis.coef[..., 2:-2, :]
        ).clone()
        disR = dis.new(
            dis.u[..., 3:-1, :], dis.theta[..., 3:-1, :], dis.coef[..., 3:-1, :]
        ).clone()
        disRR = dis.new(
            dis.u[..., 4:, :], dis.theta[..., 4:, :], dis.coef[..., 4:, :]
        ).clone()

        disLP = disL.Project_2B(disM, STEP=self.PROJ_STEPS)
        disRP = disR.Project_2B(disM, STEP=self.PROJ_STEPS)
        disLLP = disLL.Project_2B(disM, STEP=self.PROJ_STEPS)
        disRRP = disRR.Project_2B(disM, STEP=self.PROJ_STEPS)

        _, disLP_R = linear_reconstruction(disLP, disLLP, disM)
        disRP_L, _ = linear_reconstruction(disRP, disM, disRRP)
        disMP_L, disMP_R = linear_reconstruction(disM, disLP, disRP)

        return disLP_R, disMP_L, disMP_R, disRP_L, disL, disR

    def Reconstruction_WENO(self, disM: HDISBASE):
        raise NotImplementedError
        # disLL = disM.roll(-2)
        # disL = disM.roll(-1)
        # disR = disM.roll(1)
        # disRR = disM.roll(2)

        # disLP = disL.Project_AB(disM, STEP=self.PROJ_STEPS)
        # disRP = disR.Project_AB(disM, STEP=self.PROJ_STEPS)
        # disLLP = disLL.Project_AB(disM, STEP=self.PROJ_STEPS)
        # disRRP = disRR.Project_AB(disM, STEP=self.PROJ_STEPS)

        # disMP_Lm, disMP_Rm = weno5_LR(
        #     disLLP.coef, disLP.coef, disM.coef, disRP.coef, disRRP.coef
        # )
        # disMP_L = HermiteDis(disM.u, disM.theta, disMP_Lm)
        # disMP_R = HermiteDis(disM.u, disM.theta, disMP_Rm)
        # Recon_disLR = disMP_R.roll(-1)
        # Recon_disLR = Recon_disLR.Project_2B(disM, STEP=self.PROJ_STEPS)

        # Recon_disRL = disMP_L.roll(1)
        # Recon_disRL = Recon_disRL.Project_2B(disM, STEP=self.PROJ_STEPS)

        # return Recon_disLR, disMP_L, disMP_R, Recon_disRL, disL, disR

    def maxwellian_bdc_fix(self, disLP_R, disMP_L, disMP_R, disRP_L, disLO, disRO):
        # disLP_R,disMP_L,disMP_R,disRP_L,disLO,disRO
        # 左cell的右边界，中间cell的左边界，中间cell的右边界，右边cell的左边界。左边cell的原始值，右边cell的原始值。
        # 要改的是左cell的右边界、右cell的左边界。要改idx=0和idx=-1的cell
        if self.chi == 0:
            disLP_R_u = disLP_R.u.clone()
            disMP_L_u = disMP_L.u.clone()
            disMP_R_u = disMP_R.u.clone()
            disRP_L_u = disRP_L.u.clone()

            disLP_R_theta = disLP_R.theta.clone()
            disMP_L_theta = disMP_L.theta.clone()
            disMP_R_theta = disMP_R.theta.clone()
            disRP_L_theta = disRP_L.theta.clone()

            disLP_R_coef = disLP_R.coef.clone()
            disMP_L_coef = disMP_L.coef.clone()
            disMP_R_coef = disMP_R.coef.clone()
            disRP_L_coef = disRP_L.coef.clone()

            # 修正disLP_R
            disLP_R_3_theta = disMP_L_theta[:, 0, :]
            disLP_R_3_coef = disMP_L_coef[:, 0, :]
            disLP_R_3_coef[:, 1::2] = -disMP_L_coef[:, 0, 1::2]
            disLP_R_u[:, 0, :] = -disMP_L_u[:, 0, :]
            disLP_R_theta[:, 0, :] = disLP_R_3_theta
            disLP_R_coef[:, 0, :] = disLP_R_3_coef

            # 修正disRP_L
            disRP_L_3_theta = disMP_R_theta[:, -1, :]
            disRP_L_3_coef = disMP_R_coef[:, -1, :]
            disRP_L_3_coef[:, 1::2] = -disMP_R_coef[:, -1, 1::2]
            disRP_L_u[:, -1, :] = -disMP_R_u[:, -1, :]
            disRP_L_theta[:, -1, :] = disRP_L_3_theta
            disRP_L_coef[:, -1, :] = disRP_L_3_coef

            disLP_R = disLO.new(disLP_R_u, disLP_R_theta, disLP_R_coef)
            disRP_L = disLO.new(disRP_L_u, disRP_L_theta, disRP_L_coef)

            return disLP_R, disMP_L, disMP_R, disRP_L, disLO, disRO
        else:
            disLP_R_u = disLP_R.u.clone()
            disMP_L_u = disMP_L.u.clone()
            disMP_R_u = disMP_R.u.clone()
            disRP_L_u = disRP_L.u.clone()

            disLP_R_theta = disLP_R.theta.clone()
            disMP_L_theta = disMP_L.theta.clone()
            disMP_R_theta = disMP_R.theta.clone()
            disRP_L_theta = disRP_L.theta.clone()

            disLP_R_coef = disLP_R.coef.clone()
            disMP_L_coef = disMP_L.coef.clone()
            disMP_R_coef = disMP_R.coef.clone()
            disRP_L_coef = disRP_L.coef.clone()

            # 修正disLP_R
            disLP_R_3_theta = disMP_L_theta[:, 0, :]
            disLP_R_3_coef = disMP_L_coef[:, 0, :]
            disLP_R_3_coef[:, 1::2] = -disMP_L_coef[:, 0, 1::2]
            disLP_R_u[:, 0, :] = -disMP_L_u[:, 0, :]
            disLP_R_theta[:, 0, :] = disLP_R_3_theta
            disLP_R_coef[:, 0, :] = disLP_R_3_coef

            # 修正disRP_L
            disRP_L_3_theta = disMP_R_theta[:, -1, :]
            disRP_L_3_coef = disMP_R_coef[:, -1, :]
            disRP_L_3_coef[:, 1::2] = -disMP_R_coef[:, -1, 1::2]
            disRP_L_u[:, -1, :] = -disMP_R_u[:, -1, :]
            disRP_L_theta[:, -1, :] = disRP_L_3_theta
            disRP_L_coef[:, -1, :] = disRP_L_3_coef

            disLP_R = disLO.new(disLP_R_u, disLP_R_theta, disLP_R_coef)
            disRP_L = disLO.new(disRP_L_u, disRP_L_theta, disRP_L_coef)

            return disLP_R, disMP_L, disMP_R, disRP_L, disLO, disRO

    def get_convection_flux(self, dis: HDISBASE, dx, Kn):
        # uM, thetaM, coefM = dis.as_tuple()
        # coefMex = torch.cat((coefM, self.closure(dis, Kn)), -1)
        # if isinstance(dis,HermiteDisND):
        #     disMex = dis.new(uM, thetaM, coefMex, 
        #             index_tables.get(dis.indt.ORDER+1,dis.indt.DIM))
        # else:
        #     disMex = dis.new(uM, thetaM, coefMex)
        # disMexPad = self.padding_dis(disMex)

        disP = self.padding_dis(dis)
        uP, thetaP, coefP = disP.as_tuple()
        coefPex = torch.cat((coefP, self.closure(disP, Kn)), -1)
        if isinstance(dis,HermiteDisND):
            disMexPad = dis.new(uP, thetaP, coefPex, 
                    index_tables.get(dis.indt.ORDER+1,dis.indt.DIM))
        else:
            disMexPad = dis.new(uP, thetaP, coefPex)
        

        disLP_R, disMP_L, disMP_R, disRP_L, disLO, disRO = self.Reconstruction(
            disMexPad
        )

        # 在这里对maxwellian边界处的各阶矩进行修正。
        if self.bdc == "maxwellian":
            disLP_R, disMP_L, disMP_R, disRP_L, disLO, disRO = self.maxwellian_bdc_fix(
                disLP_R, disMP_L, disMP_R, disRP_L, disLO, disRO
            )
        # print("LP_R",disLP_R.u[0,0,:], disLP_R.theta[0,0,:], disLP_R.coef[0,0,:])
        # print("MP_L",disMP_L.u[0,0,:], disMP_L.theta[0,0,:], disMP_L.coef[0,0,:])

        #         print("RP_L",disRP_L.u[0,-1,:], disRP_L.theta[0,-1,:], disRP_L.coef[0,-1,:])
        #         print("MP_R",disMP_R.u[0,-1,:], disMP_R.theta[0,-1,:], disMP_R.coef[0,-1,:])

        Mulvec_LR = disLP_R.mulvec().down()
        Mulvec_ML = disMP_L.mulvec().down()
        Mulvec_MR = disMP_R.mulvec().down()
        Mulvec_RL = disRP_L.mulvec().down()

        #         print("Mulvec",1e2*Mulvec_RL.rho[0,-1,0],1e2*Mulvec_MR.rho[0,-1,0])
        #         print("dis",1e5*disRP_L.rho[0,-1,0],1e5*disMP_R.rho[0,-1,0])
        #         print(1e5*disRO.u[0,-1,0],1e5*disMex.u[0,-1,0])
        #         print(1e5*disRO.theta[0,-1,0],1e5*disMex.theta[0,-1,0])
        #         print("===")
        fluxLC = self.flux_helper(
            disLP_R.down(),
            disMP_L.down(),
            Mulvec_LR,
            Mulvec_ML,
            Kn,
            disLO.down(),
            dis,
        )
        # print(fluxLC.rho[0,0,0])
        fluxRC = self.flux_helper(
            disMP_R.down(),
            disRP_L.down(),
            Mulvec_MR,
            Mulvec_RL,
            Kn,
            dis,
            disRO.down(),
        )
        # print(100000*fluxLC.rho[0,0,0])
        # print(100000*fluxRC.rho[0,-1,0])

        sum_flux = fluxRC - fluxLC

        if self.hme:
            disL = disLO.down()
            disR = disRO.down()
            tmp_L = torch.zeros_like(dis.coef)
            tmp_R = torch.zeros_like(dis.coef)
            coef0 = torch.zeros_like(dis.coef)

            tmp_L[..., -1] = self.get_flux_hme_helper(disL, dis)
            tmp_R[..., -1] = self.get_flux_hme_helper(dis, disR)

            u, theta, coefM = dis.as_tuple()
            dis_zero = dis.new(u, theta, coef0)
            fluxLC = self.flux_helper(
                dis_zero, dis_zero, (u, theta, -tmp_L), dis_zero, Kn, disL, dis
            )  # Rn+
            fluxRC = self.flux_helper(
                dis_zero, dis_zero, dis_zero, (u, theta, tmp_R), Kn, dis, disR
            )  # Rn-
            hme_flux = fluxRC - fluxLC
            sum_flux = sum_flux + hme_flux
        return sum_flux

    def get_flux(self, dis, dx, Kn):
        sum_flux = self.get_convection_flux(dis, dx, Kn) * (1 / dx)
        return sum_flux

    def Euler(self, dis: HDISBASE, dt, dx, Kn):
        if self.split_order == 2:
            dis = self.collision(dis, dt / 2, Kn)
        flux = -dt.reshape((dis.u.shape[0], 1, 1)) * self.get_flux(dis, dx, Kn)
        dis = self.apply_flux(dis, flux)
        dis = dis.Project_to_STD(STEP=self.PROJ_STEPS)
        if self.split_order == 1:
            dis = self.collision(dis, dt, Kn)
        elif self.split_order == 2:
            dis = self.collision(dis, dt / 2, Kn)
        else:
            raise ValueError("split_order must be 1 or 2.")
        return dis

    def Heun(self, dis, dt, dx, Kn):
        if self.split_order == 2:
            dis = self.collision(dis, dt / 2, Kn)

        flux = -dt.reshape((dis.u.shape[0], 1, 1)) * self.get_flux(dis, dx, Kn)
        dis0 = dis.new(dis.u, dis.theta, dis.coef.clone())
        dis1 = self.apply_flux(dis, flux)
        dis1 = dis1.Project_to_STD(STEP=self.PROJ_STEPS)

        flux2 = -dt.reshape((dis.u.shape[0], 1, 1)) * self.get_flux(dis1, dx, Kn)
        flux2 = flux2.Project_2B(dis0, STEP=self.PROJ_STEPS)

        sumflux = 0.5 * flux + 0.5 * flux2
        dis = self.apply_flux(dis0, sumflux)
        dis = dis.Project_to_STD(STEP=self.PROJ_STEPS)

        if self.split_order == 1:
            dis = self.collision(dis, dt, Kn)
        elif self.split_order == 2:
            dis = self.collision(dis, dt / 2, Kn)
        else:
            raise ValueError("split_order must be 1 or 2.")
        return dis

    def apply_flux(self, dis: HDISBASE, flux):
        tmp = dis.coef.clone()
        tmp = dis.coef + flux.coef
        return dis.new(dis.u, dis.theta, tmp)

    def forward(self, dis: HDISBASE, T, dx, Kn, verbose=False):
        if torch.tensor(T).numel() == 1:
            T = torch.tensor(T, device=dis.u.device).repeat(dis.u.shape[0])

        trec = torch.zeros((dis.u.shape[0],), device=dis.u.device)
        while (-trec + T).gt(0).any():
            dt = torch.minimum(self.get_tstep(dis, dx), -trec + T).detach()
            dt = torch.minimum(dt, self.max_dt * torch.ones_like(dt))
            if verbose:
                print(dt, "/", T)
            dis = self.ode_step(dis, dt, dx, Kn)
            trec += dt
        return dis

    def solve_to(self, T, verbose=False):
        dis = self.f0
        dx = self.dx
        if (not torch.is_tensor(T)) and torch.tensor(T).numel() == 1:
            T = torch.tensor(T, device=dis.u.device).repeat(dis.u.shape[0])

        trec = torch.zeros((dis.u.shape[0],), device=dis.u.device)
        while (-trec + T).gt(0).any():
            dt = torch.minimum(self.get_tstep(dis, dx), -trec + T).detach()
            dt = torch.minimum(dt, self.max_dt * torch.ones_like(dt))
            if verbose:
                print(dt, "/", T)
            self.f0 = self.ode_step(dt)
            trec += dt
        return self.f0

    def run(self, dis, T, dx, Kn=None, verbose=False):
        print("running...")
        return self.forward(dis, T, dx, Kn=Kn, verbose=verbose)

    def set_BC_Value(self, LV, RV):
        self.LV = LV  # LV=[rho,u,theta] #shape [[B,1],[B,1],[B,1]]
        self.RV = RV  # RV=[rho,u,theta] #shape [[B,1],[B,1],[B,1]]
