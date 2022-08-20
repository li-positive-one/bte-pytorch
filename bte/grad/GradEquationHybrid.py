import torch
import torch.nn as nn

from scipy.special import roots_hermitenorm

from bte.utils.weno5 import *
from bte.utils.limiter import *
from bte.grad.distribution import DVDis, HermiteDis
from bte.dvm.solver import get_reflecting_bdv
Tensor = torch.Tensor


def MaxRootOfHermitePolynomial(N: int) -> float:
    return roots_hermitenorm(N)[0][-1]


def simpson(l, m, r):
    return (l + 4 * m + r) / 6

def upwind(fl:DVDis, fr:DVDis):
    _v_p=0.5 * (fl.v +fl.v.abs())
    _v_m=0.5 * (fl.v -fl.v.abs())
    return DVDis(fl.v_meta, _v_p[..., 0] * fl.f + _v_m[..., 0] * fr.f)

def linear_reconstruction(
    dis: HermiteDis, disL: HermiteDis, disR: HermiteDis, limiter=limiter_mc
):
    theta = torch.zeros_like(dis.coef)
    theta1 = dis.coef - disL.coef
    theta2 = disR.coef - dis.coef
    mask = torch.abs(theta1 * theta2) > 1e-12
    theta[mask] = (theta1 / (theta2 + 1e-12))[mask]

    theta = limiter(theta)
    RecL = dis.coef - theta / 2 * theta2
    RecR = dis.coef + theta / 2 * theta2

    return HermiteDis(dis.u, dis.theta, RecL), HermiteDis(dis.u, dis.theta, RecR)


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
        self.bgk_simple_kn = False
        self.max_dt = 1.0
        self.hme = False
        self.nrxx = False
        self.CFL = 0.45
        self.chi = 0
        self.padding_order = 2

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

    def set_initial(self, kn: torch.Tensor, f0: HermiteDis, dx: float):
        assert kn.shape[-1] == 1
        assert f0.coef.shape[-1] == self.M
        assert kn.shape == f0.u.shape
        self.kn = kn
        self.f0 = f0
        self.dx = dx

    def Grad_Vec(self, dis: HermiteDis, Kn=None):
        u0 = dis.u
        s0 = torch.sqrt(dis.theta)
        return u0 + self.MRoH * s0, u0 - self.MRoH * s0

    def Grad_Vec_LLF(self, dis: HermiteDis, Kn=None):
        u0 = dis.u
        s0 = torch.sqrt(dis.theta)
        return torch.maximum(
            torch.abs(u0 + self.MRoH * s0), torch.abs(u0 - self.MRoH * s0)
        )

    def Grad_Closure(self, dis: HermiteDis, Kn=None):
        return torch.zeros_like(dis.u)

    def NRxx_Closure(self, dis: HermiteDis, Kn: Tensor) -> Tensor:
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

    def collision_BGK(self, dis: HermiteDis, dt, Kn):
        # if torch.tensor(dt).numel() == 1:
        #     dt = torch.tensor(dt, device=dis.u.device).repeat(dis.u.shape[0])
        dt = dt.reshape((dis.u.shape[0], 1, 1))
        if self.bgk_simple_kn:
            ceo = Kn
        else:
            ceo = Kn / dis.coef[..., 0:1]
        tmp = dis.coef.clone()
        tmp[..., 1:] = dis.coef[..., 1:] * torch.exp(-dt / ceo)
        return HermiteDis(dis.u, dis.theta, tmp)

    def get_tstep_cell(self, dis: HermiteDis):
        """
        因为一般是在碰撞后求解下一步的时间步长，所以直接读取，
        若非如此，需要 _,c,theta=get_pcs(dis)
        """
        c, theta = dis.u[..., 0], dis.theta[..., 0]
        return torch.abs(c) + self.MRoH * torch.sqrt(theta)

    def get_tstep(self, dislist: HermiteDis, dx):
        z = self.get_tstep_cell(dislist)
        tstep = self.CFL * dx / z.max(dim=1)[0]
        if self.nrxx:
            tstep = self.CFL * dx * dx / z.max(dim=1)[0]
        return tstep

    def LLF_helper(
        self,
        disL: HermiteDis,
        disR: HermiteDis,
        fluxL: HermiteDis,
        fluxR: HermiteDis,
        lam,
    ):
        tmp = 0.5 * (fluxL.coef + fluxR.coef) + 0.5 * lam * (disL.coef - disR.coef)
        return HermiteDis(disL.u, disL.theta, tmp)

    def HLL_helper(
        self,
        disL: HermiteDis,
        disR: HermiteDis,
        fluxL: HermiteDis,
        fluxR: HermiteDis,
        lambda_L,
        lambda_R,
    ) -> HermiteDis:
        k0 = (torch.abs(lambda_L) - torch.abs(lambda_R)) / (lambda_L - lambda_R)
        k1 = (torch.abs(lambda_L) * lambda_R - torch.abs(lambda_R) * lambda_L) / (
            lambda_L - lambda_R
        )
        tmp = (
            0.5 * (fluxL.coef + fluxR.coef)
            + 0.5 * k0 * (fluxL.coef - fluxR.coef)
            - 0.5 * k1 * (disL.coef - disR.coef)
        )
        return HermiteDis(disL.u, disL.theta, tmp)

    def flux_helper(
        self,
        disL: HermiteDis,
        disR: HermiteDis,
        fluxL: HermiteDis,
        fluxR: HermiteDis,
        Kn,
        disLO: HermiteDis,
        disRO: HermiteDis,
        lam=0,
    ) -> HermiteDis:
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
            lambda_max = torch.max(vLmax, vRmax)
            R_Flux = self.LLF_helper(disL, disR, fluxL, fluxR, lambda_max)
        else:
            raise NotImplementedError

        return R_Flux

    def get_flux_hme_helper(self, disL: HermiteDis, disR: HermiteDis):
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

    def get_flux_hme(self, dis: HermiteDis, dx, Kn):
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

    def padding_periodic_dis(self, f: HermiteDis):
        disP = HermiteDis(
            self.padding_periodic_single(f.u),
            self.padding_periodic_single(f.theta),
            self.padding_periodic_single(f.coef),
        )
        return disP

    def padding_replicate_dis(self, f: HermiteDis):
        disP = HermiteDis(
            self.padding_replicate_single(f.u),
            self.padding_replicate_single(f.theta),
            self.padding_replicate_single(f.coef),
        )
        return disP

    def padding_constant_dis(self, f: HermiteDis):
        disP = HermiteDis(
            self.padding_constant_single(f.u),
            self.padding_constant_single(f.theta),
            self.padding_constant_single(f.coef),
        )
        return disP

    def padding_reflect_dis(self, f: HermiteDis):
        disP = HermiteDis(
            self.padding_reflect_single_u(f.u),
            self.padding_reflect_single(f.theta),
            self.padding_reflect_single(f.coef),
        )
        return disP

    def padding_linear_dis(self, f: HermiteDis):
        disP = HermiteDis(
            self.padding_linear_single_u(f.u),
            self.padding_linear_single(f.theta),
            self.padding_linear_single(f.coef),
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

    def padding_linear_single(self, f: torch.Tensor):
        # TODO
        pass

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
            return self.padding_replicate_dis(f)
        else:
            return

    def Reconstruction_constant(self, dis: HermiteDis):
        disL = HermiteDis(
            dis.u[..., :-2, :], dis.theta[..., :-2, :], dis.coef[..., :-2, :]
        )
        disM = HermiteDis(
            dis.u[..., 1:-1, :], dis.theta[..., 1:-1, :], dis.coef[..., 1:-1, :]
        )
        disR = HermiteDis(
            dis.u[..., 2:, :], dis.theta[..., 2:, :], dis.coef[..., 2:, :]
        )
        disLexP = disL.Project_2B(disM, STEP=self.PROJ_STEPS)
        disRexP = disR.Project_2B(disM, STEP=self.PROJ_STEPS)
        return disLexP, disM, disM, disRexP, disL, disR

    def Reconstruction_linear(self, dis: HermiteDis):
        disLL = HermiteDis(
            dis.u[..., :-4, :], dis.theta[..., :-4, :], dis.coef[..., :-4, :]
        ).clone()
        disL = HermiteDis(
            dis.u[..., 1:-3, :], dis.theta[..., 1:-3, :], dis.coef[..., 1:-3, :]
        ).clone()
        disM = HermiteDis(
            dis.u[..., 2:-2, :], dis.theta[..., 2:-2, :], dis.coef[..., 2:-2, :]
        ).clone()
        disR = HermiteDis(
            dis.u[..., 3:-1, :], dis.theta[..., 3:-1, :], dis.coef[..., 3:-1, :]
        ).clone()
        disRR = HermiteDis(
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

    def Reconstruction_WENO(self, disM: HermiteDis):
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

    def maxwellian_bdc_fix0(self, disLP_R, disMP_L, disMP_R, disRP_L, disLO, disRO):
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

            disLP_R = HermiteDis(disLP_R_u, disLP_R_theta, disLP_R_coef)
            disRP_L = HermiteDis(disRP_L_u, disRP_L_theta, disRP_L_coef)

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

            disLP_R = HermiteDis(disLP_R_u, disLP_R_theta, disLP_R_coef)
            disRP_L = HermiteDis(disRP_L_u, disRP_L_theta, disRP_L_coef)

            return disLP_R, disMP_L, disMP_R, disRP_L, disLO, disRO

    def maxwellian_bdc_fix1(self, disLP_R, disMP_L, disMP_R, disRP_L, disLO, disRO):
        # disLP_R,disMP_L,disMP_R,disRP_L,disLO,disRO
        # 对应pdf中的方法2
        # 修改 disLP_R 的最左边和 disRP_L的最右边

        HDIS_L = HermiteDis(disMP_L.u[...,0:1,:],disMP_L.theta[...,0:1,:],disMP_L.coef[...,0:1,:])
        VDIS_L = DVDis.empty(200,-10,10,device=HDIS_L.u.device).from_HermiteDis(HDIS_L)
        #然后计算左侧cell相应的右边界的分布函数，参照DVM中的算法       
        VDIS_LL = get_reflecting_bdv(VDIS_L, self.LV, self.xi, -1,no_linear=True)
        HDIS_LL =  HermiteDis.empty(disMP_L.ORDER,device=disMP_L.u.device).from_DVDis(VDIS_LL)
        disLP_R.u[...,0:1,:]=2*HDIS_LL.u[...,:,:]-HDIS_L.u
        disLP_R.theta[...,0:1,:]=2*HDIS_LL.theta[...,:,:]-HDIS_L.theta
        disLP_R.coef[...,0:1,:]=2*HDIS_LL.coef[...,:,:]-HDIS_L.coef

        HDIS_R = HermiteDis(disMP_R.u[...,-1:,:],disMP_R.theta[...,-1:,:],disMP_R.coef[...,-1:,:])
        VDIS_R = DVDis.empty(200,-10,10,device=HDIS_R.u.device).from_HermiteDis(HDIS_R)
        #然后计算左侧cell相应的右边界的分布函数，参照DVM中的算法       
        VDIS_RR = get_reflecting_bdv(VDIS_R, self.RV, self.xi, 1,no_linear=True)
        HDIS_RR =  HermiteDis.empty(disMP_R.ORDER,device=disMP_R.u.device).from_DVDis(VDIS_RR)
        disRP_L.u[...,-1:,:]=2*HDIS_RR.u[...,:,:]-HDIS_R.u
        disRP_L.theta[...,-1:,:]=2*HDIS_RR.theta[...,:,:]-HDIS_R.theta
        disRP_L.coef[...,-1:,:]=2*HDIS_RR.coef[...,:,:]-HDIS_R.coef

        return disLP_R, disMP_L, disMP_R, disRP_L, disLO, disRO


    def maxwellian_bdc_fix2(self, disLP_R, disMP_L, disMP_R, disRP_L, disLO, disRO):
        # disLP_R,disMP_L,disMP_R,disRP_L,disLO,disRO
        # 对应pdf中的方法2
        # 修改 disLP_R 的最左边和 disRP_L的最右边

        HDIS_L = HermiteDis(disMP_L.u[...,0:1,:],disMP_L.theta[...,0:1,:],disMP_L.coef[...,0:1,:])
        VDIS_L = DVDis.empty(200,-10,10,device=HDIS_L.u.device).from_HermiteDis(HDIS_L)
        #然后计算左侧cell相应的右边界的分布函数，参照DVM中的算法       
        VDIS_LL = get_reflecting_bdv(VDIS_L, self.LV, self.xi, -1)
        HDIS_LL =  HermiteDis.empty(disMP_L.ORDER,device=disMP_L.u.device).from_DVDis(VDIS_LL)
        disLP_R.u[...,0:1,:]=HDIS_LL.u[...,:,:]
        disLP_R.theta[...,0:1,:]=HDIS_LL.theta[...,:,:]
        disLP_R.coef[...,0:1,:]=HDIS_LL.coef[...,:,:]

        HDIS_R = HermiteDis(disMP_R.u[...,-1:,:],disMP_R.theta[...,-1:,:],disMP_R.coef[...,-1:,:])
        VDIS_R = DVDis.empty(200,-10,10,device=HDIS_R.u.device).from_HermiteDis(HDIS_R)
        #然后计算左侧cell相应的右边界的分布函数，参照DVM中的算法       
        VDIS_RR = get_reflecting_bdv(VDIS_R, self.RV, self.xi, 1)
        HDIS_RR =  HermiteDis.empty(disMP_R.ORDER,device=disMP_R.u.device).from_DVDis(VDIS_RR)
        disRP_L.u[...,-1:,:]=HDIS_RR.u[...,:,:]
        disRP_L.theta[...,-1:,:]=HDIS_RR.theta[...,:,:]
        disRP_L.coef[...,-1:,:]=HDIS_RR.coef[...,:,:]

        return disLP_R, disMP_L, disMP_R, disRP_L, disLO, disRO

    def get_convection_flux(self, dis: HermiteDis, dx, Kn):
        uM, thetaM, coefM = dis.as_tuple()
        coefMex = torch.cat((coefM, self.closure(dis, Kn)), -1)
        disMex = HermiteDis(uM, thetaM, coefMex)
        disMexPad = self.padding_dis(disMex)

        disLP_R, disMP_L, disMP_R, disRP_L, disLO, disRO = self.Reconstruction(
            disMexPad
        )

        # 在这里对maxwellian边界处的各阶矩进行修正。
        # if self.bdc == "maxwellian":
        #     disLP_R, disMP_L, disMP_R, disRP_L, disLO, disRO = self.maxwellian_bdc_fix2(
        #         disLP_R, disMP_L, disMP_R, disRP_L, disLO, disRO
        #     )

        Mulvec_LR = disLP_R.mulvec().down()
        Mulvec_ML = disMP_L.mulvec().down()
        Mulvec_MR = disMP_R.mulvec().down()
        Mulvec_RL = disRP_L.mulvec().down()

        fluxLC = self.flux_helper(
            disLP_R.down(),
            disMP_L.down(),
            Mulvec_LR,
            Mulvec_ML,
            Kn,
            disLO.down(),
            disMex.down(),
        )
        
        #===
        #在这里修正左边界的左侧flux，使用基于DVM的flux
        #===

        #先获取最左侧cell的分布函数(矩系数)
        HDIS_L = HermiteDis(disMP_L.u[...,0:1,:],disMP_L.theta[...,0:1,:],disMP_L.coef[...,0:1,:])
        #print("HDIS_L:",HDIS_L)
        VDIS_L = DVDis.empty(200,-10,10,device=HDIS_L.u.device).from_HermiteDis(HDIS_L)

        #然后计算左侧cell相应的右边界的分布函数，参照DVM中的算法       
        VDIS_LL = get_reflecting_bdv(VDIS_L, self.LV, self.xi, -1,no_linear=True)
        HDIS_LL =  HermiteDis.empty(disMP_L.ORDER,device=disMP_L.u.device).from_DVDis(VDIS_LL)
        #print("HDIS_LL:",HDIS_LL)

        #然后计算这个界面处的Flux(DVM意义下)
        Vflux_L =  upwind(VDIS_LL, VDIS_L) 
        #print(Vflux_L, Vflux_L.rho_u_theta())
        #然后将这个Flux投影到Grad空间中（以右侧cell的u/theta为准）
        Hflux_L = HermiteDis.empty(disMP_L.ORDER,device=disMP_L.u.device).from_DVDis(Vflux_L,u=HDIS_L.u,theta=HDIS_L.theta).down()

        #用这个flux替换fluxLC的最左侧的flux
        fluxLC.u[...,0:1,:]=Hflux_L.u[...,0:1,:]
        fluxLC.theta[...,0:1,:]=Hflux_L.theta[...,0:1,:]
        fluxLC.coef[...,0:1,:]=Hflux_L.coef[...,0:1,:]
        #print("L",Hflux_L.u,Hflux_L.theta,Hflux_L.coef)


        fluxRC = self.flux_helper(
            disMP_R.down(),
            disRP_L.down(),
            Mulvec_MR,
            Mulvec_RL,
            Kn,
            disMex.down(),
            disRO.down(),
        )

        # #===
        # #在这里修正右边界的右侧flux，使用基于DVM的flux
        # #===

        #先获取最左侧cell的分布函数(矩系数)
        HDIS_R = HermiteDis(disMP_R.u[...,-1:,:],disMP_R.theta[...,-1:,:],disMP_R.coef[...,-1:,:])
        VDIS_R = DVDis.empty(200,-10,10,device=disMP_R.u.device).from_HermiteDis(HDIS_R)
        #print("HDIS_R:",HDIS_R.coef.shape)
        #print("VDIS_R:",VDIS_R.f.shape)

        #然后计算左侧cell相应的右边界的分布函数，参照DVM中的算法       
        VDIS_RR = get_reflecting_bdv(VDIS_R, self.RV, self.xi, 1,no_linear=True)
        #print("VDIS_RR:",VDIS_RR.f.shape)

        #然后计算这个界面处的Flux(DVM意义下)
        Vflux_R =  upwind(VDIS_R, VDIS_RR) 
        #print("Vflux_R:",Vflux_R.f.shape)
        #然后将这个Flux投影到Grad空间中（以右侧cell的u/theta为准）
        Hflux_R = HermiteDis.empty(disMP_R.ORDER,device=disMP_R.u.device).from_DVDis(Vflux_R,u=HDIS_R.u,theta=HDIS_R.theta).down()
        #print("Hflux_R:",Hflux_R.coef.shape)
        #用这个flux替换fluxLC的最左侧的flux
        fluxRC.u[...,-1:,:]=Hflux_R.u[...,-1:,:]
        fluxRC.theta[...,-1:,:]=Hflux_R.theta[...,-1:,:]
        fluxRC.coef[...,-1:,:]=Hflux_R.coef[...,-1:,:]
        #print("R",Hflux_R.u,Hflux_R.theta,Hflux_R.coef)

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
            dis_zero = HermiteDis(u, theta, coef0)
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

    def Euler(self, dis: HermiteDis, dt, dx, Kn):
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
        dis0 = HermiteDis(dis.u, dis.theta, dis.coef.clone())
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

    def apply_flux(self, dis: HermiteDis, flux):
        tmp = dis.coef.clone()
        tmp = dis.coef + flux.coef
        return HermiteDis(dis.u, dis.theta, tmp)

    def forward(self, dis: HermiteDis, T, dx, Kn, verbose=False):
        if torch.tensor(T).numel() == 1:
            T = torch.tensor(T, device=dis.u.device).repeat(dis.u.shape[0])

        trec = torch.zeros((dis.u.shape[0],), device=dis.u.device)
        while (-trec + T).gt(0).any():
            dt = torch.minimum(self.get_tstep(dis, dx), -trec + T).detach()
            dt = torch.minimum(dt, self.max_dt * torch.ones_like(dt))
            if verbose:
                print(trec.min(), "/", T)
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
