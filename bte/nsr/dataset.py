import numpy as np
import torch
from bte.nsr.model import maxwellian
from bte.nsr.mesh import get_vmsh
from bte.nsr.utils import fvmlinspace
import torch.nn.functional as F
from scipy import signal
import matplotlib.pyplot as plt
from bte.nsr.model_LR import maxwellian_LR


class Datagenerator():
    def __init__(self):
        pass

    def get_prim(self, x):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    def init_value(self, device='cuda', dtype=torch.float32):
        raise NotImplementedError

    def get(self, device='cuda', dtype=torch.float32):
        raise NotImplementedError

    def getLR(self, device='cuda', dtype=torch.float32):
        raise NotImplementedError


class D1PeriodicCase(Datagenerator):
    def __init__(self, config):
        self.config = config

    def plot(self, x=None):
        if x is None:
            nx = 100
            L = 1
            dx = 1/nx
            x = torch.linspace(dx/2, L-dx/2, nx)
        rho, u, T = self.get_prim(x)
        plt.plot(x, rho, label="rho")
        plt.plot(x, u, label="u")
        plt.plot(x, T, label="T")
        plt.legend()

    def init_value(self, device='cuda', dtype=torch.float32):
        v, w, vL, wL = get_vmsh(self.config)
        v = torch.from_numpy(v).to(dtype=dtype)
        IV_xc = fvmlinspace(self.config.xtmesh.xmin,
                            self.config.xtmesh.xmax, self.config.xtmesh.nx)
        rho_l, u_l, T_l = self.get_prim(IV_xc)
        f0 = maxwellian(v, rho_l, u_l, T_l)
        f0 = f0.unsqueeze(0).repeat(1, 1, 1).to(device=device)
        return f0

    def get(self, device='cuda', dtype=torch.float32):
        config = self.config
        v, w, vL, wL = get_vmsh(config)

        VDIS = torch.from_numpy(v).to(device=device, dtype=dtype)
        WDIS = torch.tensor(w).to(device=device, dtype=dtype)
        nx = config["dataset"]["IV_points"]
        nt = config["dataset"]["BV_points"]//2

        xmin = config["xtmesh"]["xmin"]
        xmax = config["xtmesh"]["xmax"]
        tmax = config["xtmesh"]["tmax"]
        xL = xmax-xmin

        IV_x = torch.linspace(xmin, xmax, nx)[..., None].to(device=device)
        IV_t = torch.zeros_like(IV_x).to(device=device)

        rho_l, u_l, T_l = self.get_prim(IV_x[..., 0])
        rho_l = rho_l.to(device=device)
        u_l = u_l.to(device=device)
        T_l = T_l.to(device=device)

        f_l = maxwellian(VDIS, rho_l, u_l, T_l)
        IV_f = f_l

        # 边值
        BV_x1 = xmin*torch.ones(nt)[..., None].to(device=device)
        BV_t1 = torch.linspace(0, tmax, nt)[..., None].to(device=device)

        BV_x2 = xmax*torch.ones(nt)[..., None].to(device=device)
        BV_t2 = torch.linspace(0, tmax, nt)[..., None].to(device=device)

        BV_x = torch.cat((BV_x1, BV_x2))
        BV_t = torch.cat((BV_t1, BV_t2))

        # 内点
        Np = config["dataset"]["IN_points"]
        IN_x = torch.rand([Np, 1], device=device)*xL+xmin
        IN_t = torch.rand([Np, 1], device=device)*tmax
        return (IV_x, IV_t, IV_f), (BV_x1, BV_t1, BV_x2, BV_t2), (IN_x, IN_t)

    def getLR(self, device='cuda', dtype=torch.float32):
        config = self.config
        v, w, vL, wL = get_vmsh(config)

        VDIS = torch.from_numpy(v).to(device=device, dtype=dtype)
        WDIS = torch.tensor(w).to(device=device, dtype=dtype)

        nvx, nvy, nvz = self.config.vmesh.nv
        vmax, vmin = self.config.vmesh.vmax, self.config.vmesh.vmin
        vx = VDIS[:, 0].reshape(nvx, nvy, nvz)[:, 0, 0]
        vy = VDIS[:, 1].reshape(nvx, nvy, nvz)[0, :, 0]
        vz = VDIS[:, 2].reshape(nvx, nvy, nvz)[0, 0, :]

        wx = torch.ones((nvx,))/nvx*(vmax[0]-vmin[0])
        wy = torch.ones((nvy,))/nvy*(vmax[1]-vmin[1])
        wz = torch.ones((nvz,))/nvz*(vmax[2]-vmin[2])
        VT = (vx.to(device=device).float(), vy.to(
            device=device).float(), vz.to(device=device).float())
        WT = (wx.to(device=device).float(), wy.to(
            device=device).float(), wz.to(device=device).float())

        nx = config["dataset"]["IV_points"]
        nt = config["dataset"]["BV_points"]//2

        xmin = config["xtmesh"]["xmin"]
        xmax = config["xtmesh"]["xmax"]
        tmax = config["xtmesh"]["tmax"]
        xL = xmax-xmin

        IV_x = torch.linspace(xmin, xmax, nx)[..., None].to(device=device)
        IV_t = torch.zeros_like(IV_x).to(device=device)

        rho_l, u_l, T_l = self.get_prim(IV_x[..., 0])
        rho_l = rho_l.to(device=device)
        u_l = u_l.to(device=device)
        T_l = T_l.to(device=device)

        f_l = maxwellian_LR(VT, rho_l, u_l, T_l)
        IV_f = f_l

        # 边值
        BV_x1 = xmin*torch.ones(nt)[..., None].to(device=device)
        BV_t1 = torch.linspace(0, tmax, nt)[..., None].to(device=device)

        BV_x2 = xmax*torch.ones(nt)[..., None].to(device=device)
        BV_t2 = torch.linspace(0, tmax, nt)[..., None].to(device=device)

        BV_x = torch.cat((BV_x1, BV_x2))
        BV_t = torch.cat((BV_t1, BV_t2))

        # 内点
        Np = config["dataset"]["IN_points"]
        IN_x = torch.rand([Np, 1], device=device)*xL+xmin
        IN_t = torch.rand([Np, 1], device=device)*tmax
        return (IV_x, IV_t, IV_f), (BV_x1, BV_t1, BV_x2, BV_t2), (IN_x, IN_t)


class SmoothD1Case(D1PeriodicCase):
    def __init__(self, config):
        super().__init__(config)

    def get_prim(self, x):
        rho_l = torch.sin(2*np.pi*x)*0.5+1
        T_l = torch.sin(2*np.pi*x+0.2)*0.5+1

        rho_l = rho_l[..., None]
        u_l = 0*torch.ones(x.shape[0], 3)
        T_l = T_l[..., None]
        return rho_l, u_l, T_l


def sinfunc(x, a, b, c, k):
    return a*torch.sin(2*k*np.pi*x+c)+b


class SinK():
    def __init__(self, k, knum, seed=0, brange=None, arange=None):
        np.random.seed(seed)
        self.k = k
        self.knum = knum
        if arange is None:
            arange = (0.2, 0.4)
        if brange is None:
            brange = (0.8, 1.2)
        self.arange = arange
        self.brange = brange

        self.klist = 1+np.random.randint(0, k, size=(knum,))
        self.clist = 2*np.pi*np.random.rand(knum)
        self.blist = np.random.rand(
            knum)*(self.brange[1]-self.brange[0])+self.brange[0]
        self.alist = np.random.rand(
            knum)*(self.arange[1]-self.arange[0])+self.arange[0]

    def __call__(self, x):
        return sum([sinfunc(x, a, b, c, k) for (a, b, c, k) in zip(self.alist, self.blist, self.clist, self.klist)])


class RandomWaveD1Case(D1PeriodicCase):
    def __init__(self, config, k, knum, seed=0, *args, **kargs):
        super().__init__(config)
        self.rho = SinK(k, knum, seed, *args, **kargs)
        self.u = SinK(k, knum, seed+1234, *args, **kargs)
        self.T = SinK(k, knum, seed+12345, *args, **kargs)
        self.knum = knum

    def get_prim(self, x):
        rho = self.rho(x)/self.knum
        u = self.u(x)
        u = u-(rho*u).mean()/rho.mean()
        u_l = torch.zeros((x.shape[0], 3))
        T = self.T(x)/self.knum
        return rho[..., None], u_l, T[..., None]


class D1DirichletCase():
    def __init__(self, config):
        self.config = config

    def plot(self, x=None):
        if x is None:
            nx = 100
            L = 1
            dx = 1/nx
            x = torch.linspace(dx/2, L-dx/2, nx)
        rho, u, T = self.get_prim(x)
        plt.plot(x, rho, label="rho")
        plt.plot(x, u, label="u")
        plt.plot(x, T, label="T")
        plt.legend()

    def init_value(self, device='cuda', dtype=torch.float32):
        v, w, vL, wL = get_vmsh(self.config)
        v = torch.from_numpy(v).to(dtype=dtype)
        IV_xc = fvmlinspace(self.config.xtmesh.xmin,
                            self.config.xtmesh.xmax, self.config.xtmesh.nx)
        rho_l, u_l, T_l = self.get_prim(IV_xc)
        f0 = maxwellian(v, rho_l, u_l, T_l)
        f0 = f0.unsqueeze(0).repeat(1, 1, 1).to(device=device)
        return f0

    def get(self, device='cuda', dtype=torch.float32):
        config = self.config
        v, w, vL, wL = get_vmsh(config)
        nv = config.vmesh.nv

        VDIS = torch.from_numpy(v).to(device=device, dtype=dtype)
        WDIS = torch.tensor(w).to(device=device, dtype=dtype)
        nx = config["dataset"]["IV_points"]
        nt = config["dataset"]["BV_points"]//2

        xmin = config["xtmesh"]["xmin"]
        xmax = config["xtmesh"]["xmax"]
        tmax = config["xtmesh"]["tmax"]
        xL = xmax-xmin

        IV_x = torch.linspace(-0.5, 0.5, nx)[..., None].to(device=device)
        IV_t = torch.zeros_like(IV_x).to(device=device)

        rho_l, u_l, T_l = self.get_prim(IV_x[..., 0])
        rho_l = rho_l.to(device=device)
        u_l = u_l.to(device=device)
        T_l = T_l.to(device=device)

        IV_f = maxwellian(VDIS, rho_l, u_l, T_l)

        # 边值
        BV_x1 = -0.5*torch.ones(nt)[..., None].to(device=device)
        BV_t1 = torch.linspace(0, 0.1, nt)[..., None].to(device=device)
        BV_f1 = IV_f[0:1, ...].expand(nt, -1)

        BV_x2 = 0.5*torch.ones(nt)[..., None].to(device=device)
        BV_t2 = torch.linspace(0, 0.1, nt)[..., None].to(device=device)
        BV_f2 = IV_f[-1:, ...].expand(nt, -1)

        BV_x = torch.cat((BV_x1, BV_x2))
        BV_t = torch.cat((BV_t1, BV_t2))
        BV_f = torch.cat((BV_f1, BV_f2))

        # 内点
        Np = config.dataset['IN_points']
        IN_x = torch.rand([Np, 1], device=device)-0.5
        IN_t = torch.rand([Np, 1], device=device)*0.1
        return (IV_x, IV_t, IV_f), (BV_x, BV_t, BV_f), (IN_x, IN_t)

    def getLR(self, device='cuda', dtype=torch.float32):
        config = self.config
        v, w, vL, wL = get_vmsh(config)

        VDIS = torch.from_numpy(v).to(device=device, dtype=dtype)
        WDIS = torch.tensor(w).to(device=device, dtype=dtype)

        nvx, nvy, nvz = self.config.vmesh.nv
        vmax, vmin = self.config.vmesh.vmax, self.config.vmesh.vmin
        vx = VDIS[:, 0].reshape(nvx, nvy, nvz)[:, 0, 0]
        vy = VDIS[:, 1].reshape(nvx, nvy, nvz)[0, :, 0]
        vz = VDIS[:, 2].reshape(nvx, nvy, nvz)[0, 0, :]

        wx = torch.ones((nvx,))/nvx*(vmax[0]-vmin[0])
        wy = torch.ones((nvy,))/nvy*(vmax[1]-vmin[1])
        wz = torch.ones((nvz,))/nvz*(vmax[2]-vmin[2])
        VT = (vx.to(device=device).float(), vy.to(
            device=device).float(), vz.to(device=device).float())
        WT = (wx.to(device=device).float(), wy.to(
            device=device).float(), wz.to(device=device).float())

        nx = config["dataset"]["IV_points"]
        nt = config["dataset"]["BV_points"]//2

        xmin = config["xtmesh"]["xmin"]
        xmax = config["xtmesh"]["xmax"]
        tmax = config["xtmesh"]["tmax"]
        xL = xmax-xmin

        IV_x = torch.linspace(xmin, xmax, nx)[..., None].to(device=device)
        IV_t = torch.zeros_like(IV_x).to(device=device)

        rho_l, u_l, T_l = self.get_prim(IV_x[..., 0])
        rho_l = rho_l.to(device=device)
        u_l = u_l.to(device=device)
        T_l = T_l.to(device=device)

        f_l = maxwellian_LR(VT, rho_l, u_l, T_l)
        IV_f = f_l

        # 边值
        BV_x1 = xmin*torch.ones(nt)[..., None].to(device=device)
        BV_t1 = torch.linspace(0, tmax, nt)[..., None].to(device=device)

        BV_x2 = xmax*torch.ones(nt)[..., None].to(device=device)
        BV_t2 = torch.linspace(0, tmax, nt)[..., None].to(device=device)

        BV_x = torch.cat((BV_x1, BV_x2))
        BV_t = torch.cat((BV_t1, BV_t2))

        rho_b, u_b, T_b = self.get_prim(BV_x[..., 0])
        rho_b = rho_b.to(device=device)
        u_b = u_b.to(device=device)
        T_b = T_b.to(device=device)

        BV_f = maxwellian_LR(VT, rho_b, u_b, T_b)

        # 内点
        Np = config["dataset"]["IN_points"]
        IN_x = torch.rand([Np, 1], device=device)*xL+xmin
        IN_t = torch.rand([Np, 1], device=device)*tmax
        return (IV_x, IV_t, IV_f), (BV_x, BV_t, BV_f), (IN_x, IN_t)


def sigmoidshock(fl, fr, xl, xr, x):
    # fl shape [1,C]
    # fr shape [1,C]
    FL = fl.expand(x.shape[0], -1)
    FR = fr.expand(x.shape[0], -1)
    return torch.sigmoid((x-xl)/(xr-xl))[..., None]*(FR-FL)+FL


def sinshock(fl, fr, xl, xr, x):
    # fl shape [1,C]
    # fr shape [1,C]
    FL = fl.expand(x.shape[0], -1)
    FR = fr.expand(x.shape[0], -1)
    FS = fl+(fr-fl)*(1+torch.sin(-np.pi/2+np.pi*(x-xl)[..., None]/(xr-xl)))/2
    F = FL*(x < xl)[..., None]+FR*(x > xr)[..., None] + \
        FS*(x >= xl)[..., None]*(x <= xr)[..., None]
    return F


class SodD1Case(D1DirichletCase):
    def __init__(self, config):
        self.config = config

    def get_prim(self, x):
        # rho_l = torch.sin(2*np.pi*x)*0.5+1
        # T_l = torch.sin(2*np.pi*x+0.2)*0.5+1

        # rho_l = rho_l[..., None]
        # u_l = 0*torch.ones(x.shape[0], 3)
        # T_l = T_l[..., None]
        # return rho_l, u_l, T_l
        nx = x.shape[0]
        rho_l, u_l, P_l = 1.0, 0., 1
        rho_r, u_r, P_r = 0.125, 0., 0.1
        T_l = P_l/rho_l
        T_r = P_r/rho_r

        rho_l = rho_l*torch.ones(nx, 1)
        u_l = u_l*torch.ones(nx, 3)
        T_l = T_l*torch.ones(nx, 1)
        rho_r = rho_r*torch.ones(nx, 1)
        u_r = u_r*torch.ones(nx, 3)
        T_r = T_r*torch.ones(nx, 1)
        x = x.cpu()
        config = self.config
        rho = sigmoidshock(rho_l[0:1], rho_r[-1:], -
                           config.ivsmooth.width, config.ivsmooth.width, x)
        u = sigmoidshock(u_l[0:1], u_r[-1:], -
                         config.ivsmooth.width, config.ivsmooth.width, x)
        T = sigmoidshock(T_l[0:1], T_r[-1:], -
                         config.ivsmooth.width, config.ivsmooth.width, x)
        return rho, u, T


class D2PeriodicCase(Datagenerator):
    def __init__(self, config):
        self.config = config

    def plot(self, x=None):
        raise NotImplementedError

    def get(self, device='cuda', dtype=torch.float32):
        config = self.config
        v, w, vL, wL = get_vmsh(config)
        VDIS = torch.from_numpy(v).to(device=device, dtype=dtype)
        WDIS = torch.tensor(w).to(device=device, dtype=dtype)

        Ni = config.dataset["IV_points"]
        IV_x = torch.rand([Ni, 1], device=device)-0.5
        IV_y = torch.rand([Ni, 1], device=device)-0.5
        IV_t = torch.zeros_like(IV_x).to(device=device)

        rho_l,u_l,T_l=self.get_prim(IV_x[...,0],IV_y[...,0])
        rho_l=rho_l.to(device=device)
        u_l=u_l.to(device=device)
        T_l=T_l.to(device=device)
        f_l = maxwellian(VDIS, rho_l, u_l, T_l)
        IV_f = f_l

        # 边值
        Nv = config.dataset["BV_points"]
        BV_x1 = -0.5*torch.ones(Nv)[..., None].to(device=device)
        BV_y1 = torch.rand([Nv, 1], device=device)-0.5
        BV_t1 = torch.rand([Nv, 1], device=device)*0.1

        BV_x2 = 0.5*torch.ones(Nv)[..., None].to(device=device)
        BV_y2 = BV_y1
        BV_t2 = BV_t1

        BV_x3 = torch.rand([Nv, 1], device=device)-0.5
        BV_y3 = -0.5*torch.ones(Nv)[..., None].to(device=device)
        BV_t3 = torch.rand([Nv, 1], device=device)*0.1

        BV_x4 = BV_x3
        BV_y4 = 0.5*torch.ones(Nv)[..., None].to(device=device)
        BV_t4 = BV_t3

        BV_xa = torch.cat((BV_x1, BV_x3))
        BV_ya = torch.cat((BV_y1, BV_y3))
        BV_ta = torch.cat((BV_t1, BV_t3))

        BV_xb = torch.cat((BV_x2, BV_x4))
        BV_yb = torch.cat((BV_y2, BV_y4))
        BV_tb = torch.cat((BV_t2, BV_t4))

        # 内点
        Np = config.dataset["IN_points"]
        IN_x = torch.rand([Np, 1], device=device)-0.5
        IN_y = torch.rand([Np, 1], device=device)-0.5
        IN_t = torch.rand([Np, 1], device=device)*0.1
        return (IV_x, IV_y, IV_t, IV_f), (BV_xa, BV_ya, BV_ta, BV_xb, BV_yb, BV_tb), (IN_x, IN_y, IN_t)

    def getLR(self, device='cuda', dtype=torch.float32):
        config=self.config

        v, w, vL, wL = get_vmsh(config)
        VDIS = torch.from_numpy(v).to(device=device, dtype=dtype)
        WDIS = torch.tensor(w).to(device=device, dtype=dtype)

        nvx, nvy, nvz = self.config.vmesh.nv
        vmax, vmin = self.config.vmesh.vmax, self.config.vmesh.vmin
        vx = VDIS[:, 0].reshape(nvx, nvy, nvz)[:, 0, 0]
        vy = VDIS[:, 1].reshape(nvx, nvy, nvz)[0, :, 0]
        vz = VDIS[:, 2].reshape(nvx, nvy, nvz)[0, 0, :]

        wx = torch.ones((nvx,))/nvx*(vmax[0]-vmin[0])
        wy = torch.ones((nvy,))/nvy*(vmax[1]-vmin[1])
        wz = torch.ones((nvz,))/nvz*(vmax[2]-vmin[2])
        VT = (vx.to(device=device).float(), vy.to(
            device=device).float(), vz.to(device=device).float())
        WT = (wx.to(device=device).float(), wy.to(
            device=device).float(), wz.to(device=device).float())

        # nx = config["dataset"]["IV_points"]
        # nt = config["dataset"]["BV_points"]//2

        xmin = config["xtmesh"]["xmin"]
        xmax = config["xtmesh"]["xmax"]
        tmax = config["xtmesh"]["tmax"]
        xL = xmax-xmin

        Ni=config.dataset["IV_points"]
        IV_x=torch.rand([Ni,1],device=device)-0.5
        IV_y=torch.rand([Ni,1],device=device)-0.5
        IV_t=torch.zeros_like(IV_x).to(device=device)

        rho_l,u_l,T_l=self.get_prim(IV_x[...,0],IV_y[...,0])
        rho_l=rho_l.to(device=device)
        u_l=u_l.to(device=device)
        T_l=T_l.to(device=device)

        f_l=maxwellian_LR(VT, rho_l,u_l,T_l)
        IV_f = f_l

    ## 边值
        Nv=config.dataset["BV_points"]
        BV_x1 = -0.5*torch.ones(Nv)[...,None].to(device=device)
        BV_y1 = torch.rand([Nv,1],device=device)-0.5
        BV_t1 = torch.rand([Nv,1],device=device)*0.1

        BV_x2 = 0.5*torch.ones(Nv)[...,None].to(device=device)
        BV_y2 = BV_y1
        BV_t2 = BV_t1

        BV_x3 = torch.rand([Nv,1],device=device)-0.5
        BV_y3 = -0.5*torch.ones(Nv)[...,None].to(device=device)
        BV_t3 = torch.rand([Nv,1],device=device)*0.1

        BV_x4 = BV_x3
        BV_y4 = 0.5*torch.ones(Nv)[...,None].to(device=device)
        BV_t4 = BV_t3

        BV_xa=torch.cat((BV_x1,BV_x3))
        BV_ya=torch.cat((BV_y1,BV_y3))
        BV_ta=torch.cat((BV_t1,BV_t3))

        BV_xb=torch.cat((BV_x2,BV_x4))
        BV_yb=torch.cat((BV_y2,BV_y4))
        BV_tb=torch.cat((BV_t2,BV_t4))

        ## 内点
        Np=config.dataset["IN_points"]
        IN_x=torch.rand([Np,1],device=device)-0.5
        IN_y=torch.rand([Np,1],device=device)-0.5
        IN_t=torch.rand([Np,1],device=device)*0.1

        return (IV_x, IV_y, IV_t, IV_f), (BV_xa, BV_ya, BV_ta, BV_xb, BV_yb, BV_tb), (IN_x, IN_y, IN_t)

class SmoothD2Case(D2PeriodicCase):
    def get_prim(self, x, y):
        rho = 0.5*torch.sin(2*np.pi*x)*torch.sin(2*np.pi*y)+1
        N = x.shape[0]
        u = 0*torch.ones(N, 3)
        T = torch.ones(N, 1)
        return rho[...,None],u,T

class SmoothD2Case15(D2PeriodicCase):
    def get_prim(self, x, y):
        rho = 0.4*torch.sin(2*np.pi*x+0.3)*torch.sin(2*np.pi*y+0.4)+1
        N = x.shape[0]
        u = 0*torch.ones(N, 3)
        T = torch.ones(N, 1)
        return rho[...,None],u,T

class SmoothD2Case2(D2PeriodicCase):
    def get_prim(self, x, y):
        rho = 0.4*torch.sin(2*np.pi*x+0.3*2*np.pi)*torch.sin(2*np.pi*y+0.4*2*np.pi)+1
        N = x.shape[0]
        u = 0*torch.ones(N, 3)
        T = torch.ones(N, 1)
        return rho[...,None],u,T