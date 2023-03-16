import sys
import os

sys.path.insert(0, os.path.abspath('../..'))

import logging
logger = logging.getLogger(__name__)
from bte.nsr.model_LR import SplitNet3D,SplitNet_3D_LRNew
from bte.nsr.model_LR import *
from bte.nsr.model_LR import _m012_LR
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
torch.set_default_dtype(torch.float32)

import torch.nn as nn
import torch.functional as F
import numpy as np
from matplotlib import pyplot as plt
import time


import torch
import math
from bte.dvm import solver as bgk_solver
from bte.dvm.collision import init_kernel_mode_vector
from bte.utils.gas import get_potential,get_mu,get_kn_bzm
from matplotlib import pyplot as plt
import torch
torch.set_default_dtype(torch.float32)

from bte.nsr.dataset import SmoothD1Case,SodD1Case,SmoothD2Case
from bte.nsr.utils import orthonormalize
device="cuda"
from torch.fft import fftn, ifftn, fftshift

@hydra.main(version_base='1.2', config_path="config", config_name="Wave_D2V3")
def train(config):
    print(config)
    nv=config["vmesh"]["nv"]
    vmin=config["vmesh"]["vmin"]
    vmax=config["vmesh"]["vmax"]
    Kn=config["Kn"]

    kn = Kn
    
    omega = 0.81
    alpha = get_potential(omega)
    mu_ref = get_mu(alpha,omega,kn)
    kn_bzm = get_kn_bzm(alpha,mu_ref)

    nx=config.xtmesh.nx
    xmin,xmax=config.xtmesh.xmin,config.xtmesh.xmax

    if config.case=="Wave":
        datagen=SmoothD1Case(config)
        (IV_x,IV_t,IV_f),(BV_x1,BV_t1,BV_x2,BV_t2),(IN_x,IN_t)=datagen.getLR()
        BDC="periodic"
    else:
        datagen=SodD1Case(config)
        (IV_x,IV_t,IV_f),(BV_x,BV_t,BV_f),(IN_x,IN_t)=datagen.getLR()
        BDC="constant"

    if config["collision"]=="BGK":
        solver = bgk_solver.BGKSolver(xmin,xmax,nx,vmin,vmax,nv,device='cpu',BC_type=BDC,bgk_simple_kn="simple")
    else:
        raise ValueError

    solver.mu_ref=mu_ref
    solver.omega=omega
    print("x.dtype:",  solver.dis.x.dtype)
    solver.set_space_order(2)
    
    if config["collision"] in ["BGK","FBGK"]:
        solver.set_time_stepper("bgk-RK2")
        solver.set_collisioner("BGK")
    else:
        raise ValueError

    x, v = solver.dis.x.cpu(), solver.dis.v.cpu()

    f0=datagen.init_value()
    solver.to(device=device)

    solver.set_initial(kn, f0)
    solver.dis.v_meta=solver.dis.v_meta.to(device=device)
    solver.dis=solver.dis.to(device=device)

    fig=plt.figure()
    plt.plot(solver.dis.density().cpu()[0,:,0],label='rho')
    plt.plot(solver.dis.velocity().cpu()[0,:,0],label='ux')
    plt.plot(solver.dis.velocity().cpu()[0,:,1],label='uy')
    plt.plot(solver.dis.temperature().cpu()[0,:,0],label='T')
    plt.legend()
    plt.savefig("DVM-t0.png")

    t_final, dt = 0.1, 0.01
    CFL = 0.45
    max_dt = CFL * 1.0 / nx / vmax[0]
    print("max_dt is", max_dt)
    soln = []
    nt=10
    paths2=torch.zeros((nt,nx,math.prod(nv)))
    for t in range(nt):
        print(t)
        solver.solve_to(dt, max_dt)
        paths2[t]=solver.dis.f.cpu()
        
    print(solver.dis.density().shape)
    print(solver.dis.density().max())

    fig=plt.figure()
    plt.plot(solver.dis.density().cpu()[0,:,0],label='rho')
    plt.plot(solver.dis.velocity().cpu()[0,:,0],label='ux')
    plt.plot(solver.dis.velocity().cpu()[0,:,1],label='uy')
    plt.plot(solver.dis.temperature().cpu()[0,:,0],label='T')
    plt.legend()
    plt.savefig("DVM-t1.png")

    v=solver.dis.v_meta.v.float()
    w=solver.dis.v_meta.w.float()

    VDIS=v
    WDIS=w

    nvx,nvy,nvz=nv
    vx=v[:,0].reshape(nvx,nvy,nvz)[:,0,0]
    vy=v[:,1].reshape(nvx,nvy,nvz)[0,:,0]
    vz=v[:,2].reshape(nvx,nvy,nvz)[0,0,:]
    vtuple=(vx,vy,vz)

    wx=torch.ones((nvx,))/nvx*20
    wy=torch.ones((nvy,))/nvy*20
    wz=torch.ones((nvz,))/nvz*20

    VT=(vx.to(device=device).float(),vy.to(device=device).float(),vz.to(device=device).float())
    WT=(wx.to(device=device).float(),wy.to(device=device).float(),wz.to(device=device).float())

    hidden=80
    rank=config.network.rank

    ###############
    class ResBlock(nn.Module):
        def __init__(self, hidden, activation=torch.sin):
            super().__init__()
            self.fc1=nn.Linear(hidden,hidden)
            self.act=activation

        def forward(self, x):
            x=self.act(math.sqrt(6)*self.fc1(x))
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
            
            self.act=torch.sin

        def forward(self, x):
            x=self.bn1(self.layer1(x))
            x=torch.sin(x)
            x=self.bn2(self.layer2(x))
            x=self.bn3(self.layer3(x))
            x=self.bn4(self.layer4(x))
            x=self.bn4(self.layer5(x))
            x=self.layer6(x)
            return x

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
        def __init__(self, in_channel, out_channel1, out_channel2,out_channel3,rank, hidden):
            super().__init__()
            self.net_eq=NetBlock(in_channel, 5, hidden)
            self.net_neq1=NetBlock(in_channel, (out_channel1+ out_channel2+ out_channel3)*rank, hidden)
            self.out_channel1=out_channel1
            self.out_channel2=out_channel2
            self.out_channel3=out_channel3
        def forward(self, x):
            www=self.net_eq(x)
            rho,u,theta=www[...,0:1],www[...,1:4],www[...,4:5]
            rho=torch.exp(-rho)
            theta=torch.exp(-theta)
            fmx,fmy,fmz=maxwellian_LR(VT,rho,u,theta)
            f2=self.net_neq1(x)
            f2x=f2[...,:self.out_channel1*rank]
            f2y=f2[...,self.out_channel1*rank:(self.out_channel1+self.out_channel2)*rank]
            f2z=f2[...,(self.out_channel1+self.out_channel2)*rank:(self.out_channel1+self.out_channel2+self.out_channel3)*rank]
            
            f2x=f2x.reshape(f2x.shape[:-1]+(self.out_channel1,rank))
            f2x=(0.01**0.33)*fmx*f2x
            
            f2y=f2y.reshape(f2y.shape[:-1]+(self.out_channel2,rank))
            f2y=(0.01**0.33)*fmy*f2y
            
            f2z=f2z.reshape(f2z.shape[:-1]+(self.out_channel3,rank))
            f2z=(0.01**0.33)*fmz*f2z
            
            yx,yy,yz=torch.cat([fmx**2,f2x],dim=-1),torch.cat([fmy**2,f2y],dim=-1),torch.cat([fmz**2,f2z],dim=-1)

            return yx,yy,yz

    rank=config.network["rank"]
    hidden=80

    model=SplitNet(3,nvx,nvy,nvz,rank,hidden)
    #############
    #############
    from bte.nsr.visual_utils import read_data, fvmlinspace, fvmlinspace_t
    nx=80
    nxt=80
    IV_x0=fvmlinspace_t(-0.5,0.5,nxt).to(device=device)
    IV_y0=fvmlinspace_t(-0.5,0.5,nxt).to(device=device)
    IV_x0, IV_y0 = torch.meshgrid(IV_x0, IV_y0, indexing='xy')
    IV_x0=IV_x0[...,None]
    IV_y0=IV_y0[...,None]
    IV_t0=torch.zeros_like(IV_x0).to(device=device)
    rho_l0=0.4*torch.sin(2*np.pi*IV_x0+2*np.pi*0.3)*torch.sin(2*np.pi*IV_y0+2*np.pi*0.4)+1
    T_l0=1
    Ni0=IV_x0.shape[0]
    rho_l0=rho_l0*torch.ones(Ni0,1).to(device=device)
    u_l0=0*torch.ones(Ni0,3).to(device=device)
    T_l0=T_l0*torch.ones(Ni0,1).to(device=device)
    #print(VDIS.shape,rho_l0.shape,u_l0.shape,T_l0.shape)
    f_l0=maxwellian(VDIS, rho_l0, u_l0, T_l0)
    IV_f0 = f_l0
    
    rho,u,theta=rho_l0,u_l0,T_l0

    if config["collision"]=="BGK":
        data80=read_data(f"/home/zyli/NeuralBoltzmann/fortran/cylinder-sbgk-80-{config['Kn']}.plt")
        rhoGT,uGT,thetaGT=data80["RHO"],data80["U"],data80["T"]
    else:
        data80=read_data(f"/home/zyli/NeuralBoltzmann/fortran/cylinder-fsm-80-{config['Kn']}.plt")
        rhoGT,uGT,thetaGT=data80["RHO"],data80["U"],data80["T"]
        
    model=model.to(device=device)
    vdis=v

    def Net(x,y,t):
        inputs=torch.cat([x,y,t],dim=-1)
        fx,fy,fz=model(inputs)
        return fx.sum(dim=0),fy.sum(dim=0),fz.sum(dim=0)

    def Recons(P,Q,R):
        return torch.einsum("...ir,...jr,...kr->...ijk",P,Q,R).flatten(-3,-1)

    def LRMSE(P,Q,R):
        return 0.5*((P.mT@P)*(Q.mT@Q)*(R.mT@R)).sum()

    def LR_add(*T):
        return tuple(torch.cat(t,dim=-1) for t in zip(*T))

    def LR_sub(T1,T2):
        return tuple(torch.cat([t1,-t2],dim=-1) for t1,t2 in zip(T1,T2))

    def LR_to(T,**args):
        return  tuple(t.to(**args) for t in T)

    def net_LR(x,y,t):
        inputs=torch.cat([x,y,t],dim=-1)
        return model(inputs)

    from functorch import grad,jacfwd,vmap,jacrev

    ggrad=vmap(jacfwd(net_LR,(0,1,2)))

    def Net_LRG(x,y,t):
        (Px,Py,Pt),(Qx,Qy,Qt),(Rx,Ry,Rt)=ggrad(x, y, t)
        Px=Px[...,0]
        Py=Py[...,0]
        Pt=Pt[...,0]
        Qx=Qx[...,0]
        Qy=Qy[...,0]
        Qt=Qt[...,0]
        Rx=Rx[...,0]
        Ry=Ry[...,0]
        Rt=Rt[...,0]
        return (Px,Py,Pt),(Qx,Qy,Qt),(Rx,Ry,Rt)

    def pinn_LR(x, y, t, net, Kn=Kn):
        P,Q,R = net(x, y, t)
        (Px,Py,Pt),(Qx,Qy,Qt),(Rx,Ry,Rt)= Net_LRG(x, y, t)
        rho,u,theta = rho_u_theta_LR((P,Q,R),VT,WT)
        f_Mx, f_My, f_Mz  = maxwellian_LR(VT, rho,u,theta)

        sqrtKn=math.sqrt(Kn)
        vx=VT[0]
        vy=VT[1]
        Knr3=Kn**(1/3)
        # 可以用一些技巧缩减到4项

        Ft=[[P,Qt,R],[P,Q,Rt],[Pt,Q,R]]
        vFx=[[Px*vx[...,None],Q,R],[P*vx[...,None],Qx,R],[P*vx[...,None],Q,Rx]]
        vFy=[[Py,Q*vy[...,None],R],[P,Qy*vy[...,None],R],[P,Q*vy[...,None],Ry]]
        FM=[[-1/Knr3*f_Mx, -1/Knr3*f_My,-1/Knr3* f_Mz]]
        F=[[1/Knr3*P,1/Knr3*Q,1/Knr3*R]]
        
        Terms=Ft+vFx+vFy+FM+F
        return LR_add(*Terms)

    smooth_l1_loss =torch.nn.MSELoss(reduction='none')
    criterion=lambda x,y:0.5*smooth_l1_loss(x,y).mean(dim=0)
    criterion_norm=lambda x:0.5*smooth_l1_loss(x,torch.zeros_like(x)).mean(dim=0)

    def prim_norm_LR(P,Q,R):
        m1,m2,m3=_m012_LR((P,Q,R),VT,WT)
        return torch.cat([criterion_norm(m1),criterion_norm(m2),criterion_norm(m3)],dim=-1)

    # 构造光滑初值问题
    class LossCompose(nn.Module):
        def __init__(self, n_class, eta=1e-3):
            super().__init__()
            self.n_class=n_class
            self.w=nn.parameter.Parameter(torch.ones(n_class))
            self.register_buffer("eta",torch.Tensor([eta]))

        def forward(self, loss):
            assert loss.shape[-1]==self.n_class
            ww=self.eta**2+self.w**2
            Loss=1/2/ww*loss+torch.log(1+ww)
            return Loss.sum()#/self.n_class

    wc=(7*1)*3
    print(wc)
    LC3=LossCompose(wc).to(device=device)

    def LRMSE_ADAP(P,Q,R,W1,W2,W3):
        W1=W1[None,...,None].sqrt()
        W2=W2[None,...,None].sqrt()
        W3=W3[None,...,None].sqrt()
        P=P*W1
        Q=Q*W2
        R=R*W3
        return 0.5*((P.mT@P)*(Q.mT@Q)*(R.mT@R)).sum()/P.shape[0]

    class Adaptive_MSE(nn.Module):
        def __init__(self, nx, ny, nz, eta=1e-6):
            super().__init__()
            self.nx=nx
            self.ny=ny
            self.nz=nz
            self.w1=nn.parameter.Parameter(torch.ones(nx))
            self.w2=nn.parameter.Parameter(torch.ones(ny))
            self.w3=nn.parameter.Parameter(torch.ones(nz))
            self.register_buffer("eta",torch.Tensor([eta]))
        def forward(self,P,Q,R):
            W1 = self.eta**2+self.w1**2
            W2 = self.eta**2+self.w2**2
            W3 = self.eta**2+self.w3**2
            W = W1[:,None,None]*W2[None,:,None]*W3[None,None,:]
            Loss1,LossW = LRMSE_ADAP(P,Q,R,1/2/W1,1/2/W2,1/2/W3),torch.log(1+W).sum()
            return Loss1,LossW
        
    LA=Adaptive_MSE(nvx,nvy,nvz).to(device=device)
    LA1=Adaptive_MSE(nvx,nvy,nvz).to(device=device)
    LA2=Adaptive_MSE(nvx,nvy,nvz).to(device=device)
    class problem_lr(nn.Module):
        def __init__(self,net,la1,la2,la3,lc):
            super(problem_lr,self).__init__()
            self.net=net
            self.la1=la1
            self.la2=la2
            self.la3=la3
            self.lc=lc
    
    problem=problem_lr(model,LA,LA1,LA2,LC3)
    start_time=time.time()
    lr = config.optim.lr_scheduler.max_lr
    optimizer=torch.optim.Adam( 
                    [{'params': model.parameters(), 'lr':lr},
                    {'params': LC3.parameters(), 'lr': lr},
                    {'params': LA.parameters(), 'lr': lr},{'params': LA1.parameters(), 'lr': lr},{'params': LA2.parameters(), 'lr': lr}])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.optim["Adam_steps"], eta_min=config.optim.lr_scheduler.min_lr, last_epoch=-1)

    train_flag=True
    model.train()

    import pytorch_warmup as warmup
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=config.optim["Adam_steps"]//10)

    datasampler=SmoothD2Case(config)
    for i in range(config.optim["Adam_steps"]):
        (IV_x,IV_y,IV_t,IV_f),(BV_xa,BV_ya,BV_ta,BV_xb,BV_yb,BV_tb),(IN_x,IN_y,IN_t)=datasampler.getLR()
        IV_pred=net_LR(IV_x,IV_y,IV_t)
        IV_diff=LR_sub(IV_pred,IV_f)

        loss_IV,loss_IVW=LA1(math.sqrt(config.network.weight[0])*IV_diff[0],IV_diff[1],IV_diff[2])
        loss_IV2=prim_norm_LR(math.sqrt(config.network.weight[0])*IV_diff[0],IV_diff[1],IV_diff[2])
            
        BV_pred1=net_LR(BV_xa,BV_ya,BV_ta)
        BV_pred2=net_LR(BV_xb,BV_yb,BV_tb)
        BV_diff=LR_sub(BV_pred1,BV_pred2)
        
        loss_BV,loss_BVW=LA2(math.sqrt(config.network.weight[1])*BV_diff[0],BV_diff[1],BV_diff[2])
        loss_BV2=prim_norm_LR(math.sqrt(config.network.weight[1])*BV_diff[0],BV_diff[1],BV_diff[2])
        
        nKn=Kn
        P,Q,R=pinn_LR(IN_x, IN_y, IN_t, net_LR)

        loss_pinn,loss_pinnW=LA(math.sqrt(config.network.weight[2])*P,Q,R)
        loss_pinn2=prim_norm_LR(math.sqrt(config.network.weight[2])*P,Q,R)
            
        loss_sum=(LC3(torch.cat([loss_IV2,loss_BV2,loss_pinn2]))+loss_IV+loss_IVW+loss_BV+loss_BVW+loss_pinn+loss_pinnW)/(3*7+nvx*nvy*nvz*3)

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        with warmup_scheduler.dampening():
            scheduler.step()    

        if i%10==0:
            e_sum=loss_sum.item()
            e_IV=loss_IV.mean().item()/(nvx*nvy*nvz)
            e_IV2=loss_IV2.mean().item()
            e_BV=loss_BV.mean().item()/(nvx*nvy*nvz)
            e_BV2=loss_BV2.mean().item()
            e_pinn=loss_pinn.mean().item()/(nvx*nvy*nvz)
            e_pinn2=loss_pinn2.mean().item()
            logger.info(f"{i},{time.time()-start_time:.2e}:{e_sum:.2e},{e_IV:.2e},{e_IV2:.2e},{e_BV:.2e},{e_BV2:.2e},{e_pinn:.2e},{e_pinn2:.2e}")
        
        if i%100==0:
            IV_pred=net_LR(IV_x0,IV_y0,IV_t0+0.0)
            rho1,u1,theta1=rho_u_theta_LR(IV_pred,VT,WT)
            err1=(((rho1-rho)**2).mean()/((rho)**2).mean())**0.5
            err2=(((u1-u)**2).mean()/(1+(u)**2).mean())**0.5
            err3=(((theta1-theta)**2).mean()/((theta)**2).mean())**0.5
            logger.info(f"err at t=0.0: {err1.item():.2e}\t{err2.item():.2e}\t{err3.item():.2e}\t")
            
            IV_pred=net_LR(IV_x0,IV_y0,IV_t0+0.1)
            rho1,u1,theta1=rho_u_theta_LR(IV_pred,VT,WT)
            print(rho1.cpu().detach().numpy()[...,0].shape,rhoGT.shape)
            err1=(((rho1.cpu().detach().numpy()[...,0]-rhoGT)**2).mean()/((rhoGT)**2).mean())**0.5
            err2=(((u1.cpu().detach().numpy()[...,0]-uGT)**2).mean()/(1+(uGT)**2).mean())**0.5
            err3=(((theta1.cpu().detach().numpy()[...,0]-thetaGT)**2).mean()/((thetaGT)**2).mean())**0.5
            logger.info(f"err at t=0.1: {err1.item():.2e}\t{err2.item():.2e}\t{err3.item():.2e}\t")
        
        if i in [10,100,200,300,400,500,750,1000,2000,3000,4000,5000,6000,7000,8000,9000]:
            torch.save(model.state_dict(), f"model_epoch{i}.ckpt")
            torch.save(problem.state_dict(), f"problem_epoch{i}.ckpt")
    model.eval()

    torch.save(model.state_dict(), "model.ckpt")
    torch.save(problem.state_dict(), "problem.ckpt")


if __name__ == "__main__":
    train()