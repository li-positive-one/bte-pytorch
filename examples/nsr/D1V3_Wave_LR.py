import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

import logging

logger = logging.getLogger(__name__)
import math
import time

import hydra
import numpy as np
import torch
import torch.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt

from torch.func import  jacfwd, vmap

from bte.dvm import solver as bgk_solver
from bte.nsr.dataset import SmoothD1Case, SodD1Case
from bte.nsr.model_LR import *
from bte.utils.gas import get_kn_bzm, get_mu, get_potential

device="cuda"

@hydra.main(version_base='1.2', config_path="config", config_name="BGK_D1V3")
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
            
            self.act=torch.sin#nn.Softplus()

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
            f2y=f2[...,self.out_channel1*rank:self.out_channel1*rank+self.out_channel2*rank]
            f2z=f2[...,self.out_channel1*rank+self.out_channel2*rank:]
            
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

    model=SplitNet(2,nvx,nvy,nvz,rank,hidden)
    #############
    model=model.to(device=device)
    vdis=v

    def Net(x,t):
        inputs=torch.cat([x,t],dim=-1)
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

    def net_LR(x,t):
        inputs=torch.cat([x,t],dim=-1)
        return model(inputs)



    ggrad=vmap(jacfwd(net_LR,(0,1)))

    def Net_LRG(x,t):
        (Px,Pt),(Qx,Qt),(Rx,Rt)=ggrad(x, t)
        Px=Px[...,0]
        Pt=Pt[...,0]
        Qx=Qx[...,0]
        Qt=Qt[...,0]
        Rx=Rx[...,0]
        Rt=Rt[...,0]
        return (Px,Pt),(Qx,Qt),(Rx,Rt)

    def pinn_LR(x, t, net, Kn=Kn):
        P,Q,R = net(x,t)
        (Px,Pt),(Qx,Qt),(Rx,Rt) = Net_LRG(x,t)
        rho,u,theta = rho_u_theta_LR((P,Q,R),VT,WT)
        f_Mx, f_My, f_Mz  = maxwellian_LR(VT, rho,u,theta)

        sqrtKn=math.sqrt(Kn)
        vx=VT[0]
        Knr3=Kn**(1/3)
        # 可以用一些技巧缩减到4项
    #     Ft=[[Pt,Q,R],[P,Qt,R],[P,Q,Rt]]
    #     vFx=[[Px*vx[...,None],Q,R],[P*vx[...,None],Qx,R],[P*vx[...,None],Q,Rx]]
    #     FM=[[-1/Knr3*f_Mx, -1/Knr3*f_My,-1/Knr3* f_Mz]]
    #     F=[[1/Knr3*P,1/Knr3*Q,1/Knr3*R]]
        
        Ft=[[P,Qt,R],[P,Q,Rt]]
        vFx=[[Px*vx[...,None]+Pt,Q,R],[P*vx[...,None],Qx,R],[P*vx[...,None],Q,Rx]]
        FM=[[-1/Knr3*f_Mx, -1/Knr3*f_My,-1/Knr3* f_Mz]]
        F=[[1/Knr3*P,1/Knr3*Q,1/Knr3*R]]
        
        Terms=Ft+vFx+FM+F
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
            return Loss.sum()/self.n_class

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

    start_time=time.time()
    lr = config.optim.lr_scheduler.max_lr
    optimizer=torch.optim.AdamW( 
                    [{'params': model.parameters(), 'lr':lr},
                    {'params': LC3.parameters(), 'lr': lr},
                    {'params': LA.parameters(), 'lr': lr},{'params': LA1.parameters(), 'lr': lr},{'params': LA2.parameters(), 'lr': lr}])
    import pytorch_warmup as warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.optim["Adam_steps"], eta_min=config.optim.lr_scheduler.min_lr, last_epoch=-1)
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=config.optim["Adam_steps"]//10)

    train_flag=True
    model.train()

    for i in range(config.optim["Adam_steps"]):
        if config.case=="Wave":
            (IV_x,IV_t,IV_f),(BV_x1,BV_t1,BV_x2,BV_t2),(IN_x,IN_t)=datagen.getLR()
        else:
            (IV_x,IV_t,IV_f),(BV_x,BV_t,BV_f),(IN_x,IN_t)=datagen.getLR()

        weight_iv=math.sqrt(config.network.weight[0])
        weight_bv=math.sqrt(config.network.weight[1])
        weight_pde=math.sqrt(config.network.weight[2]*min(1,(0.1+2*(i)/config.optim["Adam_steps"])))
        IV_pred=net_LR(IV_x,IV_t)
        IV_diff=LR_sub(IV_pred,IV_f)

        loss_IV,loss_IVW=LA1(weight_iv*IV_diff[0],IV_diff[1],IV_diff[2])
        loss_IV2=prim_norm_LR(weight_iv*IV_diff[0],IV_diff[1],IV_diff[2])
            
        if config.case=="Wave":
            BV_pred1=net_LR(BV_x1,BV_t1)
            BV_pred2=net_LR(BV_x2,BV_t2)
            BV_diff=LR_sub(BV_pred1,BV_pred2)
            
            loss_BV,loss_BVW=LA2(weight_bv*BV_diff[0],BV_diff[1],BV_diff[2])
            loss_BV2=prim_norm_LR(weight_bv*BV_diff[0],BV_diff[1],BV_diff[2])
        elif config.case=="Sod":
            BV_pred=net_LR(BV_x,BV_t)
            BV_diff=LR_sub(BV_pred,BV_f)

            loss_BV,loss_BVW=LA2(weight_bv*BV_diff[0],BV_diff[1],BV_diff[2])
            loss_BV2=prim_norm_LR(weight_bv*BV_diff[0],BV_diff[1],BV_diff[2])
        else:
            raise ValueError        
        nKn=Kn
        P,Q,R=pinn_LR(IN_x, IN_t, net_LR)

        loss_pinn,loss_pinnW=LA(weight_pde*P,Q,R)
        loss_pinn2=prim_norm_LR(weight_pde*P,Q,R)
            
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
            print(optimizer.param_groups[0]['lr'])
            logger.info(f"{i},{time.time()-start_time:.2e}:{e_sum:.2e},{e_IV:.2e},{e_IV2:.2e},{e_BV:.2e},{e_BV2:.2e},{e_pinn:.2e},{e_pinn2:.2e}")
    model.eval()

    torch.save(model.state_dict(), "model.ckpt")

    VA_x=torch.linspace(-0.5,0.5,400)[...,None].to(device=device)
    VA_t=0.0*torch.ones_like(VA_x).to(device=device)
    y=model(torch.concat((VA_x,VA_t),dim=-1))
    rho,u,theta=rho_u_theta_LR(y,VT,WT)
    rho=rho.cpu()
    u=u.cpu()
    theta=theta.cpu()

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(VA_x.cpu()[...,0],rho.detach().cpu(),label=r'NN-$\rho$')
    ax.plot(VA_x.cpu()[...,0],u.detach().cpu()[...,0:1],label=r'NN-$u_x$')
    ax.plot(VA_x.cpu()[...,0],theta.detach().cpu(),label=r'NN-$T$')

    ax.plot(solver.dis.x.cpu(),solver.dis.density(f0).cpu()[0,:,0],"--",label=r'FSM-$\rho$')
    ax.plot(solver.dis.x.cpu(),solver.dis.velocity(f0).cpu()[0,:,0:1],"--",label=r'FSM-$u_x$')
    ax.plot(solver.dis.x.cpu(),solver.dis.temperature(f0).cpu()[0,:,0],"--",label=r'FSM-$T$')
    plt.legend(loc='upper right')
    plt.savefig("t0.png")

    np.savez("t0.npz",rho=rho.detach().cpu(),u=u.detach().cpu()[...,0:1],theta=theta.detach().cpu(),
                rho0=solver.dis.density(f0).cpu()[0,:,0],
                u0=solver.dis.velocity(f0).cpu()[0,:,0:1],
                theta0=solver.dis.temperature(f0).cpu()[0,:,0])

    rho0=solver.dis.density(f0).cpu()[-1,:,0:1]
    u0=solver.dis.velocity(f0).cpu()[-1,:,:]
    theta0=solver.dis.temperature(f0).cpu()[-1,:,0:1]

    err1=(((rho-rho0)**2).mean()/((rho0)**2).mean())**0.5
    err2=(((u-u0)**2).mean()/(0.01+(u0)**2).mean())**0.5
    err3=(((theta-theta0)**2).mean()/((theta0)**2).mean())**0.5
    logger.info(f"err at t=0.0: {err1.item():.3e}\t{err2.item():.3e}\t{err3.item():.3e}\t")


    VA_x=torch.linspace(-0.5,0.5,400)[...,None].to(device=device)
    VA_t=0.1*torch.ones_like(VA_x).to(device=device)
    y=net_LR(VA_x,VA_t)
    rho,u,theta=rho_u_theta_LR(y,VT,WT)
    rho=rho.cpu()
    u=u.cpu()
    theta=theta.cpu()

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(VA_x.cpu()[...,0],rho.detach().cpu(),label=r'NN-$\rho$')
    ax.plot(VA_x.cpu()[...,0],u.detach().cpu()[...,0:1],label=r'NN-$u_x$')
    ax.plot(VA_x.cpu()[...,0],theta.detach().cpu(),label=r'NN-$T$')

    ax.plot(solver.dis.x.cpu(),solver.dis.density().cpu()[0,:,0],"--",label=r'FSM-$\rho$')
    ax.plot(solver.dis.x.cpu(),solver.dis.velocity().cpu()[0,:,0:1],"--",label=r'FSM-$u_x$')
    ax.plot(solver.dis.x.cpu(),solver.dis.temperature().cpu()[0,:,0],"--",label=r'FSM-$T$')
    plt.legend(loc='upper right')

    plt.savefig("t1.png")

    np.savez("t1.npz",rho=rho.detach().cpu(),u=u.detach().cpu()[...,0:1],theta=theta.detach().cpu(),
                rho0=solver.dis.density().cpu()[0,:,0],
                u0=solver.dis.velocity().cpu()[0,:,0:1],
                theta0=solver.dis.temperature().cpu()[0,:,0])

    rho0=solver.dis.density().cpu()[-1,:,0:1]
    u0=solver.dis.velocity().cpu()[-1,:,:]
    theta0=solver.dis.temperature().cpu()[-1,:,0:1]

    err1=(((rho-rho0)**2).mean()/((rho0)**2).mean())**0.5
    err2=(((u-u0)**2).mean()/(0.01+(u0)**2).mean())**0.5
    err3=(((theta-theta0)**2).mean()/((theta0)**2).mean())**0.5
    logger.info(f"err at t=0.1: {err1.item():.3e}\t{err2.item():.3e}\t{err3.item():.3e}\t")

if __name__ == "__main__":
    train()