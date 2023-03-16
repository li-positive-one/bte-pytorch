import sys
import os
sys.path.insert(0, os.path.abspath('../..'))

import logging
logger = logging.getLogger(__name__)
from bte.nsr.model import SplitNet
from bte.nsr.model import maxwellian, rho_u_theta, _m012

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
from bte.nsr.reduced_collision import collision_fft_fg,get_new_kernel,get_reduced_kernel,ReductionCollision
from bte.nsr.dataset import SmoothD2Case,SmoothD1Case,SodD1Case
from bte.nsr.utils import orthonormalize
from bte.nsr.model import LossCompose,Problem
import pytorch_warmup as warmup

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
        (IV_x,IV_t,IV_f),(BV_x1,BV_t1,BV_x2,BV_t2),(IN_x,IN_t)=datagen.get()
        BDC="periodic"
    else:
        datagen=SodD1Case(config)
        (IV_x,IV_t,IV_f),(BV_x,BV_t,BV_f),(IN_x,IN_t)=datagen.get()
        BDC="constant"

    # if  config["collision"]=="RC":
    #     t_final, dt = 0.1, 0.01
    #     CFL = 0.45
    #     max_dt = CFL * 1.0 / nx / vmax[0]
    #     print("max_dt is", max_dt)
    #     soln = []
    #     nt=10
    #     f0=datagen.init_value()
    #     solver_bgk = bgk_solver.BGKSolver(xmin,xmax,nx,vmin,vmax,nv,device='cpu',BC_type=BDC,bgk_simple_kn="mu_ref"
    #     )
    #     solver_bgk.mu_ref=mu_ref
    #     solver_bgk.omega=omega
    #     solver_bgk.set_time_stepper("bgk-RK2")
    #     solver_bgk.set_collisioner("BGK")
    #     solver_bgk.set_initial(kn, f0)
    #     solver_bgk.dis.v_meta=solver_bgk.dis.v_meta.to(device=device)
    #     solver_bgk.dis=solver_bgk.dis.to(device=device)
    #     solver_bgk.cuda()
    #     paths_bgk=torch.zeros((nt,nx,math.prod(nv)))
    #     for t in range(nt):
    #         print(t)
    #         solver_bgk.solve_to(dt, max_dt)
    #         paths_bgk[t]=solver_bgk.dis.f.cpu()

    # if config.collision=="RC":
    #     perm = torch.randperm(400*10)
    #     k=2000
    #     t2=-1
    #     idx = perm[:k]
    #     pathsf2=paths_bgk.flatten(0,1)
    #     traindata=pathsf2[idx]
    #     RC, BASE1, BASE2=get_reduced_kernel(config, traindata, 40)

    if config.collision=="RC":
        #traindata=np.load(f"{os.environ['HOME']}/.data/D2V3_Kn{Kn}.npz")["trainpath"]
        traindata=np.load(f"{os.environ['HOME']}/.data/D2V3_FSM_Kn{Kn}.npz")["trainpath"]
        traindata=torch.from_numpy(traindata)
        RC, BASE1, BASE2=get_reduced_kernel(config, traindata, 40)

    if config["collision"]=="BGK":
        solver = bgk_solver.BGKSolver(xmin,xmax,nx,vmin,vmax,nv,device='cpu',BC_type=BDC,bgk_simple_kn="simple"
        )
    elif config["collision"]=="FBGK":
        solver = bgk_solver.BGKSolver(xmin,xmax,nx,vmin,vmax,nv,device='cpu',BC_type=BDC,bgk_simple_kn="mu_ref"
        )
    elif config["collision"] in["FSM","RC"]:
        solver = bgk_solver.BGKSolver(xmin,xmax,nx,vmin,vmax,nv,device='cpu',BC_type=BDC,bgk_simple_kn="mu_ref"
        )
    else:
        raise ValueError

    solver.mu_ref=mu_ref
    solver.omega=omega
    print("x.dtype:",  solver.dis.x.dtype)
    solver.set_space_order(2)
    
    if config["collision"] in ["BGK","FBGK"]:
        solver.set_time_stepper("bgk-RK2")
        solver.set_collisioner("BGK")
    elif config["collision"] in ["FSM","RC"]:
        solver.set_time_stepper("IMEX")
        solver.set_collisioner("FFT")
    else:
        raise ValueError

    x, v = solver.dis.x.cpu(), solver.dis.v.cpu()

    f0=datagen.init_value()
    solver.to(device=device)

    if config.collision in["FSM","RC"]:
        kn=kn_bzm
    else:
        pass
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


    model=SplitNet(config.network.neurons , VDIS, config.network.multires, xdim=2)

    model=model.to(device=device)
    vdis=v

    def net(x,y,t):
        inputs=torch.cat([x,y,t],dim=-1)
        return model(inputs)

    def nn_helper(x,y,t):
        result=net(x,y,t).sum(dim=0)
        return result

    from functorch import grad,jacfwd,vmap,jacrev

    ggrad=vmap(jacfwd(net,(0,1,2)))

    def pinn(x, y, t, net,Kn=Kn):
        f = net(x,y,t)
        fx, fy, ft=ggrad(x,y,t)
        f_x=(fx[...,0])#.transpose(0,1)
        f_y=(fy[...,0])
        f_t=(ft[...,0])#.transpose(0,1)

        if config.collision=="BGK":
            rho,u,theta = rho_u_theta(f,VDIS,WDIS)
            f_M = maxwellian(VDIS,rho,u,theta)
            pde = f_t + vdis[...,0] * f_x + vdis[...,1] * f_y -1/kn*(f_M-f)
        elif config.collision=="FBGK" :
            rho,u,theta = rho_u_theta(f,VDIS,WDIS)
            f_M = maxwellian(VDIS,rho,u,theta)
            kn_bgk=(mu_ref*2/(2*theta)**(1-omega)/rho)
            kn_bgk=torch.maximum(kn_bgk,0.001*torch.ones_like(kn_bgk))
            pde = f_t + vdis[...,0] * f_x + vdis[...,1] * f_y -1/kn_bgk*(f_M-f)
        elif  config.collision=="FSM" :
            pde = f_t + vdis[...,0] * f_x + vdis[...,1] * f_y -1/kn_bzm*solver.coller.do_collision(f)
        elif config.collision=="RC":
            pde = (f_t + vdis[...,0] * f_x + vdis[...,1] * f_y)  -1/kn_bzm * RC.do_collision(f)
        else:
            raise ValueError
        return pde

    # 构造光滑初值问题
    wc=(math.prod(nv)+7*1)*3
    LC3=LossCompose(wc).to(device=device)
    smooth_l1_loss =torch.nn.MSELoss(reduction='none')

    criterion=lambda x,y:smooth_l1_loss(x,y).mean(dim=0)
    criterion_norm=lambda x:smooth_l1_loss(x,torch.zeros_like(x)).mean(dim=0)
    def prim_norm(f,order=2):
        m1,m2,m3=_m012(f,vdis,w)
        return torch.cat([criterion_norm(m1),criterion_norm(m2),criterion_norm(m3)],dim=-1)

    start_time=time.time()
    lr=config.optim.lr_scheduler.max_lr
    optimizer=torch.optim.Adam( 
                    [{'params': model.parameters(), 'lr':lr},
                    {'params': LC3.parameters(), 'lr':lr}])
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,5e-3, total_steps=config.optim["Adam_steps"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,config.optim["Adam_steps"])
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=config.optim["Adam_steps"]//10)
   
#############
    from bte.nsr.visual_utils import read_data, fvmlinspace, fvmlinspace_t
    if config["collision"]=="BGK":
        data80=read_data(f"/home/zyli/NeuralBoltzmann/fortran/cylinder-sbgk-80-{config['Kn']}.plt")
        rhoGT,uGT,thetaGT=data80["RHO"],data80["U"],data80["T"]
    else:
        data80=read_data(f"/home/zyli/NeuralBoltzmann/fortran/cylinder-fsm-80-{config['Kn']}.plt")
        rhoGT,uGT,thetaGT=data80["RHO"],data80["U"],data80["T"]
    nx=80
    nxt=80
    IV_x0=fvmlinspace_t(-0.5,0.5,nxt).to(device=device)
    IV_y0=fvmlinspace_t(-0.5,0.5,nxt).to(device=device)
    IV_x0, IV_y0 = torch.meshgrid(IV_x0, IV_y0, indexing='xy')
    IV_x0=IV_x0[...,None]
    IV_y0=IV_y0[...,None]
    IV_t0=torch.zeros_like(IV_x0).to(device=device)
    rho_l0=0.5*torch.sin(2*np.pi*IV_x0)*torch.sin(2*np.pi*IV_y0)+1
    T_l0=1
    Ni0=IV_x0.shape[0]
    rho_l0=rho_l0*torch.ones(Ni0,1).to(device=device)
    u_l0=0*torch.ones(Ni0,3).to(device=device)
    T_l0=T_l0*torch.ones(Ni0,1).to(device=device)
    #print(VDIS.shape,rho_l0.shape,u_l0.shape,T_l0.shape)
    f_l0=maxwellian(VDIS, rho_l0, u_l0, T_l0)
    IV_f0 = f_l0
    
    rho,u,theta=rho_u_theta(IV_f0,vdis,w)
    IV_pred=net(IV_x0,IV_y0,IV_t0+0.0)
    rho1,u1,theta1=rho_u_theta(IV_pred,VDIS,WDIS)
#######
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,5e-3, total_steps=config.optim["Adam_steps"])
    problem=Problem(config, model, LC3)
    train_flag=True
    datasampler=SmoothD2Case(config)
    for i in range(config.optim["Adam_steps"]):
        (IV_x,IV_y,IV_t,IV_f),(BV_xa,BV_ya,BV_ta,BV_xb,BV_yb,BV_tb),(IN_x,IN_y,IN_t)=datasampler.get()
        IV_pred=net(IV_x,IV_y,IV_t)

        loss_IV=criterion(IV_pred,IV_f)*config.network.weight[0]
        loss_IV2=prim_norm(IV_pred-IV_f)*config.network.weight[0]

        if config.case=="Wave":
            BV_pred1=net(BV_xa,BV_ya,BV_ta)
            BV_pred2=net(BV_xb,BV_yb,BV_tb)
            loss_BV=criterion(BV_pred1,BV_pred2)*config.network.weight[1]
            loss_BV2=prim_norm(BV_pred1-BV_pred2)*config.network.weight[1]
        elif config.case=="Sod":
            BV_pred=net(BV_x,BV_t)
            loss_BV=criterion(BV_pred,BV_f)*config.network.weight[1]
            loss_BV2=prim_norm(BV_pred-BV_f)*config.network.weight[1]
        else:
            raise ValueError

        pde=pinn(IN_x,IN_y,IN_t,net)
        loss_pinn=criterion_norm(pde)*config.network.weight[2]#*min(1,(0.1+2*(i)/config.optim["Adam_steps"]))
        loss_pinn2=prim_norm(pde)*config.network.weight[2]#*min(1,(0.1+2*(i)/config.optim["Adam_steps"]))
        
        loss_sum=LC3(torch.cat([loss_IV,loss_IV2,loss_BV,loss_BV2,loss_pinn,loss_pinn2]))

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()
        #with warmup_scheduler.dampening():
        scheduler.step()   
        
        if i%10==0:
            e_sum=loss_sum.item()
            e_IV=loss_IV.mean().item()
            e_IV2=loss_IV2.mean().item()
            e_BV=loss_BV.mean().item()
            e_BV2=loss_BV2.mean().item()
            e_pinn=loss_pinn.mean().item()
            e_pinn2=loss_pinn2.mean().item()
            logger.info(f"{i},{time.time()-start_time:.2e}:{e_sum:.2e},{e_IV:.2e},{e_IV2:.2e},{e_BV:.2e},{e_BV2:.2e},{e_pinn:.2e},{e_pinn2:.2e}")
        if i in [10,100,200,300,400,500,750,1000,2000,3000,4000,5000,6000,7000,8000,9000]:
            torch.save(model.state_dict(), f"model_epoch{i}.ckpt")
            torch.save(problem.state_dict(), f"problem_epoch{i}.ckpt")
        
        if i%100==0:
            IV_pred=net(IV_x0,IV_y0,IV_t0+0.0)
            rho1,u1,theta1=rho_u_theta(IV_pred,VDIS,WDIS)
            err1=(((rho1-rho)**2).mean()/((rho)**2).mean())**0.5
            err2=(((u1-u)**2).mean()/(1+(u)**2).mean())**0.5
            err3=(((theta1-theta)**2).mean()/((theta)**2).mean())**0.5
            logger.info(f"err at t=0.0: {err1.item():.2e}\t{err2.item():.2e}\t{err3.item():.2e}\t")
            
            IV_pred=net(IV_x0,IV_y0,IV_t0+0.1)
            rho1,u1,theta1=rho_u_theta(IV_pred,VDIS,WDIS)
            err1=(((rho1.cpu().detach().numpy()[...,0]-rhoGT)**2).mean()/((rhoGT)**2).mean())**0.5
            err2=(((u1.cpu().detach().numpy()[...,0]-uGT)**2).mean()/(1+(uGT)**2).mean())**0.5
            err3=(((theta1.cpu().detach().numpy()[...,0]-thetaGT)**2).mean()/((thetaGT)**2).mean())**0.5
            logger.info(f"err at t=0.1: {err1.item():.2e}\t{err2.item():.2e}\t{err3.item():.2e}\t")

    model.eval()

    torch.save(model.state_dict(), "model.ckpt")
    torch.save(problem.state_dict(), "problem.ckpt")

if __name__ == "__main__":
    train()