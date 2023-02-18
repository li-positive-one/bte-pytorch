import datapane as dp
import yaml
import torch
import numpy as np
from matplotlib import pyplot as plt
from bte.nsr.refsolution import bgk_solver
from bte.nsr.model import rho_u_theta
from bte.nsr.mesh import get_vmsh
from omegaconf import DictConfig, OmegaConf
import os

from pandas import Series,DataFrame
import pandas as pd

def save_dict_csv(filename,data):
    df = DataFrame(data)
    df.index.name='index'
    df.to_csv(filename)
    return
    
def save_tensor_csv(filename,data,columns):
    df = DataFrame(data,columns=columns)
    df.index.name='index'
    df.to_csv(filename)
    
def plot_solution(config, dataset, Problem, t=0.0):
    device='cuda'
    nx=100
    vmin=config["vmesh"]["vmin"]
    vmax=config["vmesh"]["vmax"]
    nv=config["vmesh"]["nv"]
    solver=bgk_solver(nx,vmin,vmax,nv).cuda()

    x, v = solver.solver.dis.x, solver.solver.dis.v

    f0=dataset[0][2].cpu().numpy()
    f1=solver(f0,config["Kn"],t)

    VA_x=torch.linspace(-0.5,0.5,config["xtmesh"]["nx"])[...,None].to(device=device)
    VA_t=t*torch.ones_like(VA_x).to(device=device)

    y=Problem(VA_x,VA_t)
    v,w,vL,wL=get_vmsh(config)
    VDIS=torch.from_numpy(v).to(device=device)
    WDIS=torch.tensor(w).to(device=device)
    rho,u,theta=rho_u_theta(y,VDIS,WDIS)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(VA_x.cpu()[...,0],rho.detach().cpu())
    ax.plot(VA_x.cpu()[...,0],u.detach().cpu())
    ax.plot(VA_x.cpu()[...,0],theta.detach().cpu())

    ax.plot(VA_x.cpu()[...,0],solver.solver.dis.density(torch.from_numpy(f1).cuda()).cpu()[0,:,0],"--")
    ax.plot(VA_x.cpu()[...,0],solver.solver.dis.velocity(torch.from_numpy(f1).cuda()).cpu()[0,:,0],"--")
    ax.plot(VA_x.cpu()[...,0],solver.solver.dis.temperature(torch.from_numpy(f1).cuda()).cpu()[0,:,0],"--")

    return fig

def plot_dataset(dataset):
    (IV_x,IV_t,IV_f),(BV_x1,BV_t1,BV_x2,BV_t2),(IN_x,IN_t)=dataset
    fig,ax=plt.subplots(figsize=(14,7))
    ax.scatter(IN_x.cpu(),IN_t.cpu())
    ax.scatter(IV_x.cpu(),IV_t.cpu())
    ax.scatter(BV_x1.cpu(),BV_t1.cpu())
    ax.scatter(BV_x2.cpu(),BV_t2.cpu())
    return fig

def genreport(config, results):
    datafig=plot_dataset(results["dataset"])
    solu0fig=plot_solution(config,results["dataset"],results["Problem"],0.0)
    solu1fig=plot_solution(config,results["dataset"],results["Problem"],config["xtmesh"]["tmax"])
    report = dp.Report(
        dp.Text("# Report \n ## config \n```\n"+OmegaConf.to_yaml(config)+"```\n"),
        dp.Text("## Data points\n"),
        dp.Plot(datafig),
        dp.Text(f"## Solution at {0.0}s\n"),
        dp.Plot(solu0fig),
        dp.Text(f"## Solution at {config['xtmesh']['tmax']}s\n"),
        dp.Plot(solu1fig)
    )      

    report.save(path=os.getcwd()+"/report.html")