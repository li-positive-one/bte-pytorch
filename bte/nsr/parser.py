import torch
import random
import numpy as np
from matplotlib import pyplot as plt
from bte.dvm import solver as bgk_solver
from bte.utils.gas import get_potential,get_mu,get_kn_bzm

def config_to_solver(config):
    nv=config["vmesh"]["nv"]
    vmin=config["vmesh"]["vmin"]
    vmax=config["vmesh"]["vmax"]
    Kn=config["Kn"]

    kn = Kn
    omega = 0.81
    alpha = get_potential(omega)
    mu_ref = get_mu(alpha,omega,kn)
    kn_bzm = get_kn_bzm(alpha,mu_ref)

    nx=400

    if config["collision"]=="BGK":
        solver = bgk_solver.BGKSolver(-0.5,0.5,nx,vmin,vmax,nv,device='cpu',BC_type='periodic',bgk_simple_kn="simple",dtype=torch.get_default_dtype())
    elif config["collision"]=="FBGK":
        solver = bgk_solver.BGKSolver(-0.5,0.5,nx,vmin,vmax,nv,device='cpu',BC_type='periodic',bgk_simple_kn="mu_ref",dtype=torch.get_default_dtype())
    elif config["collision"]=="FSM":
        solver = bgk_solver.BGKSolver(-0.5,0.5,nx,vmin,vmax,nv,device='cpu',BC_type='periodic',bgk_simple_kn="mu_ref",dtype=torch.get_default_dtype())
    else:
        raise ValueError

    return solver