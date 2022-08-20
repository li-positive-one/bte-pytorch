from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))
import torch
import math
from bte.dvm import solver as bgk_solver
from matplotlib import pyplot as plt
import torch
torch.set_default_dtype(torch.float64)
import numpy as np
from bte.utils.gas import *

pr = 2.0/3.0
kn = 0.01
omega = 0.81
alpha = get_potential(omega)
mu_ref = get_mu(alpha,omega,kn)
kn_bzm = get_kn_bzm(alpha,mu_ref)

Ma = 1.3

###
DOM = 3
GAMMA = (DOM+2)/DOM
rho_L = 1.0
u_L = np.sqrt(GAMMA/2)*Ma
T_L = 1.0/2

rho_R = (GAMMA + 1) * Ma * Ma / ((GAMMA - 1) * Ma * Ma + 2)
u_R = np.sqrt(GAMMA/2) * Ma / rho_R
T_R =  ((2 * GAMMA * Ma * Ma - (GAMMA - 1)) / (GAMMA + 1))/rho_R/2

sos=np.sqrt(GAMMA*T_L)
print(f"Ma:{u_L/sos}")
###


xmin = -50
xmax = 50
nx = 100

rho_l=rho_L*torch.ones(nx,1)
u_l = torch.zeros(nx,DOM)
u_l[...,0] = u_L*torch.ones(nx)
T_l=T_L*torch.ones(nx,1)

rho_r=rho_R*torch.ones(nx,1)
u_r = torch.zeros(nx,DOM)
u_r[...,0]=u_R*torch.ones(nx)
T_r=T_R*torch.ones(nx,1)

def maxwellian(v, rho, u, T):
    return (rho / torch.sqrt(2 * math.pi * T)**v.shape[-1]) * torch.exp(-((u[..., None, :] - v) ** 2).sum(dim=-1) / (2 * T))

solver = bgk_solver.BGKSolver(xmin, xmax, nx, (-10,-10,-10), (10,10,10), (64,32,32),v_discrete="uni",device='cuda')
solver.mu_ref=mu_ref
solver.omega=omega

solver.set_space_order(2)
solver.set_time_stepper("bgk-RK1")
solver.set_collisioner("BGK")

x, v = solver.dis.x.cpu(), solver.dis.v.cpu()

f0_l = lambda v: maxwellian(v, rho_l, u_l, T_l)
f0_r = lambda v: maxwellian(v, rho_r, u_r, T_r)

print(x.shape,v.shape)
f0 = f0_l(v) * (x < 0)[:, None] + f0_r(v) * (x >= 0)[:, None]
f0 = f0.unsqueeze(0).repeat(1, 1, 1).cuda()

solver.cuda()
print("f0.shape:", f0.shape)
solver.set_initial(kn_bzm, f0)
solver.dis.v_meta=solver.dis.v_meta.cuda()
solver.dis=solver.dis.cuda()

t_final, dt = 0.1, 1.0
CFL = 0.45
vmax = 10
max_dt = CFL * (xmax - xmin) / nx / vmax
print("max_dt is", max_dt)
soln = []

for t in range(100):
    print(t)
    solver.solve_to(dt, max_dt)

print(solver.dis.density().shape)
print(solver.dis.density().max())

plt.plot(solver.dis.density().cpu()[0,:,0],label='rho')
plt.plot(solver.dis.velocity().cpu()[0,:,0],label='ux')
plt.plot(solver.dis.temperature().cpu()[0,:,0],label='T')
plt.legend()
plt.savefig(f"ss_bgk_Ma{Ma}.png")

np.savez(f"ss_bgk_Ma{Ma}.npz",
            rho=solver.dis.density().cpu()[0,:,0].numpy(),
            u=solver.dis.velocity().cpu()[0,:,0].numpy(),
            theta=solver.dis.temperature().cpu()[0,:,0].numpy())
