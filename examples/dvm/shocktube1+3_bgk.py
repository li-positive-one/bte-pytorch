from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))

import torch
import math
from bte.dvm import solver as bgk_solver
from bte.utils.gas import get_potential,get_mu
from matplotlib import pyplot as plt
import torch
torch.set_default_dtype(torch.float64)

device="cuda"

kn = 1e-4
omega = 0.81
alpha = get_potential(omega)
mu_ref = get_mu(alpha,omega,kn)
rho_l, u_l, P_l = 1.0, 0., 1
rho_r, u_r, P_r = 0.125, 0., 0.1
T_l=P_l/rho_l
T_r=P_r/rho_r
print(f"L:rho:{rho_l},u:{u_l},T:{T_l}")
print(f"R:rho:{rho_r},u:{u_r},T:{T_r}")
nx=200
vmax=10

rho_l=rho_l*torch.ones(nx,1)
u_l=u_l*torch.ones(nx,1)
T_l=T_l*torch.ones(nx,1)
rho_r=rho_r*torch.ones(nx,1)
u_r=u_r*torch.ones(nx,1)
T_r=T_r*torch.ones(nx,1)

def maxwellian(v, rho, u, T):
    return (rho / torch.sqrt(2 * math.pi * T)**v.shape[-1]) * torch.exp(-((u[..., None, :] - v) ** 2).sum(dim=-1) / (2 * T))
        
f0_l = lambda v: maxwellian(v, rho_l, u_l, T_l)
f0_r = lambda v: maxwellian(v, rho_r, u_r, T_r)

solver = bgk_solver.BGKSolver(-0.5,0.5,nx,(-10,-10,-10),(10,10,10),(100,40,40),device='cuda',bgk_simple_kn="simple2")
solver.mu_ref=mu_ref
solver.omega=omega
solver.set_space_order(2)
solver.set_time_stepper("IMEX")
solver.set_collisioner("BGK")
    
x, v = solver.dis.x.cpu(), solver.dis.v.cpu()
print(x.shape,v.shape)
f0 = f0_l(v) * (x < 0)[:, None] + f0_r(v) * (x >= 0)[:, None]
f0 = f0.unsqueeze(0).repeat(1, 1, 1).to(device=device)
solver.to(device=device)
print("f0.shape:", f0.shape)
solver.set_initial(kn, f0)
solver.dis.v_meta=solver.dis.v_meta.to(device=device)
solver.dis=solver.dis.to(device=device)

# solve
t_final, dt = 0.1, 0.001
CFL = 0.45
max_dt = CFL * 1.0 / nx / vmax
print("max_dt is", max_dt)
soln = []
for t in range(100):
    print(t)
    solver.solve_to(dt, max_dt)
print(solver.dis.density().shape)
print(solver.dis.density().max())

plt.plot(solver.dis.density().cpu()[0,:,0],label='rho')
plt.plot(solver.dis.velocity().cpu()[0,:,0],label='ux')
plt.plot(solver.dis.velocity().cpu()[0,:,1],label='uy')
plt.plot(solver.dis.temperature().cpu()[0,:,0],label='T')
plt.legend()
plt.savefig(f"shocktube1+3_bgk_kn{kn}.png")