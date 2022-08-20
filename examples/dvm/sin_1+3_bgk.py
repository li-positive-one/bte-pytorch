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

kn = 0.5
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

def maxwellian(v, rho, u, T):
    return (rho / torch.sqrt(2 * math.pi * T)**v.shape[-1]) * torch.exp(-((u[..., None, :] - v) ** 2).sum(dim=-1) / (2 * T))
        

solver = bgk_solver.BGKSolver(-1,1,nx,(-10,-10,-10),(10,10,10),(80,40,40),device='cuda',bgk_simple_kn="simple2",BC_type="periodic")
solver.mu_ref=mu_ref
solver.omega=omega
solver.set_space_order(2)
solver.set_time_stepper("IMEX")
solver.set_collisioner("BGK")
    
x, v = solver.dis.x.cpu(), solver.dis.v.cpu()
print(x.shape,v.shape)

P_l=  1
rho_l=2+0.5*torch.cos(math.pi*x).reshape(nx,1)
u_l = torch.zeros(nx,3)
u_l[...,0] = 1+0.5*torch.sin(math.pi*x)
u_l[...,1] = 0.5*torch.sin(math.pi*x)
T_l=P_l/rho_l
f0_l = lambda v: maxwellian(v, rho_l, u_l, T_l)


f0 = f0_l(v)
f0 = f0.unsqueeze(0).repeat(1, 1, 1).cuda()
solver.cuda()
print("f0.shape:", f0.shape)
solver.set_initial(kn, f0)
solver.dis.v_meta=solver.dis.v_meta.cuda()
solver.dis=solver.dis.cuda()

# solve
t_final, dt = 0.1, 0.001
CFL = 0.45
max_dt = CFL * 1.0 / nx / vmax
print("max_dt is", max_dt)
soln = []
for t in range(400):
    print(f"step {t}")
    print(solver.dis.density().sum())
    solver.solve_to(dt, max_dt)
print(solver.dis.density().shape)
print(solver.dis.density().max())

plt.plot(solver.dis.x.cpu().flatten(), solver.dis.density().cpu()[0,:,0],label='rho')
#plt.plot(solver.dis.velocity().cpu()[0,:,0],label='ux')
#plt.plot(solver.dis.velocity().cpu()[0,:,1],label='uy')
#plt.plot(solver.dis.temperature().cpu()[0,:,0],label='T')
plt.legend()
plt.savefig(f"sin_1+3_bgk_kn{kn}.png")