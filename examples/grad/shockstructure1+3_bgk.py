from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath('../..'))

import torch
import numpy as np
from bte.grad import GradEquation
from bte.grad import distribution
from bte.grad.distribution import HermiteDisND as DISND
from matplotlib import pyplot as plt


torch.set_default_dtype(torch.float64)

from bte.utils.gas import *

pr = 2.0/3.0
kn = 1
omega = 0.81
alpha = get_potential(omega)
mu_ref = get_mu(alpha,omega,kn)
kn_bzm=get_kn_bzm(alpha,mu_ref)

Ma =  1.3

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

DIM=3
M=10

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


solver = GradEquation.Equation_HermiteBased(M=M, bdc='constant')
solver.set_order(2,2)
solver.mu_ref=mu_ref
solver.omega=omega

dx = float(xmax - xmin) / nx
x = torch.arange(start=xmin + dx / 2, end=xmax + dx / 2, step=dx)

x=x[...,None]

rho=(rho_l)*(x<0)+(rho_r)*(x>=0)
u=(u_l)*(x<0)+(u_r)*(x>=0)
T=(T_l)*(x<0)+(T_r)*(x>=0)
print(rho.shape,u.shape,T.shape)


indt=distribution.index_table_expand(M,dim=DIM)
lenM=len(indt.iNto1)
f=torch.zeros((1,nx,1+DIM+lenM)).cuda()
solver.cuda()
solver.no_closure=False
f[...,0:DIM]=u
f[...,DIM:DIM+1]=T
f[...,DIM+1:DIM+2]=rho

print(f.shape)
dis0=DISND(f[...,0:DIM],f[...,DIM:DIM+1],f[...,DIM+1:],indt)
kn=kn_bzm*torch.ones((1,nx,1)).cuda()

dx=(xmax-xmin)/nx
solver.set_initial(kn,dis0,dx)
solver.hme=False

dis1=solver.run(dis0, 100, dx, Kn=kn, verbose=True)

plt.plot(dis1.density().cpu()[0,:,0],label='rho')
plt.plot(dis1.velocity().cpu()[0,:,0],label='ux')
plt.plot(dis1.temperature().cpu()[0,:,0],label='T')
plt.legend()
plt.savefig(f"ss1+3_bgk_M{M}_Ma{Ma}.png")

np.savez(f"ss1+3_bgk_M{M}_Ma{Ma}.npz",
            rho=dis1.density().cpu()[0,:,0].numpy(),
            u=dis1.velocity().cpu()[0,:,0].numpy(),
            theta=dis1.temperature().cpu()[0,:,0].numpy())