from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath('../..'))

import torch

from bte.grad import GradEquation
from bte.grad import distribution
from bte.grad.distribution import HermiteDisND as DISND
from matplotlib import pyplot as plt

torch.set_default_dtype(torch.float32)

rho_l, u_l, P_l = 1.0, 0., 1
rho_r, u_r, P_r = 0.125, 0., 0.1
T_l=P_l/rho_l
T_r=P_r/rho_r

DIM=3
nx=1600
xmax=0.5
xmin=-0.5
M=5

rho_l=rho_l*torch.ones(nx)
u_l=u_l*torch.ones(nx)
T_l=T_l*torch.ones(nx)
rho_r=rho_r*torch.ones(nx)
u_r=u_r*torch.ones(nx)
T_r=T_r*torch.ones(nx)

solver = GradEquation.Equation_HermiteBased(M=M, bdc='cauchy')
solver.set_order(2,2)
solver.set_collision("Maxwellian")
dx = float(xmax - xmin) / nx
x = torch.arange(start=xmin + dx / 2, end=xmax + dx / 2, step=dx)

rho=(rho_l)*(x<0)+(rho_r)*(x>=0)
u=0*x
T=(T_l)*(x<0)+(T_r)*(x>=0)

indt=distribution.index_table_expand(M,dim=DIM)
lenM=len(indt.iNto1)
f=torch.zeros((1,nx,1+DIM+lenM)).cuda()
solver.cuda()
solver.no_closure=False
f[...,0]=u
f[...,1:DIM]=0
f[...,DIM]=T
f[...,DIM+1]=rho

print(f.shape)
dis0=DISND(f[...,0:DIM],f[...,DIM:DIM+1],f[...,DIM+1:],indt)
kn=1e-4*torch.ones((1,nx,1)).cuda()

dx=(xmax-xmin)/nx
solver.set_initial(kn,dis0,dx)
solver.hme=False
solver.max_dt=1e-5
dis1=solver.run(dis0, 0.1, dx, Kn=kn, verbose=True)

plt.plot(dis1.density().cpu()[0,:])

plt.savefig("shocktube1+3_maxwellian.png")