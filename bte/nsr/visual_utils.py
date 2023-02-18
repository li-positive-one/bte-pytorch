import re
import numpy as np
import torch

def read_data(filename, verbose=True):
    with open(filename) as f:
        h1=f.readline()
        h2=f.readline()
        var=h1.strip().replace(" ","").lstrip("VARIABLES=").split(",")
        i=re.findall("I\s*=\s*(\d*)",h2)[0]
        j=re.findall("J\s*=\s*(\d*)",h2)[0]
        i,j=int(i),int(j)
        sij=re.findall("VARLOCATION=\(\[(\d+)-(\d+)\]=CELLCENTERED\)",h2)[0]
        data=[]
        for l in f.readlines():
            data=data+[float(n) for n in l.split()]
        output={}
        bi=0
        output["I"]=i
        output["J"]=j
        for c,v in enumerate(var):
            if c+1<int(sij[0]):
                output[v]=np.array(data[bi:bi+i*j]).reshape((i,j))
                bi=bi+i*j
            else:
                output[v]=np.array(data[bi:bi+(i-1)*(j-1)]).reshape((i-1,j-1))
                bi=bi+(i-1)*(j-1)
    return output

from scipy.interpolate import RegularGridInterpolator
def fvmlinspace(vmin,vmax,nv):
    dv=(vmax-vmin)/nv
    return np.linspace(vmin+dv/2,vmax-dv/2,nv)

def remesh(data,newN):
    originN=data.shape[0]
    x = fvmlinspace(0,1,originN)
    y = fvmlinspace(0,1,originN)
    xg, yg = np.meshgrid(x, y, indexing='ij')
    interp = RegularGridInterpolator((x, y), data,bounds_error=False,fill_value=None)
    Nx,Ny=fvmlinspace(0,1,newN),fvmlinspace(0,1,newN)
    Nxg, Nyg = np.meshgrid(Nx,Ny, indexing='ij')
    Nxy=np.stack((Nxg, Nyg),axis=-1).reshape((newN*newN,2))
    return interp(Nxy).reshape((newN,newN)+data.shape[2:])
def read_tecplot(filename):
    with open(filename) as f:
        head1=f.readline()
        p=[s.strip().strip('"') for s in head1.split("=")[1].split(",")]
        head2=f.readline()
        data=[]
        for l in f.readlines():
            data.append([float(s) for s in l.split()])
        data=np.array(data)
    return p,data
def fvmlinspace_t(vmin,vmax,nv):
    dv=(vmax-vmin)/nv
    return torch.linspace(vmin+dv/2,vmax-dv/2,nv)

def fvmlinspace(vmin,vmax,nv):
    dv=(vmax-vmin)/nv
    return np.linspace(vmin+dv/2,vmax-dv/2,nv)
