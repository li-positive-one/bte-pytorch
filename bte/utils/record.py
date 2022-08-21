import torch

def collect_data_G13(dis):
    RHO=dis.density().cpu()
    MU=dis.velocity().cpu()
    THETA=dis.temperature().cpu()
    SIGMA=torch.zeros(RHO.shape[:-1]+(3,3))
    for i in range(3):
        for j in range(3):
            idx=[0,0,0]
            idx[i]+=1
            idx[j]+=1
            SIGMA[...,i,j]=dis._M(0,*idx)[...,0]
    Q=torch.zeros(RHO.shape[:-1]+(3,))
    for i in range(3):
        idx=[0,0,0]
        idx[i]+=1
        Q[...,i]=dis._M(1,*idx)[...,0]
    data={
        "rho":RHO,
        "u":MU,
        "theta":THETA,
        "sigma":SIGMA,
        "Q":Q,
    }
    return data

def collect_data_G13Ex(dis,f=None):
    if f is None:
        f=dis.f
    RHO=dis.density().cpu()
    MU=dis.velocity().cpu()
    THETA=dis.temperature().cpu()
    SIGMA=torch.zeros(RHO.shape[:-1]+(3,3))
    for i in range(3):
        for j in range(3):
            idx=[0,0,0]
            idx[i]+=1
            idx[j]+=1
            SIGMA[...,i,j]=dis._M(0,*idx,f)[...,0]
    Q=torch.zeros(RHO.shape[:-1]+(3,))
    for i in range(3):
        idx=[0,0,0]
        idx[i]+=1
        Q[...,i]=dis._M(1,*idx,f)[...,0]
    U0_ijk=torch.zeros(RHO.shape[:-1]+(3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                idx=[0,0,0]
                idx[i]+=1
                idx[j]+=1
                idx[k]+=1
                U0_ijk[...,i,j,k]=dis._M(0,*idx,f)[...,0]
    U1_ij=torch.zeros(RHO.shape[:-1]+(3,3))
    for i in range(3):
        for j in range(3):
            idx=[0,0,0]
            idx[i]+=1
            idx[j]+=1
            U1_ij[...,i,j]=dis._M(1,*idx,f)[...,0]
    W2=dis._M(2,0,0,0,f).cpu()
    data={
        "rho":RHO,
        "u":MU,
        "theta":THETA,
        "sigma":SIGMA,
        "Q":Q,
        "U0_ijk":U0_ijk,
        "U1_ij":U1_ij,
        "W2":W2
    }
    return data
