import torch
import numpy as np

DATAPATH="/home/zyli/NRXXPKU/examples/test33/A/A_10_10.bin"
MAX_DEG=10

def load_collision_kernel(kernel_path=DATAPATH,  max_order=MAX_DEG, order=None,dtype=torch.float32):
    assert (order is None) or isinstance(order,int)
    if isinstance(order,int):
        assert order<=max_order

    n_mom=(max_order + 1) * (max_order + 2) * (max_order + 3) // 6
    nn=(order + 1) * (order + 2) * (order + 3) // 6
    with open(kernel_path) as f:
        bdata = np.fromfile(f, dtype=np.float64)
    s= bdata [5 * n_mom * n_mom + 5 * n_mom];
    s = s+ bdata[5 * n_mom * n_mom + 5];
    s = -1/s
    bdata=bdata*s
    bdata=bdata.reshape([n_mom,n_mom,n_mom])

    if order is None:
        re=torch.from_numpy(bdata)
    else:
        re=torch.from_numpy(bdata[:nn,:nn,:nn])
    return re.to(dtype=dtype)

def collision_f(Kernel,f):
    fs=f.shape
    f=f.reshape([-1,f.shape[-1]])
    q=torch.einsum("kij,Ni,Nj->Nk",Kernel,f,f)
    q=q.reshape(fs)
    return q