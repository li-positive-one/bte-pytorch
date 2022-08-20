import torch
from math import factorial
from .operators import make_tensor
from bte.grad.distribution import HermiteDisND
from .specials import eval_hermitenorm_seq

class DisTranfer():
    def __init__(self,v_meta,indt):
        assert len(v_meta.v_dims)==indt.DIM
        self.v_meta=v_meta
        self.indt=indt
        self.hermite={}
        self.ORDER=indt.ORDER
        self.DIM=indt.DIM
        self.v_shape=tuple([len(v.v) for v in v_meta.v_dims])
        for d,vm in enumerate(self.v_meta.v_dims):
            self.hermite[d]=eval_hermitenorm_seq(self.ORDER,vm.v.cpu()).permute(1,2,0).squeeze()
        self.M=torch.zeros((len(self.indt.i1toN),)+self.v_shape)
        for i,orders in self.indt.i1toN.items():
            #在这里构造相应的Hermite多项式的值
            self.M[i]=self.getHermite(orders)
        self.M=self.M.to(device=v_meta.v_dims[0].v.device)
        
    def getHermite(self,order):
        ans=torch.zeros(self.v_shape)
        fs=[self.hermite[d][...,di]/factorial(di)*self.v_meta.v_dims[d].w.cpu().flatten() for d,di in enumerate(order)]
        ans=make_tensor(fs)
        return ans
            
    def DVMtoHermite(self,DVDis):
        # 暂时只支持DIM=1,2,3
        f=DVDis.f
        f=f.reshape(f.shape[:-1]+(self.v_shape))
        if self.DIM==3:
            coef=torch.einsum("...ijk,cijk->...c",f,self.M)
        elif self.DIM==2:
            coef=torch.einsum("...ij,cij->...c",f,self.M)
        elif self.DIM==1:
            coef=torch.einsum("...i,ci->...c",f,self.M)
        else:
            raise ValueError
        gdis=HermiteDisND(torch.zeros(f.shape[:-self.DIM]+(self.DIM,),device=f.device),torch.ones(f.shape[:-self.DIM]+(1,),device=f.device),coef,self.indt)
        return gdis

    def HermitetoDVM(self,gdis):
        pass