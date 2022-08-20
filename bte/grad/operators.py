from typing import Iterable
import torch
import functools

def make_tensor(tensors:Iterable[torch.Tensor])->torch.Tensor:
    #将一系列一维tensor张成一个大张量
    DIM=len(tensors)
    tensors_reshape=[]
    for d in range(DIM):
        t_tmp=tensors[d].reshape([1,]*d+[-1,]+[1,]*(DIM-d-1))
        tensors_reshape.append(t_tmp)
    ans=functools.reduce(lambda a,b:a*b,tensors_reshape)    
    return ans