import torch
import itertools
import math
import functools 
from typing import Tuple, List

def get_index_table(M:int,dim:int=3):
    """
    get_index_table return two dictionary.
    The first is (int,int,...)->int
    The second is int->(int,int,...)

    Args:
        M (int): order
        dim (int, optional): dimension. Defaults to 3.

    Returns:
        indexNto1dict, index1toNdict
    """
    indexNto1dict={}
    index1toNdict={}
    c=0
    for m in range(M+1):
        for index in itertools.product(*([range(m+1)]*dim)):
            re_index=index[::-1]    
            if sum(re_index)==m:
                indexNto1dict[re_index]=c
                index1toNdict[c]=re_index
                c=c+1
    return indexNto1dict, index1toNdict

def get_index_len(M:int,dim=3):
    """
    get_index_len 

    Args:
        M (int): order.
        dim (int, optional): dimension. Defaults to 3.

    Returns:
        int: (M+dim)*(M+dim-1)*...(M+1)/(dim!)
    """
    return functools.reduce(lambda x,y:x*y,[M+i+1 for i in range(dim)])//math.factorial(dim)

class index_table():
    def __init__(self,M:int,dim:int=3):
        indexNto1dict, index1toNdict = get_index_table(M,dim)
        self.iNto1 = indexNto1dict
        self.i1toN = index1toNdict
        self.ORDER = M
        self.DIM = dim
        self.len = len(indexNto1dict)
    def get_order(self,index1,dim=0):
        assert dim<=self.DIM
        return [self.i1toN[ind][dim] for ind in index1]
        
def get_change_index(indt:index_table,index_base:List[int],index_tuple_add):
    outputs=[]
    assert len(indt.i1toN[0])==len(index_tuple_add)
    for index in index_base:
        index_tuple=indt.i1toN[index]
        index_change=tuple([index_tuple[i]+index_tuple_add[i] for i in range(len(index_tuple_add))])
        outputs.append(indt.iNto1.get(index_change,None))
    return outputs

def get_change_index_projection(indt:index_table,index_base:List[int],index_tuple_add):
    tmp1=get_change_index(indt,index_base,index_tuple_add)    
    origin=[]
    goal=[]
    for i in index_base:
        if tmp1[i] is not None:
            origin.append(i)
            goal.append(tmp1[i])
    return torch.tensor(origin),torch.tensor(goal)

class index_table_expand(index_table):
    def __init__(self,M:int,dim:int=3):
        super().__init__(M, dim)
        length=len(self.iNto1)
        DIM=dim
        self.d1=[]
        self.u1=[]
        self.d2=[]
        v1=[]
        v2=[]
        m1=[]
        for i in range(DIM):
            d1=[0,]*DIM
            d1[i]=-1
            id1=get_change_index_projection(self,list(range(length)),tuple(d1))
            self.d1.append(id1)

            d2=[0,]*DIM
            d2[i]=-2
            id2=get_change_index_projection(self,list(range(length)),tuple(d2))
            self.d2.append(id2)

            u1=[0,]*DIM
            u1[i]=1
            iu1=get_change_index_projection(self,list(range(length)),tuple(u1))
            self.u1.append(iu1)
            v1.append(self.iNto1[tuple(u1)])
          
            u2=[0,]*DIM
            u2[i]=2
            v2.append(self.iNto1[tuple(u2)])
            
            m1.append(torch.tensor(self.get_order(self.u1[i][0].tolist(),dim=i)) + 1)
                      
        self.v1=torch.tensor(v1)
        self.v2=torch.tensor(v2)
        self.m1=m1
        
        
        
class index_table_global_():
    def __init__(self):
        self.cache={}
    def get(self,ORDER,DIM):
        if (ORDER,DIM) not in self.cache:
            self.cache[(ORDER,DIM)]=index_table_expand(ORDER,DIM)
        return self.cache[(ORDER,DIM)]
    
index_tables = index_table_global_()