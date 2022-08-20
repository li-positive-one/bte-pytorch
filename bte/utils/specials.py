import torch

def eval_hermitenorm(n: int, x: torch.Tensor):
    if n < 0:
        return float("nan") * torch.ones_like(x)
    elif n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return x
    else:
        y3 = torch.zeros_like(x)
        y2 = torch.ones_like(x)
        for k in range(n, 1, -1):
            y1 = x * y2 - k * y3
            y3 = y2
            y2 = y1
        return x * y2 - y3

def eval_hermitenorm_seq(n: int, x: torch.Tensor):
    y=torch.zeros((n+1,)+x.shape)
    if n < 0:
        raise ValueError
    if n >= 0:
        y[0]=torch.ones_like(x)
    if n >= 1:
        y[1]=x
    if n >= 2:
        for j in range(2,n+1):
            y[j]=x*y[j-1]-(j-1)*y[j-2]
    return y