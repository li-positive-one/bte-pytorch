import torch

def linear_reconstruction_MinMod(fL:torch.Tensor, f:torch.Tensor, fR:torch.Tensor)->torch.Tensor:
    """linear_reconstruction using MinMod limiter

    Args:
        fL (torch.Tensor): left cell 
        f (torch.Tensor): cell
        fR (torch.Tensor): right cell

    Returns:
        torch.Tensor: leftRec, rightRec
    """
    theta1 = f - fL
    theta2 = fR - f
    slope = (theta1.sign()+theta2.sign())/2*torch.minimum(theta1.abs(),theta2.abs())
    RecL = f - slope / 2 
    RecR = f + slope / 2 
    return RecL, RecR

def linear_reconstruction_VanLeer(fL:torch.Tensor, f:torch.Tensor, fR:torch.Tensor)->torch.Tensor:
    """linear_reconstruction using VanLeer limiter

    Args:
        fL (torch.Tensor): left cell 
        f (torch.Tensor): cell
        fR (torch.Tensor): right cell

    Returns:
        torch.Tensor: leftRec, rightRec
    """
    theta1 = f - fL
    theta2 = fR - f
    slope = (theta1.sign()+theta2.sign())*theta1.abs()*theta2.abs()/(theta1.abs()+theta2.abs()+1e-8)
    RecL = f - slope / 2 
    RecR = f + slope / 2 
    return RecL, RecR