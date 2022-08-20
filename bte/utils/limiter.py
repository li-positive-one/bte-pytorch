import torch
import torch.nn.functional as F


def van_leer_limiter(r:torch.Tensor)->torch.Tensor:
    r"""
    van_leer_limiter

    .. math::
        \phi _{{vl}}(r)={\frac  {r+\left|r\right|}{1+\left|r\right|}}
    
    Args:
        r (torch.Tensor): [description]

    Returns:
        torch.Tensor: [description]
    """
    return (r + torch.abs(r)) / (1.0 + torch.abs(r))


def limiter_minmod(theta:torch.Tensor)->torch.Tensor:
    r"""
    limiter_minmod

    Args:
        theta (torch.Tensor): [description]

    .. math::
        \phi _{{mm}}(r)=\max \left[0,\min \left(1,r\right)\right]

    Returns:
        torch.Tensor: [description]
    """
    return F.relu(torch.minimum(torch.ones_like(theta), theta))


def limiter_superbee(theta:torch.Tensor)->torch.Tensor:
    r"""
    limiter_superbee 

    Args:
        theta (torch.Tensor): [description]

    .. math::
        \phi _{{sb}}(r)=\max \left[0,\min \left(2r,1\right),\min \left(r,2\right)\right]

    Returns:
        torch.Tensor: [description]
    """
    return torch.maximum(
        torch.minimum(2 * theta, torch.ones_like(theta)),
        torch.minimum(theta, 2 * torch.ones_like(theta)),
    )


def limiter_mc(theta:torch.Tensor)->torch.Tensor:
    r"""
    limiter_mc 

    Args:
        theta (torch.Tensor): [description]

    .. math::
        \phi _{{mc}}(r)=\max \left[0,\min \left(2r,0.5(1+r),2\right)\right]

    Returns:
        torch.Tensor: [description]
    """
    return torch.maximum(
        torch.zeros_like(theta),
        torch.minimum(
            torch.minimum(2 * theta, 2 * torch.ones_like(theta)), 0.5 * (theta + 1)
        ),
    )


def limiter_zero(theta:torch.Tensor)->torch.Tensor:
    """
    limiter_zero Always return 0. Linear reconstruction with limiter_zero is equal to constant reconstruction.

    .. math::
        \phi _{{0}}(r)=0

    Args:
        theta (torch.Tensor): [description]

    Returns:
        torch.Tensor: zeros tensor shape same with theta.
    """
    return torch.zeros_like(theta)
