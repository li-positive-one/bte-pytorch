import math
import sys

import pytest
import torch

sys.path.append(".")

from bte.dvm import ops

config1 = [((-10,), (10,), (400,), 100),
          ((-10, -10, -10), (10, 10, 10), (100, 100, 100), 100),
          ]


@pytest.mark.parametrize('vmin,vmax,nv,nx', config1)
def test_maxwellian(vmin, vmax, nv, nx):
    D = len(nv)
    rho = torch.ones((nx, 1))
    u = torch.ones((nx, D))
    T = torch.ones((nx, 1))
    v, w, vL, wL = ops.NDmsh(vmin, vmax, nv)
    f = ops.maxwellian(v, rho, u, T)
    assert f.shape == (nx, math.prod(nv))

    rho1, u1, T1 = ops.rho_u_theta(f, v, w)
    assert torch.allclose(rho, rho1)
    assert torch.allclose(u, u1)
    assert torch.allclose(T, T1)

config2 = [ ((-10, -10, -10), (10, 10, 10), (100, 100, 100), 100),
          ]


@pytest.mark.parametrize('vmin,vmax,nv,nx', config2)
def test_collision(vmin, vmax, nv, nx):
    D = len(nv)
    rho = torch.ones((nx, 1))
    u = torch.ones((nx, D))
    T = torch.ones((nx, 1))
    v, w, vL, wL = ops.NDmsh(vmin, vmax, nv)
    f = ops.maxwellian(v, rho, u, T)
    phi,psi,phipsi=ops.init_kernel_mode_vector(vmin,vmax,nv,dtype=torch.float32)
    Q=ops.collision_fsm(f,1.0,phi,psi,phipsi)
    print(Q.abs().max())
    assert Q.abs().max()<1e-5