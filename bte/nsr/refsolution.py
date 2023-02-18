from bte.dvm import solver as bte_bgk_solver
import torch
import torch.nn as nn


def config_to_solver(config):
    pass

class bgk_solver(nn.Module):
    def __init__(self, nx, vmin, vmax, nv, device="cuda", BC_type="periodic", dtype=torch.float64) -> None:
        super().__init__()
        solver = bte_bgk_solver.BGKSolver(
            -0.5, 0.5, nx, vmin, vmax, nv, device=device, BC_type=BC_type, dtype = dtype
        )
        solver.set_space_order(2)
        solver.set_time_stepper("bgk-RK2")
        self.device = device
        if self.device == "cuda":
            solver.cuda()
        self.solver = solver
        CFL = 0.45
        max_dt = CFL * 1.0 / nx / max(vmax[0], -vmin[0])
        self.max_dt = max_dt

    def forward(self, f0, kn, t):
        f0 = torch.from_numpy(f0).unsqueeze(0).repeat(1, 1, 1).to(self.device)
        self.solver.set_initial(kn, f0)
        self.solver.dis.dx = self.solver.dis.dx.to(self.device)
        self.solver.solve_to(t, self.max_dt)
        return self.solver.dis.f.cpu().numpy()
