import logging
import math
import os
import sys
import time
sys.path.insert(0, os.path.abspath('../..'))

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytorch_warmup as warmup
import torch
from matplotlib import pyplot as plt
from torch.func import jacfwd, vmap

from bte.dvm import solver as bgk_solver
from bte.nsr.dataset import SmoothD1Case, SodD1Case
from bte.nsr.model import (LossCompose, Problem, SplitNet, _m012, maxwellian,
                           rho_u_theta)
from bte.nsr.reduced_collision import get_reduced_kernel
from bte.utils.gas import get_kn_bzm, get_mu, get_potential


logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.float32)
device = "cuda"


@hydra.main(version_base='1.2', config_path="config", config_name="BGK_D1V3")
def train(config):
    print(config)
    nv = config["vmesh"]["nv"]
    vmin = config["vmesh"]["vmin"]
    vmax = config["vmesh"]["vmax"]
    Kn = config["Kn"]

    kn = Kn

    omega = 0.81
    alpha = get_potential(omega)
    mu_ref = get_mu(alpha, omega, kn)
    kn_bzm = get_kn_bzm(alpha, mu_ref)

    nx = config.xtmesh.nx
    xmin, xmax = config.xtmesh.xmin, config.xtmesh.xmax

    if config.case == "Wave":
        datagen = SmoothD1Case(config)
        (IV_x, IV_t, IV_f), (BV_x1, BV_t1, BV_x2,
                             BV_t2), (IN_x, IN_t) = datagen.get()
        BDC = "periodic"
    else:
        datagen = SodD1Case(config)
        (IV_x, IV_t, IV_f), (BV_x, BV_t, BV_f), (IN_x, IN_t) = datagen.get()
        BDC = "constant"

    if config["collision"] == "RC":
        t_final, dt = 0.1, 0.01
        CFL = 0.45
        max_dt = CFL * 1.0 / nx / vmax[0]
        print("max_dt is", max_dt)
        soln = []
        nt = 10
        f0 = datagen.init_value()
        solver_bgk = bgk_solver.BGKSolver(
            xmin, xmax, nx, vmin, vmax, nv, device='cpu', BC_type=BDC, bgk_simple_kn="mu_ref")
        solver_bgk.mu_ref = mu_ref
        solver_bgk.omega = omega
        solver_bgk.set_time_stepper("bgk-RK2")
        solver_bgk.set_collisioner("BGK")
        solver_bgk.set_initial(kn, f0)
        solver_bgk.dis.v_meta = solver_bgk.dis.v_meta.to(device=device)
        solver_bgk.dis = solver_bgk.dis.to(device=device)
        solver_bgk.cuda()
        paths_bgk = torch.zeros((nt, nx, math.prod(nv)))
        for t in range(nt):
            print(t)
            solver_bgk.solve_to(dt, max_dt)
            paths_bgk[t] = solver_bgk.dis.f.cpu()

    if config.collision == "RC":
        perm = torch.randperm(400*10)
        k = 2000
        t2 = -1
        idx = perm[:k]
        pathsf2 = paths_bgk.flatten(0, 1)
        traindata = pathsf2[idx]
        RC, BASE1, BASE2 = get_reduced_kernel(config, traindata, 40)

    if config["collision"] == "BGK":
        solver = bgk_solver.BGKSolver(
            xmin, xmax, nx, vmin, vmax, nv, device='cpu', BC_type=BDC, bgk_simple_kn="simple")
    elif config["collision"] == "FBGK":
        solver = bgk_solver.BGKSolver(
            xmin, xmax, nx, vmin, vmax, nv, device='cpu', BC_type=BDC, bgk_simple_kn="mu_ref")
    elif config["collision"] in ["FSM", "RC"]:
        solver = bgk_solver.BGKSolver(
            xmin, xmax, nx, vmin, vmax, nv, device='cpu', BC_type=BDC, bgk_simple_kn="mu_ref")
    else:
        raise ValueError

    solver.mu_ref = mu_ref
    solver.omega = omega
    print("x.dtype:",  solver.dis.x.dtype)
    solver.set_space_order(2)

    if config["collision"] in ["BGK", "FBGK"]:
        solver.set_time_stepper("bgk-RK2")
        solver.set_collisioner("BGK")
    elif config["collision"] in ["FSM", "RC"]:
        solver.set_time_stepper("IMEX")
        solver.set_collisioner("FFT")
    else:
        raise ValueError

    x, v = solver.dis.x.cpu(), solver.dis.v.cpu()

    f0 = datagen.init_value()
    solver.to(device=device)

    if config.collision in ["FSM", "RC"]:
        kn = kn_bzm
    else:
        pass
    solver.set_initial(kn, f0)
    solver.dis.v_meta = solver.dis.v_meta.to(device=device)
    solver.dis = solver.dis.to(device=device)

    fig = plt.figure()
    plt.plot(solver.dis.density().cpu()[0, :, 0], label='rho')
    plt.plot(solver.dis.velocity().cpu()[0, :, 0], label='ux')
    plt.plot(solver.dis.velocity().cpu()[0, :, 1], label='uy')
    plt.plot(solver.dis.temperature().cpu()[0, :, 0], label='T')
    plt.legend()
    plt.savefig("DVM-t0.png")

    t_final, dt = 0.1, 0.01
    CFL = 0.45
    max_dt = CFL * 1.0 / nx / vmax[0]
    print("max_dt is", max_dt)
    soln = []
    nt = 10
    paths2 = torch.zeros((nt, nx, math.prod(nv)))
    for t in range(nt):
        print(t)
        solver.solve_to(dt, max_dt)
        paths2[t] = solver.dis.f.cpu()

    print(solver.dis.density().shape)
    print(solver.dis.density().max())

    fig = plt.figure()
    plt.plot(solver.dis.density().cpu()[0, :, 0], label='rho')
    plt.plot(solver.dis.velocity().cpu()[0, :, 0], label='ux')
    plt.plot(solver.dis.velocity().cpu()[0, :, 1], label='uy')
    plt.plot(solver.dis.temperature().cpu()[0, :, 0], label='T')
    plt.legend()
    plt.savefig("DVM-t1.png")

    v = solver.dis.v_meta.v.float()
    w = solver.dis.v_meta.w.float()

    VDIS = v
    WDIS = w

    model = SplitNet(config.network.neurons, VDIS, config.network.multires)

    model = model.to(device=device)
    vdis = v

    def net(x, t):
        inputs = torch.cat([x, t], dim=-1)
        return model(inputs)

    ggrad = vmap(jacfwd(net, (0, 1)))

    def pinn(x, t, net):
        f = net(x, t)

        fx, ft = ggrad(x, t)
        f_x = (fx[..., 0])
        f_t = (ft[..., 0])

        if config.collision == "BGK":
            rho, u, theta = rho_u_theta(f, VDIS, WDIS)
            f_M = maxwellian(VDIS, rho, u, theta)
            pde = f_t + VDIS[..., 0] * f_x - 1/kn*(f_M-f)
        elif config.collision == "FBGK":
            rho, u, theta = rho_u_theta(f, VDIS, WDIS)
            f_M = maxwellian(VDIS, rho, u, theta)
            kn_bgk = (mu_ref*2/(2*theta)**(1-omega)/rho)
            kn_bgk = torch.maximum(kn_bgk, 0.001*torch.ones_like(kn_bgk))
            pde = f_t + VDIS[..., 0] * f_x - 1/kn_bgk*(f_M-f)
        elif config.collision == "FSM":
            pde = f_t + VDIS[..., 0] * f_x - 1 / \
                kn_bzm * solver.coller.do_collision(f)
        elif config.collision == "RC":
            pde = (f_t + VDIS[..., 0] * f_x) - 1/kn_bzm * RC.do_collision(f)
        else:
            raise ValueError
        return pde

    # 构造光滑初值问题

    wc = (math.prod(nv)+7*1)*3
    LC3 = LossCompose(wc).to(device=device)

    smooth_l1_loss = torch.nn.MSELoss(reduction='none')

    def criterion(x, y): return smooth_l1_loss(x, y).mean(dim=0)

    def criterion_norm(x): return smooth_l1_loss(
        x, torch.zeros_like(x)).mean(dim=0)

    def prim_norm(f, order=2):
        m1, m2, m3 = _m012(f, vdis, w)
        return torch.cat([criterion_norm(m1), criterion_norm(m2), criterion_norm(m3)], dim=-1)

    start_time = time.time()
    optimizer = torch.optim.Adam(
        [{'params': model.parameters(), 'lr': 1e-3},
         {'params': LC3.parameters(), 'lr': 1e-3}])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, config.optim["Adam_steps"])
    warmup_scheduler = warmup.LinearWarmup(
        optimizer, warmup_period=config.optim["Adam_steps"]//10)
    problem = Problem(config, model, LC3)
    # problem.load_state_dict(torch.load(MODELPATH))

    for i in range(config.optim["Adam_steps"]):
        IV_pred = net(IV_x, IV_t)

        loss_IV = criterion(IV_pred, IV_f)*config.network.weight[0]
        loss_IV2 = prim_norm(IV_pred-IV_f)*config.network.weight[0]

        if config.case == "Wave":
            BV_pred1 = net(BV_x1, BV_t1)
            BV_pred2 = net(BV_x2, BV_t2)
            loss_BV = criterion(BV_pred1, BV_pred2)*config.network.weight[1]
            loss_BV2 = prim_norm(BV_pred1-BV_pred2)*config.network.weight[1]
        elif config.case == "Sod":
            BV_pred = net(BV_x, BV_t)
            loss_BV = criterion(BV_pred, BV_f)*config.network.weight[1]
            loss_BV2 = prim_norm(BV_pred-BV_f)*config.network.weight[1]
        else:
            raise ValueError

        pde = pinn(IN_x, IN_t, net)
        loss_pinn = criterion_norm(
            pde)*config.network.weight[2]*min(1, (0.1+2*(i)/config.optim["Adam_steps"]))
        loss_pinn2 = prim_norm(
            pde)*config.network.weight[2]*min(1, (0.1+2*(i)/config.optim["Adam_steps"]))

        loss_sum = LC3(
            torch.cat([loss_IV, loss_IV2, loss_BV, loss_BV2, loss_pinn, loss_pinn2]))

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()
        with warmup_scheduler.dampening():
            scheduler.step()

        if i % 10 == 0:
            e_sum = loss_sum.item()
            e_IV = loss_IV.mean().item()
            e_IV2 = loss_IV2.mean().item()
            e_BV = loss_BV.mean().item()
            e_BV2 = loss_BV2.mean().item()
            e_pinn = loss_pinn.mean().item()
            e_pinn2 = loss_pinn2.mean().item()
            logger.info(
                f"{i},{time.time()-start_time:.2e}:{e_sum:.2e},{e_IV:.2e},{e_IV2:.2e},{e_BV:.2e},{e_BV2:.2e},{e_pinn:.2e},{e_pinn2:.2e}")
        if i in [10, 100, 200, 300, 400, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9999]:
            torch.save(problem.state_dict(), f"problem_epoch{i}.ckpt")
    model.eval()

    torch.save(problem.state_dict(), "problem.ckpt")

    VA_x = torch.linspace(-0.5, 0.5, 400)[..., None].to(device=device)
    VA_t = 0.0*torch.ones_like(VA_x).to(device=device)
    y = model(torch.concat((VA_x, VA_t), dim=-1))
    rho, u, theta = rho_u_theta(y, vdis, w)
    rho = rho.cpu()
    u = u.cpu()
    theta = theta.cpu()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(VA_x.cpu()[..., 0], rho.detach().cpu(), label=r'NN-$\rho$')
    ax.plot(VA_x.cpu()[..., 0], u.detach().cpu()[..., 0:1], label=r'NN-$u_x$')
    ax.plot(VA_x.cpu()[..., 0], theta.detach().cpu(), label=r'NN-$T$')

    ax.plot(solver.dis.x.cpu(), solver.dis.density(
        f0).cpu()[0, :, 0], "--", label=r'FSM-$\rho$')
    ax.plot(solver.dis.x.cpu(), solver.dis.velocity(
        f0).cpu()[0, :, 0:1], "--", label=r'FSM-$u_x$')
    ax.plot(solver.dis.x.cpu(), solver.dis.temperature(
        f0).cpu()[0, :, 0], "--", label=r'FSM-$T$')
    plt.legend(loc='upper right')
    plt.savefig("t0.png")

    np.savez("t0.npz", rho=rho.detach().cpu(), u=u.detach().cpu()[..., 0:1], theta=theta.detach().cpu(),
             rho0=solver.dis.density(f0).cpu()[0, :, 0],
             u0=solver.dis.velocity(f0).cpu()[0, :, 0:1],
             theta0=solver.dis.temperature(f0).cpu()[0, :, 0])

    rho0 = solver.dis.density(f0).cpu()[-1, :, 0:1]
    u0 = solver.dis.velocity(f0).cpu()[-1, :, :]
    theta0 = solver.dis.temperature(f0).cpu()[-1, :, 0:1]

    err1 = (((rho-rho0)**2).mean()/((rho0)**2).mean())**0.5
    err2 = (((u-u0)**2).mean()/(1+(u0)**2).mean())**0.5
    err3 = (((theta-theta0)**2).mean()/((theta0)**2).mean())**0.5
    logger.info(
        f"err at t=0.0: {err1.item():.3e}\t{err2.item():.3e}\t{err3.item():.3e}\t")

    VA_x = torch.linspace(-0.5, 0.5, 400)[..., None].to(device=device)
    VA_t = 0.1*torch.ones_like(VA_x).to(device=device)
    y = net(VA_x, VA_t)
    rho, u, theta = rho_u_theta(y, vdis, w)
    rho = rho.cpu()
    u = u.cpu()
    theta = theta.cpu()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(VA_x.cpu()[..., 0], rho.detach().cpu(), label=r'NN-$\rho$')
    ax.plot(VA_x.cpu()[..., 0], u.detach().cpu()[..., 0:1], label=r'NN-$u_x$')
    ax.plot(VA_x.cpu()[..., 0], theta.detach().cpu(), label=r'NN-$T$')

    ax.plot(solver.dis.x.cpu(), solver.dis.density().cpu()
            [0, :, 0], "--", label=r'FSM-$\rho$')
    ax.plot(solver.dis.x.cpu(), solver.dis.velocity().cpu()
            [0, :, 0:1], "--", label=r'FSM-$u_x$')
    ax.plot(solver.dis.x.cpu(), solver.dis.temperature().cpu()
            [0, :, 0], "--", label=r'FSM-$T$')
    plt.legend(loc='upper right')

    plt.savefig("t1.png")

    np.savez("t1.npz", rho=rho.detach().cpu(), u=u.detach().cpu()[..., 0:1], theta=theta.detach().cpu(),
             rho0=solver.dis.density().cpu()[0, :, 0],
             u0=solver.dis.velocity().cpu()[0, :, 0:1],
             theta0=solver.dis.temperature().cpu()[0, :, 0])

    rho0 = solver.dis.density().cpu()[-1, :, 0:1]
    u0 = solver.dis.velocity().cpu()[-1, :, :]
    theta0 = solver.dis.temperature().cpu()[-1, :, 0:1]

    err1 = (((rho-rho0)**2).mean()/((rho0)**2).mean())**0.5
    err2 = (((u-u0)**2).mean()/(1+(u0)**2).mean())**0.5
    err3 = (((theta-theta0)**2).mean()/((theta0)**2).mean())**0.5
    logger.info(
        f"err at t=0.1: {err1.item():.3e}\t{err2.item():.3e}\t{err3.item():.3e}\t")


if __name__ == "__main__":
    train()