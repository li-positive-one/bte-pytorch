import numpy as np

# u_t = f(u) + g(u)
# f(u) non-stiff, explicit
# g(u) stiff, implicit

# Butcher tableau
# c|A
# ---
#  |b

# g(u,args)  BGK part: g=1/tau(M(f)-f)
# f(u,args)  flux and other collision part

# Implicit-Explicit schemes for BGK kinetic equations

# y size [B,C,W]

class IMEX_RK:
    r"""IMEX-RK method

        y_t = f(t,y) + g(t,y)

        # f(t,y) non-stiff, explicit
        # g(t,y) stiff, implicit

        Args:
            b1 ([type]): [description]
            A1 ([type]): [description]
            b2 ([type]): [description]
            A2 ([type]): [description]

        Raises:
            ValueError: A1,b1,A2,b2 shape must be consistent.
    """

    def __init__(self, b1, A1, b2, A2) -> None:
        self.Stage = len(b1)
        if len(b2) != self.Stage:
            raise ValueError("Invalid Butcher tableau")
        if A1.shape[0] != A1.shape[1] or A1.shape[0] != self.Stage:
            raise ValueError("Invalid Butcher tableau")
        if A2.shape[0] != A2.shape[1] or A2.shape[0] != self.Stage:
            raise ValueError("Invalid Butcher tableau")
        self.b1 = b1
        self.A1 = A1
        self.b2 = b2
        self.A2 = A2
        self.c1 = [sum(a1) for a1 in A1]
        self.c2 = [sum(a2) for a2 in A2]

    def __call__(self, t0, y0, h, f, g, g_solver):
        r"""
        __call__ [summary]

        Args:
            f ([type]): [description]
            t0 ([type]): [description]
            y0 ([type]): [description]
            h ([type]): [description]
            g ([type]): [description]
            g_solver ([type]): the solver of u^{(i)}=H(\tilde u^{(i)},\Delta t A_{ii};G), u=g_solver(tilde_u, \Delta t A_{ii})

        Returns:
            [type]: [description]
        """
        f_list = []
        g_list = []
        for s in range(self.Stage):
            res_f = sum([self.A1[s, i] * f_list[i] for i in range(s)])
            res_g = sum([self.A2[s, i] * g_list[i] for i in range(s)])
            tilde_u = y0 + h * (res_f + res_g)
            tnow = t0 + self.c2[s]*h
            y = g_solver(tilde_u, h * self.A2[s, s], tnow)
            f_list.append(f(self.c1[s],y))
            g_list.append(g(self.c2[s],y))
        y = y0 + h * (
            sum([f_list[j] * self.b1[j] for j in range(self.Stage)])
            + sum([(g_list[j]) * self.b2[j] for j in range(self.Stage)])
        )
        return y


class IMEX_RK1(IMEX_RK):
    def __init__(self) -> None:
        b1 = np.array([1])
        A1 = np.array([[0]])
        b2 = np.array([1])
        A2 = np.array([[1]])
        super().__init__(b1, A1, b2, A2)


class IMEX_RK2(IMEX_RK):
    def __init__(self) -> None:
        b1 = np.array([0, 0.5, 0.5])
        A1 = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
        b2 = np.array([0, 0.5, 0.5])
        A2 = np.array([[0.5, 0, 0], [-0.5, 0.5, 0], [0, 0.5, 0.5]])
        super().__init__(b1, A1, b2, A2)


class IMEX_RK3(IMEX_RK):
    def __init__(self) -> None:
        alpha = 0.24169426078821
        beta = 0.06042356519705
        eta = 0.1291528696059
        b1 = np.array([0, 1 / 6, 1 / 6, 2 / 3])
        A1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 1 / 4, 1 / 4, 0]])
        b2 = np.array([0, 1 / 6, 1 / 6, 2 / 3])
        A2 = np.array(
            [
                [alpha, 0, 0, 0],
                [-alpha, alpha, 0, 0],
                [0, 1 - alpha, alpha, 0],
                [beta, eta, 0.5 - alpha - beta - eta, alpha],
            ]
        )
        super().__init__(b1, A1, b2, A2)


#  Stiffly Accurate Scheme, which mean A2[-1,:]=b2
class IMEX_RK1_stiff(IMEX_RK):
    def __init__(self) -> None:
        b1 = np.array([1, 0])
        A1 = np.array([[0, 0], [1, 0]])
        b2 = np.array([0, 1])
        A2 = np.array([[0, 0], [0, 1]])
        super().__init__(b1, A1, b2, A2)


class IMEX_RK2_stiff(IMEX_RK):
    def __init__(self) -> None:
        b1 = np.array([0, 1, 0])
        A1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        b2 = np.array([0.5, 0, 0.5])
        A2 = np.array([[0, 0, 0], [0, 1, 0], [0.5, 0.0, 0.5]])
        super().__init__(b1, A1, b2, A2)