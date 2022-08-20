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
    def __init__(self, b1, A1, b2, A2) -> None:
        """
        __init__ [summary]

         u_t = f(u) + g(u)
        # f(u) non-stiff, explicit
        # g(u) stiff, implicit

        Args:
            b1 ([type]): [description]
            A1 ([type]): [description]
            b2 ([type]): [description]
            A2 ([type]): [description]

        Raises:
            ValueError: A1,b1,A2,b2 shape must be consistent.
        """
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
        # Bucher(b1,A1) is for the f
        # Bucher(b2,A2) is for the g

    def __call__(self, f, y0, h, getTau, getU, Maxwellian):
        y_list = []
        f_list = []
        g_list = []
        U_list = []
        tau_list = []
        for s in range(self.Stage):
            #print(y_list,f_list,self.A1[s, :])
            #print([self.A1[s, i] * f_list[i] for i in range(s)])
            res_f = sum([self.A1[s, i] * f_list[i] for i in range(s)])
            res_g = sum([self.A2[s, i] * g_list[i] for i in range(s)])
            B = y0 + h * res_f + h * res_g
            U = getU(B)
            U_list.append(U)
            tau = getTau(U)
            tau_list.append(tau)
            # print(tau.shape,B.shape,Maxwellian(U).shape)
            y = (tau * B + h * self.A2[s, s] * Maxwellian(U)) / (
                tau + h * self.A2[s, s]
            )
            y_list.append(y)
            f_list.append(f(y))
            g = (Maxwellian(U) - y_list[s]) / tau_list[s]
            g_list.append(g)
            # print(y.sum())

        # print(len(f_list),len(g_list))
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
