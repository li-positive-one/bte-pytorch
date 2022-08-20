import numpy as np
from collections.abc import Callable

class RK:
    r"""Explicit Runge-Kutta Method

    Create a runge-kutta method to integrate :math:`u_t=f(t,u)`

    .. math::
        \begin{aligned} y_{n+1}&=y_{n}+h\sum _{i=1}^{s}b_{i}k_{i}\\k_{1}&=f(t_{n},y_{n}),\\k_{2}&=f(t_{n}+c_{2}h,y_{n}+h(a_{21}k_{1})),\\k_{3}&=f(t_{n}+c_{3}h,y_{n}+h(a_{31}k_{1}+a_{32}k_{2})),\\&\ \ \vdots \\k_{s}&=f(t_{n}+c_{s}h,y_{n}+h(a_{s1}k_{1}+a_{s2}k_{2}+\cdots +a_{s,s-1}k_{s-1})).
        \end{aligned}

    Args:
        c (list): runge-kutta coef
        b (list): runge-kutta coef
        A (list): runge-kutta coef

    Butcher tableau
    
    .. math::
        \begin{array}{c|c}
        c & A \\ \hline
          & b
        \end{array}

    """

    def __init__(self, c: list, b: list, A: list) -> None:
        self.Stage = len(c)
        if len(b) != self.Stage or A.shape[0] != A.shape[1] or A.shape[0] != self.Stage:
            raise ValueError("Invalid Butcher tableau")
        self.c = c
        self.b = b
        self.A = A

    def __call__(self, f: Callable, t0: float, u0: float, h: float, f0=None):
        """ single step of runge-kutta

        u_t = f(u,t)
        
        f_0=f(u0,t0)

        Args:
            f (Callable): [description]
            t0 (float): [description]
            u0 (float): [description]
            h (float): [description]
            f0 ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        k_list = []
        for i in range(self.Stage):
            if i == 0 and f0 is not None and self.A[0, 0] == 0:
                k_list.append(f0)
            else:
                k_list.append(
                    f(
                        t0 + self.c[i] * h,
                        u0 + h * sum([self.A[i, j] * k_list[j] for j in range(i)]),
                    )
                )
        u = u0 + h * sum([k_list[j] * self.b[j] for j in range(self.Stage)])
        return u


class RK1(RK):
    r""" 1st order Runge-Kutta

    Butcher tableau:
    
    .. math::
        \begin{array}{c|c}
        0 & 0 \\ \hline
          & 1
        \end{array}
    """

    def __init__(self) -> None:
        c = np.array([0])
        b = np.array([1])
        A = np.array([[0]])
        super().__init__(c, b, A)


class RK2(RK):
    r""" 2nd order Runge-Kutta

    Butcher tableau:
    
    .. math::
        \begin{array}{c|cc}
        0 & 0 & 0\\ 
        \alpha & \alpha & 0 \\ \hline
          & (1 - \frac{1}{2\alpha}) & \frac{1}{2\alpha}
        \end{array}

    Args:
        alpha (float, optional): Defaults to 1/2.
    """

    def __init__(self, alpha:float=1/2) -> None:
        c = np.array([0, alpha])
        b = np.array([1 - 1 / (2 * alpha), 1 / (2 * alpha)])
        A = np.array([[0, 0], [alpha, 0]])
        super().__init__(c, b, A)


class RK3_SSP(RK):
    def __init__(self) -> None:
        c = np.array([0, 1, 0.5])
        b = np.array([1 / 6, 1 / 6, 2 / 3])
        A = np.array([[0, 0, 0], [1, 0, 0], [1 / 4, 1 / 4, 0]])
        super().__init__(c, b, A)


class RK4(RK):
    r""" 4nd order Runge-Kutta

    Butcher tableau:

    .. math::
        \begin{array}{c|cc}
        0 &  & & &\\ 
        \frac{1}{2} & \frac{1}{2} & & & \\ 
        \frac{1}{2} & 0 & \frac{1}{2} & & \\
        1 & 0 & 0 & 1 & \\ \hline
        & \frac{1}{6} & \frac{1}{3} & \frac{1}{3} & \frac{1}{6}
        \end{array}

    """

    def __init__(self) -> None:
        c = np.array([0, 0.5, 0.5, 1])
        b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
        A = np.array([[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 1 / 2, 0, 0], [0, 0, 1, 0]])
        super().__init__(c, b, A)


class RK4_2(RK):
    def __init__(self) -> None:
        c = np.array([0, 1 / 3, 2 / 3, 1])
        b = np.array([1 / 8, 3 / 8, 3 / 8, 1 / 8])
        A = np.array([[0, 0, 0, 0], [1 / 3, 0, 0, 0], [-1 / 3, 1, 0, 0], [1, -1, 1, 0]])
        super().__init__(c, b, A)
