from collections.abc import Sequence
import torch
import matplotlib.pyplot as plt
class Surface:
    def domain(self) -> tuple[Sequence[int | float], Sequence[int| float]]: raise NotImplementedError
    def start(self) -> Sequence[int | float]: raise NotImplementedError
    def minimum(self) -> Sequence[int | float]: raise NotImplementedError
    def __call__(self, x, y) -> torch.Tensor: raise NotImplementedError
    def plot(self, cmap='gray', steps = 1000, levels=20, figsize=None, show=False):
        xlim, ylim = self.domain()
        xstep = (xlim[1] - xlim[0]) / steps
        ystep = (ylim[1] - ylim[0]) / steps
        y, x = torch.meshgrid(torch.arange(ylim[0], ylim[1], xstep), torch.arange(xlim[0], xlim[1], ystep), indexing='xy')
        z = [self(xv, yv).numpy() for xv, yv in zip(x, y)]
        fig, ax = plt.subplots(1, 1, figsize=figsize, layout='tight')
        ax.set_title("Loss landscape")
        ax.set_frame_on(False)
        cmesh = ax.pcolormesh(x.numpy(), y.numpy(), z, cmap=cmap, zorder=0)
        if levels: ax.contour(x.numpy(), y.numpy(), z, levels=levels)
        current_coord = self.start()
        minimum = self.minimum()
        ax.scatter([current_coord[0]], [current_coord[1]], s=6)
        ax.scatter([minimum[0]], [minimum[1]], s=6, c='red')
        fig.colorbar(cmesh, ax=ax)
        if show: plt.show()
        return fig, ax


class PowSum(Surface):
    def __init__(self, xpow, ypow, xmul=1, ymul=1, xadd=0, yadd=0):
        self.xpow = xpow
        self.ypow = ypow
        self.xmul = xmul
        self.ymul = ymul
        self.xadd = xadd
        self.yadd = yadd

    def __call__(self, x, y):
        return (x * self.xmul + self.xadd).abs() ** self.xpow + (y * self.ymul + self.yadd).abs() ** self.ypow

    def domain(self): return (-1,1), (-1,1)
    def start(self): return (-0.93, 0.88)
    def minimum(self): return (self.xadd, self.yadd)

cross = PowSum(0.5, 0.5)
star = PowSum(1, 1)
convex = PowSum(2, 2)

class Rosenbrock(Surface):
    def __init__(self, a = 1, b = 100):
        self.a = a
        self.b = b
    def __call__(self, x, y):
        return (self.a - x) ** 2 + self.b * (y - x ** 2) ** 2
    def domain(self): return (-2, 2), (-1, 3)
    def start(self): return (-1.9, -0.9)
    def minimum(self): return (1, 1)
rosenbrock = Rosenbrock()

class Rastrigin(Surface):
    def __init__(self, A=10):
        self.A = A

    def __call__(self, x, y):
        return self.A * 2 + x ** 2 - self.A * torch.cos(2 * torch.pi * x) + y ** 2 - self.A * torch.cos(2 * torch.pi * y)

    def domain(self): return (-5.12, 5.12), (-5.12, 5.12)
    def start(self): return (-4.5, 4.3)
    def minimum(self): return (0, 0)
rastrigin = Rastrigin()

class Ackley(Surface):
    def __init__(self, a=20, b=0.2, c=2 * torch.pi, domain=32.768):
        self.a = a
        self.b = b
        self.c = c
        self.domain_ = domain

    def __call__(self, x, y):
        return -self.a * torch.exp(-self.b * torch.sqrt((x ** 2 + y ** 2) / 2)) - torch.exp(
            (torch.cos(self.c * x) + torch.cos(self.c * y)) / 2) + self.a + torch.exp(torch.tensor(1))
    def domain(self): return (-self.domain_, self.domain_), (-self.domain_, self.domain_)
    def start(self): return (-self.domain_ + self.domain_ / 100, self.domain_ - self.domain_ / 95)
    def minimum(self): return (0, 0)

ackley = Ackley()

class Sphere(Surface):
    def __call__(self, x, y):
        return x ** 2 + y ** 2
    def domain(self): return (-5.12, 5.12), (-5.12, 5.12)
    def start(self): return (-5, 4.9)
    def minimum(self): return (0, 0)
sphere = Sphere()

class Beale(Surface):
    def __init__(self, a=1.5, b=2.25, c=2.625):
        self.a = a
        self.b = b
        self.c = c
    def __call__(self, x, y):
        return (self.a - x + x * y) ** 2 + (self.b - x + x * y ** 2) ** 2 + (self.c - x + x * y ** 3) ** 2
    def domain(self): return (-4.5, 4.5), (-4.5, 4.5)
    def start(self): return (-4, -4)
    def minimum(self): return (3, 0.5)
beale = Beale()

class Booth(Surface):
    def __call__(self, x, y):
        return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2
    def domain(self): return (-10, 10), (-10, 10)
    def start(self): return (0, -8)
    def minimum(self): return (1, 3)
booth = Booth()

class GoldsteinPrince(Surface):
    def __call__(self, x, y):
        return (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * (
                    30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
    def domain(self): return (-3, 3), (-3, 3)
    def start(self): return (-2.9, -1.9)
    def minimum(self): return (0, -1)
goldstein_prince = GoldsteinPrince()