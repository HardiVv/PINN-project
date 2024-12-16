import numpy as np
import torch


def exact_solution(x, t, alpha):
    """
    Analytical solution to the 1D heat equation
    u(x, 0) = sin(pi x),
    u(0, t) = u(1, t) = 0 (boundary conditions)
    """
    return torch.exp(-np.pi**2 * alpha * t) * torch.sin(np.pi * x)
