import numpy as np
import torch


def exact_solution(x, t, alpha):
    """
    Analytical solution to the 1D heat equation.
    With boundary conditions
    - u(x, 0) = sin(pi x)
    - u(0, t) = u(1, t) = 0

    Args:
    - t: time
    - x: position
    - alpha: thermal diffusivity

    Returns:
    - Solution tensor
    """
    return torch.exp(-np.pi**2 * alpha * t) * torch.sin(np.pi * x)
