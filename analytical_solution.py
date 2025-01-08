import numpy as np
import torch


def heat_eq_1D(x, t, alpha):
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

def heat_eq_1D_with_source(x, t, alpha):
    """
    Analytical solution to the 1D heat equation with a source term.
    With boundary conditions:
    - u(x, 0) = sin(pi x) / (alpha * pi^2 + 1)
    - u(0, t) = u(1, t) = 0

    Args:
    - t: time
    - x: position
    - alpha: thermal diffusivity

    Returns:
    - Solution tensor
    """

    # Computed manually
    return (torch.sin(np.pi * x) / (alpha * np.pi**2 + 1)) * \
        (1 - torch.exp(-(alpha * np.pi**2 + 1) * t))