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
    And a source term f(x, t) = 1

    Args:
    - t: time
    - x: position
    - alpha: thermal diffusivity

    Returns:
    - Solution tensor
    """

    # The analytical solution with source term
    term_1 = torch.exp(-np.pi**2 * alpha * t) * torch.sin(np.pi * x)
    term_2 = (1 - torch.exp(-t)) * (x - x**2)  # Ensures boundaries to be 0

    return term_1 + term_2