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


def exact_solution_source(x, t, alpha):
    """
    Analytical solution to the 1D heat equation with a source term.
    With boundary conditions:
    - u(x, 0) = sin(pi x) / (alpha * pi^2 + 1)
    - u(0, t) = u(1, t) = 0
    And a source term f(x, t) = 1.

    Args:
    - t: time
    - x: position
    - alpha: thermal diffusivity

    Returns:
    - Solution tensor
    """

    # Ensures x and t are torch tensors, makes code flexible
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    if isinstance(t, np.ndarray):
        t = torch.tensor(t, dtype=torch.float32)

    # The analytical solution with the source term included
    term_1 = (torch.sin(np.pi * x) / (alpha * np.pi**2 + 1)) * (1 - torch.exp(-(alpha * np.pi**2 + 1) * t))
    term_2 = 1 / (alpha * np.pi**2 + 1)

    return term_1 + term_2
