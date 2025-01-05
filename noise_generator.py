import torch
import numpy as np
from analytical_solution import exact_solution


# Generate noisy experimental data
def generate_noisy_data(alpha_true=0.1, noise_level=0.1):
    # Create grid of points
    x = torch.linspace(0, 1, 50)
    t = torch.linspace(0, 1, 50)
    X, T = torch.meshgrid(x, t, indexing="ij")

    # Generate exact solution
    U = exact_solution(X, T, alpha_true)

    # Add Gaussian noise
    U_noisy = U + np.random.normal(0, noise_level * np.abs(U), U.shape)

    return x, t, U, U_noisy
