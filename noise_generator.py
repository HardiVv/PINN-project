import torch
import numpy as np


# Generate experimental data with Gaussian noise 
def generate_noisy_data(alpha_true=0.1, noise_level=0.05, equation=None):

    # Grid points
    x = torch.linspace(0, 1, 50)
    t = torch.linspace(0, 1, 50)
    X, T = torch.meshgrid(x, t, indexing="ij")

    # Generate exact solution
    U = equation(X, T, alpha_true)
    
    # Add Gaussian noise
    U_noisy = U + np.random.normal(0, noise_level * np.abs(U), U.shape)

    return x, t, U, U_noisy
