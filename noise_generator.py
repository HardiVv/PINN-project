import torch
from analytical_solution import exact_solution


# Generates noisy exp. data with Pytorch not numpy
def generate_noisy_data(alpha_true=0.05, noise_level=0.05, device="cpu"):
    # Create grid of points
    x = torch.linspace(0, 1, 50, device=device)
    t = torch.linspace(0, 1, 50, device=device)
    X, T = torch.meshgrid(x, t, indexing="ij")

    # Generate exact solution
    U = exact_solution(X, T, alpha_true)

    # Add Gaussian noise with PyTorch to keep everything on CPU.
    noise = torch.normal(mean=torch.zeros_like(U),
                         std=noise_level * torch.abs(U))
    U_noisy = U + noise

    return x, t, U, U_noisy
