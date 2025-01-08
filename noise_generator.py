import torch
import numpy as np


# Generate noisy data
def generate_noisy_data(
    analytical_solution,
    x_range=(0, 1),
    t_range=(0, 1),
    alpha_true=0.1,
    noise_level=0.1,
    grid_size=50
):
    """
    Generate noisy data based on an analytical PDE solution.

    Args:
        analytical_solution (func): Function that returns the analytic solution
        x_range (tuple): Range of x values for grid (default (0, 1))
        t_range (tuple): Range of t values for grid (default (0, 1))
        alpha_true (float): True alpha value for the solution (default 0.1)
        noise_level (float): Standard deviation of noise relative to solution
        grid_size (int): Number of grid points for both x and t

    Returns:
        x (torch.Tensor): Grid of x values
        t (torch.Tensor): Grid of t values
        U (torch.Tensor or np.ndarray): Exact solution
        U_noisy (torch.Tensor or np.ndarray): The noisy solution
    """
    # Create grid of points separately
    x = torch.linspace(x_range[0], x_range[1], grid_size)
    t = torch.linspace(t_range[0], t_range[1], grid_size)
    X, T = torch.meshgrid(x, t, indexing="ij")

    # Generate exact solution using the analytical solution function
    U = analytical_solution(X, T, alpha_true)

    # If the output is a torch tensor, we use a tensor-based approach
    if isinstance(U, torch.Tensor):
        noise = torch.normal(mean=0.0, std=noise_level * torch.abs(U))
        U_noisy = U + noise
    else:
        # If it's anything else, we assume a numpy ndarray
        noise = np.random.normal(0, noise_level * np.abs(U), U.shape)
        U_noisy = U + noise

    return x, t, U, U_noisy
