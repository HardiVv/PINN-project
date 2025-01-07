# flake8: noqa: F401  # Ignore the unused import warning

import torch
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from analytical_solutions import exact_solution, exact_solution_source
from noise_generator import generate_noisy_data

def generate_filename(epoch, inversion, solution_func_name, with_source):
    """
    Helper function to save the PINN model training steps depending on sources, and type of equation.

    Args:
    - epoch: The number of epochs
    - inversion: A flag to switch between data inversion (with noisy data) or standard training.
    - solution_func_name: A flag stating which PDE is being trained.
    - with_source: A flag to indicate if the source term has been included or not.

    Returns: png 
    """
    # Generate a unique filename based on the equation and source term
    equation_name = "source" if "source" in solution_func_name else "no_source"
    source_status = "with_source" if with_source else "without_source"
    inversion_status = "inversion" if inversion else "training"

    return f"plots/{inversion_status}/{equation_name}_{source_status}_epoch{epoch}.png"

def train_pinn(pinn, params, solution_func, inversion=False, with_source=False):
    """
    Trains the PINN model.

    Args:
    - pinn: The initialized FCN model.
    - params: A dictionary of parameters loaded from params.yaml.
    - solution_func: The function to compute the exact solution (either exact_solution or exact_solution_source).
    - inversion: A flag to switch between data inversion (with noisy data) or standard training.
    - with_source: A flag to include the source term in the PDE or not.

    Returns:
    - None
    """

    # Load the number of training epochs
    training_epochs = params["training_params"]["training_epochs"]

    # Ensure the directory exists
    os.makedirs("plots/training", exist_ok=True)

    if inversion:
        # Ensure the directory for saving plots exists
        inversion_plot_dir = "plots/inversion"
        os.makedirs(inversion_plot_dir, exist_ok=True)

        # Visualization and data generation with the selected solution function (either with or without source term)
        x, t, U_exact, U_noisy = generate_noisy_data(solution_func)

        # Convert noisy data to torch tensors
        U_noisy_torch = U_noisy.clone().detach().reshape(-1, 1).type(torch.float32)

        # Plotting exact and noisy solutions
        fig2 = plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.contourf(x, t, U_exact, levels=20, cmap="viridis")
        plt.colorbar(label="Exact Solution")
        plt.title("Exact Solution")
        plt.xlabel("x")
        plt.ylabel("t")

        plt.subplot(1, 2, 2)
        plt.contourf(x, t, U_noisy, levels=20, cmap="viridis")
        plt.colorbar(label="Noisy Solution")
        plt.title("Noisy Solution")
        plt.xlabel("x")
        plt.ylabel("t")
        plt.tight_layout()

        filename = generate_filename(0, inversion, solution_func.__name__, with_source)
        plt.savefig(filename)
        plt.close()

        # Training variables
        alpha_estimates = []
        losses = []

    # Define the domain
    x_physics = torch.linspace(0, 1, 50).view(-1, 1).requires_grad_(True)  # Spatial domain
    t_physics = torch.linspace(0, 1, 50).view(-1, 1).requires_grad_(True)  # Time domain
    X, T = torch.meshgrid(x_physics[:, 0], t_physics[:, 0], indexing="ij")
    X_physics = torch.cat([X.reshape(-1, 1), T.reshape(-1, 1)], dim=1)  # (N, 2)

    # Define boundary points
    x_boundary = torch.tensor([[0.0], [1.0]])
    t_boundary = torch.linspace(0, 1, 50).view(-1, 1)
    boundary_left = torch.cat([x_boundary[0].repeat(t_boundary.size(0), 1), t_boundary], dim=1)
    boundary_right = torch.cat([x_boundary[1].repeat(t_boundary.size(0), 1), t_boundary], dim=1)
    boundary = torch.cat([boundary_left, boundary_right], dim=0)

    # Define initial condition points
    x_initial = torch.linspace(0, 1, 50).view(-1, 1)
    t_initial = torch.zeros_like(x_initial)
    initial_points = torch.cat([x_initial, t_initial], dim=1)

    # Hyperparameters
    alpha_true = 0.1  # True value of thermal diffusivity
    lambda_bc = 100.0  # Increased weight for boundary loss
    lambda_pde = 1.0  # Weight for PDE loss
    lambda_ic = 100.0  # Weight for initial condition loss
    lambda_data = 10.0
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(training_epochs):
        optimizer.zero_grad()

        # Boundary loss (enforcing zero at boundaries)
        u_boundary = pinn(boundary)
        loss_bc = torch.mean(u_boundary**2)

        # Initial condition loss
        u_initial = pinn(initial_points)
        u_initial_true = torch.sin(np.pi * x_initial)
        loss_ic = torch.mean((u_initial - u_initial_true) ** 2)

        # Physics loss (PDE) - Now conditionally including the source term
        u = pinn(X_physics)  # Predict u(x, t)

        u_x = torch.autograd.grad(u, X_physics, torch.ones_like(u), create_graph=True)[0][:, 0:1]
        u_xx = torch.autograd.grad(u_x, X_physics, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_t = torch.autograd.grad(u, X_physics, torch.ones_like(u), create_graph=True)[0][:, 1:2]

        if inversion:
            # Data loss comparing with noisy data
            loss_data = torch.mean((u - U_noisy_torch) ** 2)
            if with_source:  # source term is included
                loss_pde = torch.mean((u_t - (getattr(pinn, "alpha", 0.1) * u_xx + 1)) ** 2)  # Source term is 1
            else:
                loss_pde = torch.mean((u_t - getattr(pinn, "alpha", 0.1) * u_xx) ** 2)  # No source term
        else:
            if with_source:  # source term is included
                loss_pde = torch.mean((u_t - alpha_true * u_xx - 1) ** 2)  # Source term is 1
            else:
                loss_pde = torch.mean((u_t - alpha_true * u_xx) ** 2)  # No source term

        # Total loss
        loss = (
            lambda_bc * loss_bc
            + lambda_pde * loss_pde
            + lambda_ic * loss_ic
            + (lambda_data * loss_data if inversion else 0.0)
        )

        loss.backward()
        optimizer.step()

        # Print progress and plot results
        if epoch % 500 == 0:
            if inversion:
                alpha_estimates.append(pinn.alpha.item())
                losses.append(loss.item())
                print(f"Estimated Alpha: {pinn.alpha.item():.4f}")

            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            print(f"Boundary Loss: {loss_bc.item():.4f}")
            print(f"Initial Condition Loss: {loss_ic.item():.4f}")
            print(f"PDE Loss: {loss_pde.item():.4f}")

            with torch.no_grad():
                x_test = torch.linspace(0, 1, 100).view(-1, 1)
                t_test = torch.tensor(0.1).repeat(100).view(-1, 1)
                X_test = torch.cat([x_test, t_test], dim=1)
                u_pred = pinn(X_test).detach()

                # Use the chosen solution function (either exact_solution or exact_solution_source)
                u_exact = solution_func(x_test, t_test, alpha_true)

                plt.figure(figsize=(4.5, 2.5))
                plt.plot(x_test, u_exact, label="Exact Solution", color="gray", linestyle="dashed")
                plt.plot(x_test, u_pred, label="PINN Solution", color="blue")
                plt.legend()
                plt.title(f"Epoch {epoch}")
                plt.tight_layout()
                filename = generate_filename(epoch, inversion, solution_func.__name__, with_source)
                plt.savefig(filename)
                plt.close()

    if inversion:
        # Plot alpha estimates
        plt.figure(figsize=(4, 3))
        plt.plot(np.linspace(0, 5000, len(alpha_estimates)), alpha_estimates, label="Estimated Alpha")
        plt.axhline(y=alpha_true, color="r", linestyle="--", label="True Alpha")
        plt.title("Alpha Estimation")
        plt.xlabel("Epoch")
        plt.ylabel("Alpha")
        plt.legend()
        plt.tight_layout()
        filename = generate_filename(0, inversion, solution_func.__name__, with_source)
        plt.savefig(filename)
        plt.close()
