import torch
import numpy as np
import matplotlib.pyplot as plt
from noise_generator import generate_noisy_data


def train_inversion(pinn, params, exact_solution, inversion=False, with_source=False):
    """
    Trains the PINN model and performes inversion if True

    Args:
    - pinn: The initialized FCN model.
    - params: A dictionary of parameters loaded from params.yaml
    - exact_solution: Analytical solution of
      1D heat equation or 1D heat equation with source term
    - inversion (boolean): Determines if to perform inversion
    - with_source (boolean): Checks if soulution has source

    Returns:
    - Plots
    """

    if inversion:
        # Visualization and data generation
        x, t, U_exact, U_noisy = generate_noisy_data(equation=exact_solution)
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

        if with_source:
            plt.suptitle("1D heat eq + source")
            plt.savefig("plots/source/noisydata.png")
        else:
            plt.suptitle("1D heat eq")
            plt.savefig("plots/nosource/noisydata.png")

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

    # Retrieve hyperparameters from the params dictionary
    alpha_true = params["hyperparameters"]["alpha_true"]
    lambda_bc = params["hyperparameters"]["lambda_bc"]
    lambda_pde = params["hyperparameters"]["lambda_pde"]
    lambda_ic = params["hyperparameters"]["lambda_ic"]
    lambda_data = params["hyperparameters"]["lambda_data"]
    optimizer = torch.optim.AdamW(pinn.parameters(), lr=1e-3)  # Adam or AdamW

    # Training
    for epoch in range(5001):
        optimizer.zero_grad()

        # Boundary loss (enforcing zero at boundaries)
        u_boundary = pinn(boundary)
        loss_bc = torch.mean(u_boundary**2)

        # Initial condition loss
        u_initial = pinn(initial_points)
        u_initial_true = torch.sin(np.pi * x_initial)
        loss_ic = torch.mean((u_initial - u_initial_true) ** 2)

        # Physics loss (PDE)
        u = pinn(X_physics)  # Predict u(x, t)

        u_x = torch.autograd.grad(u, X_physics, torch.ones_like(u), create_graph=True)[0][:, 0:1]
        u_xx = torch.autograd.grad(u_x, X_physics, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_t = torch.autograd.grad(u, X_physics, torch.ones_like(u), create_graph=True)[0][:, 1:2]

        # Apply source term if the equation has a source term
        if inversion:
            loss_data = torch.mean((u - U_noisy_torch) ** 2)
            if with_source:
                loss_pde = torch.mean((u_t - pinn.alpha * u_xx - 1) ** 2)  # subtract source term -1
            else:  # Data loss comparing with noisy data
                loss_pde = torch.mean((u_t - pinn.alpha * u_xx) ** 2)
        else:
            if with_source:
                loss_pde = torch.mean((u_t - alpha_true * u_xx - 1) ** 2)  # subtract source term -1
            else:
                loss_pde = torch.mean((u_t - alpha_true * u_xx) ** 2)

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
                u_exact = exact_solution(x_test, t_test, alpha_true)

                plt.figure(figsize=(4.5, 2.5))
                plt.plot(x_test, u_exact, label="Exact Solution", color="gray", linestyle="dashed")
                plt.plot(x_test, u_pred, label="PINN Solution", color="blue")
                plt.grid()
                plt.legend()

                if with_source:
                    if inversion:
                        plt.title(f"Epoch {epoch}, inversion: 1D heat eq + source")
                        plt.tight_layout()
                        plt.savefig(f"plots/source/inversion/heat_eq_inv_source_epoch{epoch}.png")
                    else:
                        plt.title(f"Epoch {epoch}, training: 1D heat eq + source")
                        plt.tight_layout()
                        plt.savefig(f"plots/source/training/heat_eq_source_epoch{epoch}.png")
                else:
                    if inversion:
                        plt.title(f"Epoch {epoch}, inversion: 1D heat eq")
                        plt.tight_layout()
                        plt.savefig(f"plots/nosource/inversion/heat_eq_inv_epoch{epoch}.png")
                    else:
                        plt.title(f"Epoch {epoch}, training: 1D heat eq")
                        plt.tight_layout()
                        plt.savefig(f"plots/nosource/training/heat_eq_train_epoch{epoch}.png")

                plt.close()

    if inversion:
        # Plot alpha estimates
        plt.figure(figsize=(4, 3))
        plt.plot(np.linspace(0, 5000, len(alpha_estimates)), alpha_estimates, label="Estimated Alpha")
        plt.axhline(y=alpha_true, color="r", linestyle="--", label="True Alpha")
        plt.xlabel("Epoch")
        plt.ylabel("Alpha")
        plt.grid()
        plt.legend()

        if with_source:
            plt.title("α estimation: 1D heat eq + source")
            plt.tight_layout()
            plt.savefig("plots/source/alpha_estimate.png")
        else:
            plt.title("α Estimation: 1D heat eq")
            plt.tight_layout()
            plt.savefig("plots/nosource/alpha_estimate.png")

        plt.close()
