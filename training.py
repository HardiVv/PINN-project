import torch
import numpy as np
import matplotlib.pyplot as plt

from analytical_solution import exact_solution

def train_pinn(pinn, params):
    """
    Trains the PINN model.
    Args:
    - pinn: The initialized FCN model.
    - params: A dictionary of parameters loaded from params.yaml.
    """

    # Define the domain
    x_physics = torch.linspace(0, 1, 50).view(-1, 1).requires_grad_(True)  # Spatial domain
    t_physics = torch.linspace(0, 1, 50).view(-1, 1).requires_grad_(True)  # Temporal domain
    X, T = torch.meshgrid(x_physics[:, 0], t_physics[:, 0])
    X_physics = torch.cat([X.reshape(-1, 1), T.reshape(-1, 1)], dim=1)  # (N, 2)

    # Define boundary points
    x_boundary = torch.tensor([[0.], [1.]])
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
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)

    # Training
    for epoch in range(5001):
        optimizer.zero_grad()

        # Boundary loss (enforcing zero at boundaries)
        u_boundary = pinn(boundary)
        loss_bc = torch.mean(u_boundary**2)

        # Initial condition loss
        u_initial = pinn(initial_points)
        u_initial_true = torch.sin(np.pi * x_initial)
        loss_ic = torch.mean((u_initial - u_initial_true)**2)

        # Physics loss (PDE)
        u = pinn(X_physics)  # Predict u(x, t)
        u_x = torch.autograd.grad(u, X_physics, torch.ones_like(u), create_graph=True)[0][:, 0:1]
        u_xx = torch.autograd.grad(u_x, X_physics, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_t = torch.autograd.grad(u, X_physics, torch.ones_like(u), create_graph=True)[0][:, 1:2]
        loss_pde = torch.mean((u_t - alpha_true * u_xx)**2)

        # Total loss
        loss = (lambda_bc * loss_bc +
                lambda_pde * loss_pde +
                lambda_ic * loss_ic)
        loss.backward()
        optimizer.step()

        # Print progress and plot results
        if epoch % 500 == 0:
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
                plt.legend()
                plt.title(f"Epoch {epoch}")
                plt.tight_layout()
                plt.savefig(f"plots/heat_eq_epoch{epoch}.png")

