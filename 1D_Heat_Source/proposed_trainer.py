import torch
import numpy as np
from yaml_loader import load_config
from FCN_Modeller import FCN
from analytic_heat_source import exact_solution_source
from general_noise_generator import generate_noisy_data


def train_pinn(pinn, params, config, inversion=False):
    """
    Trains the PINN model
    Args:
    - pinn: The initialized FCN model.
    - params: A dictionary of training parameters loaded from params.yaml.
    - config: A dictionary of the model's specific configuration.
    - inversion: Flag for inversion training.
    """

    # Visualization and data generation (only used for inversion)
    if inversion:
        x, t, U_exact, U_noisy = generate_noisy_data(
            analytical_solution=exact_solution_source,
            alpha_true=config.get("alpha_true", 0.1),
            noise_level=config.get("noise_level", 0.1),
            grid_size=config.get("grid_size", 50),
        )
        U_noisy_torch = U_noisy.clone().detach().reshape(-1, 1).type(torch.float32)

    # Defines the domain
    x_physics = torch.linspace(0, 1, 50).view(-1, 1).requires_grad_(True)
    t_physics = torch.linspace(0, 1, 50).view(-1, 1).requires_grad_(True)
    X, T = torch.meshgrid(x_physics[:, 0], t_physics[:, 0], indexing="ij")
    X_physics = torch.cat([X.reshape(-1, 1), T.reshape(-1, 1)], dim=1)

    # Defines boundary and initial condition points
    x_boundary = torch.tensor([[0.0], [1.0]])
    t_boundary = torch.linspace(0, 1, 50).view(-1, 1)
    boundary_left = torch.cat([x_boundary[0].repeat(t_boundary.size(0), 1), t_boundary], dim=1)
    boundary_right = torch.cat([x_boundary[1].repeat(t_boundary.size(0), 1), t_boundary], dim=1)
    boundary = torch.cat([boundary_left, boundary_right], dim=0)

    x_initial = torch.linspace(0, 1, 50).view(-1, 1)
    t_initial = torch.zeros_like(x_initial)
    initial_points = torch.cat([x_initial, t_initial], dim=1)

    # Training parameters
    alpha_true = config.get("alpha_true", 0.1)
    lambda_bc = config.get("lambda_bc", 100.0)
    lambda_pde = config.get("lambda_pde", 1.0)
    lambda_ic = config.get("lambda_ic", 100.0)
    lambda_data = config.get("lambda_data", 10.0)
    learning_rate = params.get("learning_rate", 0.001)
    training_epochs = params.get("training_epochs", 5001)

    optimizer = torch.optim.Adam(pinn.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(training_epochs):
        optimizer.zero_grad()

        # Boundary loss
        u_boundary = pinn(boundary)
        loss_bc = torch.mean(u_boundary**2)

        # Initial condition loss
        u_initial = pinn(initial_points)
        u_initial_true = torch.sin(np.pi * x_initial)
        loss_ic = torch.mean((u_initial - u_initial_true) ** 2)

        # Physics loss
        u = pinn(X_physics)
        u_x = torch.autograd.grad(u, X_physics, torch.ones_like(u), create_graph=True)[0][:, 0:1]
        u_xx = torch.autograd.grad(u_x, X_physics, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_t = torch.autograd.grad(u, X_physics, torch.ones_like(u), create_graph=True)[0][:, 1:2]

        if inversion:
            loss_data = torch.mean((u - U_noisy_torch) ** 2)
            loss_pde = torch.mean((u_t - pinn.alpha * u_xx) ** 2)
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

        # Print progress
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    print("Training complete!")


if __name__ == "__main__":
    # Load configuration
    config = load_config("params.yaml")
    model_configs = [config["model_params"], config["model_2_params"]]
    training_params = config["training_params"]

    # Train models
    for i, model_config in enumerate(model_configs, start=1):
        print(f"\nTraining Model {i}...")
        model = FCN(model_config)
        train_pinn(model, training_params, model_config)
