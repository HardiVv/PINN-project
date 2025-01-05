import torch
#  from analytical_solution import exact_solution
from models import FCN


def parameter_inversion(params):
    """
    Perform parameter inversion to estimate unknown physical parameters.

    Args:
    - params: Dictionary of parameters loaded from params.yaml.
    """
    # Initialize PINN model
    pinn = FCN(**params["model_params"])

    # Define the domain of x and time as a grid of 50 discretized points
    x_physics = torch.linspace(0, 1, 50).view(-1, 1).requires_grad_(True)
    t_physics = torch.linspace(0, 1, 50).view(-1, 1).requires_grad_(True)
    X, T = torch.meshgrid(x_physics[:, 0], t_physics[:, 0])
    X_physics = torch.cat(
        [X.reshape(-1, 1), T.reshape(-1, 1)], dim=1
    )  # entire domain as tensor

    # Define initial and boundary condition points of model
    x_initial = torch.linspace(0, 1, 50).view(-1, 1)
    t_initial = torch.zeros_like(x_initial)
    initial_points = torch.cat([x_initial, t_initial], dim=1)
    x_boundary = torch.tensor([[0.0], [1.0]])
    t_boundary = torch.linspace(0, 1, 50).view(-1, 1)
    boundary_left = torch.cat(
        [x_boundary[0].repeat(t_boundary.size(0), 1), t_boundary], dim=1
    )
    boundary_right = torch.cat(
        [x_boundary[1].repeat(t_boundary.size(0), 1), t_boundary], dim=1
    )
    boundary = torch.cat([boundary_left, boundary_right], dim=0)

    # Initializes unknown parameter (thermal diffusivity in this case)
    alpha = torch.tensor(0.3, requires_grad=True)  # initial guess

    # Optimizer for the model and parameter
    optimizer = torch.optim.Adam(list(pinn.parameters()) + [alpha], lr=1e-3)

    # Training loop
    for epoch in range(15001):
        optimizer.zero_grad()  # zero gradient for backpropagation

        # Boundary loss, assures respect of boundaries
        u_boundary = pinn(boundary)
        loss_bc = torch.mean(u_boundary**2)

        # Initial condition loss, predicted vs true initial
        u_initial = pinn(initial_points)
        u_initial_true = torch.sin(torch.pi * x_initial)
        loss_ic = torch.mean((u_initial - u_initial_true) ** 2)

        # Physics loss, calculates residual of heat equation
        u = pinn(X_physics)
        u_x = torch.autograd.grad(u, X_physics, torch.ones_like(u),
                                  create_graph=True)[0][:, 0:1]

        u_xx = torch.autograd.grad(
            u_x, X_physics, torch.ones_like(u_x), create_graph=True
        )[0][:, 0:1]

        u_t = torch.autograd.grad(u, X_physics, torch.ones_like(u),
                                  create_graph=True)[0][:, 1:2]

        loss_pde = torch.mean((u_t - alpha * u_xx) ** 2)

        # Total loss
        loss = (
            params["training_params"]["lambda_bc"] * loss_bc
            + params["training_params"]["lambda_ic"] * loss_ic
            + params["training_params"]["lambda_pde"] * loss_pde
        )
        loss.backward()
        optimizer.step()

        # Prints progress of learning
        if epoch % 500 == 0:
            print(
                f"Epoch {epoch}\
                  Loss: {loss.item():.4f}, Alpha: {alpha.item():.4f}"
            )

    # Return the learned parameter
    print(f"Learned alpha: {alpha.item()}")
    return alpha.item()
