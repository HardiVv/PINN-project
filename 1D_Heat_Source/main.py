import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt

from FCN_Modeller import FCN
from proposed_trainer import train_pinn
from analytic_heat_source import exact_solution_source
from general_noise_generator import generate_noisy_data
from yaml_loader import load_config


def load_params():
    """Load model and training parameters from param.yaml"""
    with open("params.yaml", "r") as file:
        return yaml.safe_load(file)


def main():
    # Load parameters from config files
    params = load_params()
    model_config = params["model_params"]
    training_config = params["training_params"]

    # Set random seed for reproducibility
    seed = training_config.get("random_seeds", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate noisy data if inversion is set to True
    if training_config.get("inversion", False):
        print("Generating noisy data for parameter inversion...")
        x, t, U_exact, U_noisy = generate_noisy_data(
            analytical_solution=exact_solution_source,
            alpha_true=model_config.get("alpha_true", 0.1),
            noise_level=model_config.get("noise_level", 0.1),
            grid_size=50,
        )
    else:
        print("Skipping noisy data generation...")

    # Initialize the PINN model
    pinn = FCN(config=model_config)

    # Train the model
    print("Training the PINN model...")
    train_pinn(pinn, training_config, model_config, inversion=training_config.get("inversion", False))

    # Perform any post-training visualization
    print("Visualizing the analytical solution...")
    x_vals = torch.linspace(0, 1, 50)
    t_vals = torch.linspace(0, 1, 50)
    X, T = torch.meshgrid(x_vals, t_vals, indexing="ij")
    U_analytic = exact_solution_source(X, T, alpha=0.1).numpy()

    # Plotting the analytical solution
    plt.figure(figsize=(10, 5))
    cp = plt.contourf(X.numpy(), T.numpy(), U_analytic, levels=20, cmap="viridis")
    plt.colorbar(cp, label="Temperature")
    plt.title("Analytical Solution of the 1D Heat Equation with Source")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.tight_layout()
    plt.show()

    print("Main execution completed!")


if __name__ == "__main__":
    main()
