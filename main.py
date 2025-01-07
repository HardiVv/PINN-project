import yaml
import torch
import numpy as np

from models import FCN
from training import train_pinn
from analytical_solutions import exact_solution, exact_solution_source


def load_params():
    """
    Loads model and training params from params.yaml
    """
    with open("params.yaml", "r") as file:
        return yaml.safe_load(file)


def main():
    params = load_params()

    # Set random seed for reproducibility
    seeds = params["training_params"]["random_seeds"]
    torch.manual_seed(seeds)
    np.random.seed(seeds)

    # Add 'inversion' key to config, which is used later
    params["model_params"]["inversion"] = False  # Initially no inversion

    # Initialize PINN model with the config dictionary
    pinn = FCN(config=params["model_params"])

    # First training phase: using exact_solution (no source term)
    print("Training PINN with exact solution (no source term)...")
    train_pinn(pinn, params, solution_func=exact_solution, inversion=False, with_source=False)

    # Perform parameter inversion (now using the source term in the PDE)
    params["model_params"]["inversion"] = True  # Enables inversion next phase
    pinn_inversion = FCN(config=params["model_params"])

    print("Performing parameter inversion with exact solution source term...")
    train_pinn(pinn_inversion, params, solution_func=exact_solution_source, inversion=True, with_source=True)


if __name__ == "__main__":
    main()
