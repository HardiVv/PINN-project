import yaml
import torch
import numpy as np

from models import FCN
from train_and_inversion import train_inversion
from analytical_solution import heat_eq_1D, heat_eq_1D_with_source


def load_params():
    """
    Loads model and training params from params.yaml
    """
    with open("params.yaml", "r") as file:
        return yaml.safe_load(file)


def get_exact_solution():
    """
    Gets equation type and sets with_source value
    """
    params = load_params()
    solution_type = params["equation"]["type"]
    if solution_type == "heat_eq_1D":
        return heat_eq_1D, False
    elif solution_type == "heat_eq_1D_with_source":
        return heat_eq_1D_with_source, True
    else:
        raise ValueError("Invalid solution type.")


def main():
    """
    Trains pinn and does the parameter inversion
    """
    # Load parameters and equation type
    params = load_params()
    exact_solution_func, with_source = get_exact_solution()

    # Set random seed for reproducibility
    seeds = params["training_params"]["random_seeds"]
    torch.manual_seed(seeds)
    np.random.seed(seeds)

    # Initialize PINN model
    pinn = FCN(**params["model_params"], inversion=False)

    # Train PINN
    print("==============================")
    print("\tTraining PINN")
    print("==============================")
    train_inversion(pinn, params, exact_solution=exact_solution_func, inversion=False, with_source=with_source)

    # Parameter inversion
    pinn_inversion = FCN(**params["model_params"], inversion=True)
    print("==============================")
    print("\tPINN inversion")
    print("==============================")
    train_inversion(pinn_inversion, params, exact_solution=exact_solution_func, inversion=True, with_source=with_source)


if __name__ == "__main__":
    main()
