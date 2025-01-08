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

    # Phase 1: Train and invert on no-source PDE
    print("Phase 1: Training and inversion with no source term...")

    # Task 1: Train PINN on no-source PDE
    print("Training PINN (no inversion, no source term)...")
    params["model_params"]["inversion"] = False  # No inversion
    pinn_no_source = FCN(config=params["model_params"])  # Fresh PINN
    train_pinn(pinn_no_source, params, solution_func=exact_solution, inversion=False)

    # Task 2: Perform parameter inversion on the trained PINN (no source term)
    print("Performing parameter inversion (no source term)...")
    params["model_params"]["inversion"] = True  # Enable inversion
    pinn_no_source_inversion = FCN(config=params["model_params"])  # Fresh PINN for inversion
    train_pinn(pinn_no_source_inversion, params, solution_func=exact_solution, inversion=True)

    # Phase 2: Train and invert on source PDE
    print("Phase 2: Training and inversion with source term...")

    # Task 1: Train PINN on source PDE
    print("Training PINN (no inversion, with source term)...")
    params["model_params"]["inversion"] = False  # No inversion
    pinn_with_source = FCN(config=params["model_params"])  # Fresh PINN
    train_pinn(pinn_with_source, params, solution_func=exact_solution_source, inversion=False)

    # Task 2: Perform parameter inversion on the trained PINN (with source term)
    print("Performing parameter inversion (with source term)...")
    params["model_params"]["inversion"] = True  # Enable inversion
    pinn_with_source_inversion = FCN(config=params["model_params"])  # Fresh PINN for inversion
    train_pinn(pinn_with_source_inversion, params, solution_func=exact_solution_source, inversion=True)


if __name__ == "__main__":
    main()
