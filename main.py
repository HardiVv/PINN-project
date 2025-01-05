import yaml
import torch
import numpy as np

from models import FCN
from training import train_pinn


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

    # Initialize PINN model
    pinn = FCN(**params["model_params"], inversion=False)

    # Train PINN
    print("Training PINN...")
    train_pinn(pinn, params, inversion=False)

    # Perform parameter inversion
    pinn_inversion = FCN(**params["model_params"], inversion=True)

    print("Performing parameter inversion...")
    train_pinn(pinn_inversion, params, inversion=True)


if __name__ == "__main__":
    main()
