import yaml
import torch
import numpy as np

from models import FCN
from training import train_pinn


def load_params():
    """
    Loads model and training params from params.yaml
    """
    with open('params.yaml', 'r') as file:
        return yaml.safe_load(file)


def main():

    params = load_params()

    # Set random seed for reproducibility
    seeds = params['training_params']['random_seeds']
    torch.manual_seed(seeds)
    np.random.seed(seeds)


    # Initialize PINN model
    pinn = FCN(**params['model_params'])

    # Train PINN
    print("Training PINN...")
    train_pinn(pinn, params)

    # TODO @ralferster
    # Perform parameter inversion
    print("Performing parameter inversion...")


if __name__ == "__main__":
    main()
