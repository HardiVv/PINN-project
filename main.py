import yaml
import torch
import numpy as np

from models import FCN
from training import train_pinn
from analytical_solution import heat_eq_1D, heat_eq_1D_with_source


def load_params():
    """
    Loads model and training params from params.yaml
    """
    with open('params.yaml', 'r') as file:
        return yaml.safe_load(file)
    
def get_exact_solution():
    params = load_params()
    solution_type = params['equation']['type']
    if solution_type == "heat_eq_1D":
        return heat_eq_1D, False
    elif solution_type == "heat_eq_1D_with_source":
        return heat_eq_1D_with_source, True
    else:
        raise ValueError("Invalid solution type specified in params.yaml")

def main():

    params = load_params()
    exact_solution_func, with_source = get_exact_solution()

    # Set random seed for reproducibility
    seeds = params['training_params']['random_seeds']
    torch.manual_seed(seeds)
    np.random.seed(seeds)

# Initialize PINN model
    pinn = FCN(**params['model_params'], inversion=False)
    
    # Train PINN with the selected exact solution
    print("=============")
    print("Training PINN")
    print("=============")
    train_pinn(pinn, params,
               exact_solution=exact_solution_func,
               inversion=False,
               with_source=with_source)

    # Parameter inversion
    pinn_inversion = FCN(**params['model_params'], inversion=True)
    print("=============")
    print("PINN inversion")
    print("=============")
    train_pinn(pinn_inversion, params,
               exact_solution=exact_solution_func,
               inversion=True,
               with_source=with_source)

if __name__ == "__main__":
    main()
