import torch
import torch.nn as nn


class FCN(nn.Module):
    """
    Fully-Connected Neural Network for PINN

    Args:
    - N_INPUT: Number of input features
    - N_OUTPUT: Number of output features
    - N_HIDDEN: Number of neurons in hidden layers
    - N_LAYERS: Number of hidden layers
    """

    def __init__(self, config):
        super().__init__()
        N_INPUT = config.get("N_INPUT", 2)  # default 2 for gen. use
        N_OUTPUT = config.get("N_OUTPUT", 1)  # default 1  for gen. use
        N_HIDDEN = config.get("N_HIDDEN", 32)  # default 32  for gen. use
        N_LAYERS = config.get("N_LAYERS", 3)  # default 3  for gen. use

        self.inversion = config.get("inversion", False)  # default False
        activation = nn.Tanh  # used due to good behaviour

        self.fcs = nn.Sequential(nn.Linear(N_INPUT, N_HIDDEN), activation())
        self.fch = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(N_HIDDEN, N_HIDDEN), activation())
                for _ in range(N_LAYERS - 1)
            ]
        )
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

        if self.inversion:
            # Learnable parameter for alpha (diffusion coefficient), initialed
            self.alpha = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x
