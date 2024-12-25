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
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN),
            activation())
        self.fch = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation())
            for _ in range(N_LAYERS - 1)]
        )
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x
