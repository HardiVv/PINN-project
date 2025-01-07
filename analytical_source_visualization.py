import torch
import matplotlib.pyplot as plt

from analytical_solutions import exact_solution_source

# Parameters
alpha = 0.1

# Define grid
x = torch.linspace(0, 1, 50)
t = torch.linspace(0, 1, 50)
X, T = torch.meshgrid(x, t, indexing="ij")  # Create 2D grid for x and t

# Analytical solution
u = exact_solution_source(X, T, alpha).numpy()

assert u.shape == X.shape

# Plotting the solution
fig, ax = plt.subplots(figsize=(10, 5))
cp = ax.contourf(X, T, u, levels=20, cmap="viridis")
plt.colorbar(cp, label="Temperature")
ax.set_title("Analytical Solution of 1D Heat Equation with Source")
ax.set_xlabel("x")
ax.set_ylabel("t")
plt.tight_layout()
plt.show()
