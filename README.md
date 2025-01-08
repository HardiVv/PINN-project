# Making a PINN (physics informed neural network)

Use pytorch to solve 1D heat equation with and without a source term.

$$u_t(x,t)=\alpha u_{xx}(x,t) \text{ where } x \in \mathbb{R} \text{ and }  t>0$$

$u(x,t)$ is the temperature at position $x$ and time $t$.<br>
$\alpha$ is the thermal diffusivity. <br>

For simplicity, let's consider
$$u(x,0) = \sin ⁡(\pi x)$$ as the initial condition
and 
$$u(0,t)=u(1,t)=0$$ boundary conditions

Then the analytical solution is:
$$u(x,t)=e^{−\pi^2 \alpha t} \sin⁡(\pi x)$$

Adding a source term $f(x,t)=1$ changes the equation to the following form

$$u_t(x,t)=\alpha u_{xx}(x,t) + 1 \text{ where } x \in \mathbb{R} \text{ and }  t>0$$

For simplicity, we considered
$$u(x,0) = \sin ⁡(\pi x)$$ the same initial condition
and
$$u(0,t)=u(1,t)=0$$ boundary conditions

Then the analytical solution is:
$$\exp\left(-\pi^2 \alpha t\right) \sin\left(\pi x\right) + \left(1 - \exp(-t)\right)\left(x - x^2\right)$$