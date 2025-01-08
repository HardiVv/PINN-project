# Making a PINN (physics informed neural network)

Use pytorch to solve 1D heat equation

$$u_t(x,t)=\alpha u_{xx}(x,t) \text{ where } x \in \mathbb{R} \text{ and }  t>0$$

$u(x,t)$ is the temperature at position $x$ and time $t$.<br>
$\alpha$ is the thermal diffusivity. <br>

For simplicity, let's consider the initial condition
$$u(x,0) = \sin ⁡(\pi x)$$
and boundary conditions
$$u(0,t)=u(1,t)=0$$

Then the analytical solution is:
$$u(x,t)=e^{−\pi^2 \alpha t} \sin⁡(\pi x)$$

