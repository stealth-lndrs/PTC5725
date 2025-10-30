#!/usr/bin/env python3
# Diffusion equation u_t = Cd u_xx on [0,1], u(0,t)=0, u_x(1,t)=0, u(0,x)=sin(pi x/2)
# Chebyshev collocation mapped to [0,1] + manual RK4 with BC enforcement each step.
# Reproduces the report figures and CSV.
