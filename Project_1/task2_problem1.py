import numpy as np
import matplotlib.pyplot as plt

# Linear Problem 1

T = 0.5
M = 500 # grid blocks
time_steps = 500
dt = T/time_steps
Ldom = 2
dx = Ldom/M
a = 1
x = np.linspace(dx * 0.5, Ldom + 0.5 * dx - dx, M)

# Stability Check
def CFL(a, dt, dx):
    return a * dt/dx

# define initial condition
def u0_func(x):
    v = np.zeros_like(x)
    L = np.logical_and(x >= 0.8, x <= 1.2)
    v[L] = 1.0
    return v

u0 = u0_func(x)

# Calculate the numerical Solution
u_old = np.copy(u0)
u = np.zeros(M)

for i in range(time_steps):
    for j in range(0, M-1):
        u[j] = u_old[j] - a * dt/dx * (u_old[j] - u_old[j-1])
    # Set the boundary conditions
    u[0] = u[1]
    u[-1] = u[-2]
    u_old = np.copy(u)
    
# Calculate the analytical Solution
u_analytical = np.zeros(M)

# new argument
y = x - a * T

# find index where y stays within domain [0,2]
L1 = np.logical_and(y >= 0,  y <= Ldom)

# u(x,t) = u0(x-a*T)
u_analytical[L1] = u0_func(y[L1])

plt.figure()
plt.plot(x, u0, '-r', label='Initial Condition')
plt.plot(x, u, 'ob', label='Numerical Solution')
plt.plot(x, u_analytical, '-g', label='Analytical Solution at T=0.5')
plt.axis([0, Ldom, -0.1, 1.1])
plt.xlabel('x-axis')
plt.ylabel('u_0(x)')
plt.title('Solution of Linear Advection 1')
plt.legend()
plt.grid()
plt.show()
