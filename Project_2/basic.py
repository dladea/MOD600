# Basic Problem for Project 2: Modelling Traffic Flow
# Created by: Dea Lana Asri - 277575
# dl.asri@stud.uis.no
# Date: 20.02.2025

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define spatial grid and time points
L = 1.5  # Length of the domain
nx = 100  # Number of spatial points
dx = L / nx  # Spatial step
spatial_grid = np.linspace(0, L, nx)  # Spatial grid
T = 2  # Final time
nt = 100  # Number of time steps
dt = T / nt  # Time step
time_steps = np.linspace(0, T, nt)  # Time grid

# Flux function
def flux(u):
    return 0.25 * u**2  

# Initial conditions
def u0(x):
    return np.where(x <= 1/2, 2*x, 
           np.where((x > 1/2) & (x <= 1), 1, 
           np.where((x > 1) & (x <= 3/2), 3 - 2*x, np.nan)))

# Exact solution using method of characteristics
def uxt(x, t):
    return np.where(x <= (1+t)/2, 2 * (x / (1+t)),
           np.where((x > (1+t)/2) & (x <= (2+t)/2), 1,
           np.where((x > (1+t)/2) & (x <= 3/2), 3 - 2 * (x - 1.5 * t) / (1 - t), np.nan)))
    

# Set up the initial condition
u_initial = u0(spatial_grid)
u_old = np.copy(u_initial)
u = np.zeros(nx)
u_numerical = [np.copy(u_initial)]

# Numerical solution
for n in range(1, nt):
    # Compute the flux
    f = flux(u_old)
    u[1:-1] = u_old[1:-1] - (dt / dx) * (f[1:-1] - f[:-2])
    # Boundary Condition
    u[0] = u[1]
    u[-1] = u[-2]
    # Update u_old
    u_old = np.copy(u)
    u_numerical.append(np.copy(u))

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(0, 2)
ax.set_ylim(0, 1.5)
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
ax.set_title("Modelling Traffic Flow: Basic Problem")
ax.grid()

# Plot the initial condition (always present)
ax.plot(spatial_grid, u_initial, 'b--', label='Initial Condition')

# Create an empty line for the evolving solution
line, = ax.plot([], [], 'r-', label='Exact Solution')
line2, = ax.plot([], [], 'g-', label='Numerical Solution')
time_text = ax.text(1.0, 1.4, '', fontsize=10)

# Function to update the animation
def update(frame):
    t = time_steps[frame]
    u_exact = uxt(spatial_grid, t)  # Compute solution u(x,t)
    line.set_data(spatial_grid, u_exact)  # Update the solution curve
    line2.set_data(spatial_grid, u_numerical[frame])  # Update the solution curve
    time_text.set_text(f"t = {t:.2f}")  # Update time label
    return line, line2, time_text

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(time_steps), interval=50, blit=True)

# Show the animation
plt.legend()
plt.show()
