# Project 3: Task 1
# Created by: Dea Lana Asri - 277575
# dl.asri@stud.uis.no
# Date: 21.03.2025

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define spatial grid and time points
L = 4  # Length of the domain
nx = 1000  # Number of spatial points
dx = L / nx  # Spatial step
spatial_grid = np.linspace(0, L, nx)  # Spatial grid
T = 10  # Final time
nt = 4000  # Number of time steps
dt = T / nt  # Time step
time_steps = np.linspace(0, T, nt)  # Time grid

# Flux function
def flux(u):
    return u * (1 - u)

# Lax-Friedrichs numerical flux
def lax_friedrichs_flux(u_left, u_right):
    return 0.5 * (flux(u_left) + flux(u_right)) - 0.5 * (dx / dt) * (u_right - u_left)

# Initial conditions
def u0(x):
    return np.select(
        [
            (x >= 0) & (x <= 0.5),
            (x > 0.5) & (x <= 1.5),
            (x > 1.5) & (x <= 2.5),
            (x > 2.5) & (x <= 3.5),
            (x > 3.5) & (x <= 4)
        ],
        [0.2, 0.4, 0.6, 0.7, 0.4],
        default=np.nan
    )

# Generate Random k(x) function
def k(spatial_grid):
    np.random.seed(1234)  # for reproducibility
    ki = np.random.normal(1, 0.25, size=9)  # Generate 9 random values

    # Add boundary conditions 
    ki = np.insert(ki, 0, 1)  # Insert 1 at the beginning
    ki = np.append(ki, 1)  # Append 1 at the end

    # Linear interpolation to match the spatial grid
    kx = np.interp(spatial_grid, np.linspace(0, 4, len(ki)), ki)
    
    return kx

# Set up the initial condition
u_initial = u0(spatial_grid)
u_old = np.copy(u_initial)
u = np.zeros(nx)
u_numerical = [np.copy(u_initial)]
kx = list(k(spatial_grid))



# Numerical solution
for n in range(1, nt):
    # Compute all fluxes first
    F = np.zeros(nx + 1)
    for i in range(nx + 1):
        u_left = u_old[i-1] if i > 0 else u_old[0]
        u_right = u_old[i] if i < nx else u_old[-1]
        F[i] = lax_friedrichs_flux(u_left, u_right)

    # Update solution
    for i in range(1, nx-1):
        u[i] = u_old[i] - (dt / dx) * (kx[i+1] * F[i+1] - kx[i] * F[i])
    
    # Boundary conditions
    u[0] = u[1]  
    u[-1] = u[-2] 
    
    # Update u_old and save the result
    u_old = np.copy(u)
    u_numerical.append(np.copy(u))

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(0, 4)
ax.set_ylim(0, 2)
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
    #u_exact = uxt(spatial_grid, t)  # Compute solution u(x,t)
    #line.set_data(spatial_grid, u_exact)  # Update the solution curve
    line2.set_data(spatial_grid, u_numerical[frame])  # Update the solution curve
    time_text.set_text(f"t = {t:.2f}")  # Update time label
    return line, line2, time_text

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(time_steps), interval=50, blit=True)

# Show the animation
plt.legend()
plt.show()