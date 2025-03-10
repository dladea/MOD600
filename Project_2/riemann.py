# Riemann Problem for Project 2: Modelling Traffic Flow
# Created by: Dea Lana Asri - 277575
# dl.asri@stud.uis.no
# Date: 09.03.2025

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define spatial grid and time points
L = 5  # Length of the domain
nx = 1000  # Number of spatial points
dx = L / nx  # Spatial step
spatial_grid = np.linspace(-1, L-1, nx)  # Spatial grid
T = 2/3  # Final time
nt = 500  # Number of time steps
dt = T / nt  # Time step
time_steps = np.linspace(0, T, nt)  # Time grid

# Flux function
def flux(u):
    return 2 * u * (1 - u)

# Lax-Friedrichs numerical flux
def lax_friedrichs_flux(u_left, u_right):
    return 0.5 * (flux(u_left) + flux(u_right)) - 0.5 * (dx / dt) * (u_right - u_left)

# Initial conditions
def u0(x):
    return np.where(x < 0, 0, 
           np.where((x >= 0) & (x <= 1), 3/4, 
           np.where((x > 1), 0, np.nan)))

# Exact solution using Similarity and Shock Solution
def uxt(x, t):
    return np.where(x <= t/2, 0,
           np.where((x > t/2) & (x <= 1-t), 3/4,
           np.where((x > 1-t) & (x < 1 + 2*t), (1+2*t-x)/(4*t),
           np.where(x >= 1 + 2*t, 0, np.nan))))
    

# Set up the initial condition
u_initial = u0(spatial_grid)
u_old = np.copy(u_initial)
u = np.zeros(nx)
u_numerical = [np.copy(u_initial)]


for n in range(1, nt):
    # Compute all fluxes first
    F = np.zeros(nx + 1)
    for i in range(nx + 1):
        u_left = u_old[i-1] if i > 0 else u_old[0]
        u_right = u_old[i] if i < nx else u_old[-1]
        F[i] = lax_friedrichs_flux(u_left, u_right)
    
    # Update solution
    for i in range(nx):
        u[i] = u_old[i] - (dt / dx) * (F[i+1] - F[i])
    
    # Boundary conditions
    u[0] = u[1]
    u[-1] = u[-2]
    
    # Update u_old and save the result
    u_old = np.copy(u)
    u_numerical.append(np.copy(u))  

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-1, 4)
ax.set_ylim(-0.5, 1.5)
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
ax.set_title("Modelling Traffic Flow: Riemann Problem")
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
ani = animation.FuncAnimation(fig, update, frames=len(time_steps), interval=25, blit=True)

# Save final figure
# Save the final snapshot (t = T)
plt.figure()
plt.xlim(-1, 4)
plt.ylim(-0.5, 1.5)
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title(f"Riemann Problem t = {T:.2f}")
plt.grid()

# Plot numerical and exact solutions at the final time
plt.plot(spatial_grid, u_initial, 'b--', label='Initial Condition')
plt.plot(spatial_grid, u_numerical[-1], 'g-', label='Numerical Solution')
u_exact_final = uxt(spatial_grid, T)  
plt.plot(spatial_grid, u_exact_final, 'r-', label='Exact Solution')

plt.legend()
plt.savefig('Riemann23.png')  # Save as PNG
plt.show()

# Show the animation
plt.legend()
plt.show()
