# Investigations of a Traffic Flow Model: Initial Condition (1)
# Created by: Dea Lana Asri - 277575
# dl.asri@stud.uis.no
# Date: 09.03.2025

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define spatial grid and time points
L = 4  # Length of the domain
nx = 1000  # Number of spatial points
dx = L / nx  # Spatial step
spatial_grid = np.linspace(-0.5, L-0.5, nx)  # Spatial grid
T = 10  # Final time
nt = 2000  # Number of time steps
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
            (x >= -0.5) & (x <= 0),
            (x > 0) & (x <= 0.5),
            (x > 0.5) & (x <= 1.5),
            (x > 1.5) & (x <= 2.5),
            (x > 2.5) & (x <= 3.5)
        ],
        [0.2, 0.4, 0.6, 0.8, 0.9],
        default=np.nan
    )

# # Exact solution using Similarity and Shock Solution
# def uxt(x, t):
#     return np.where(x <= t/2, 0,
#            np.where((x > t/2) & (x <= 1-t), 3/4,
#            np.where((x > 1-t) & (x < 1 + 2*t), (1+2*t-x)/(4*t),
#            np.where(x >= 1 + 2*t, 0, np.nan))))
    

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
fig, ax = plt.subplots(1, 4, figsize=(20, 5), tight_layout=True)
fig.suptitle('Modelling Traffic Flow: Initial Condition 1')


ax[0].set_xlabel("u")
ax[0].set_ylabel("f(u)")
ax[0].set_title("Flux Function")

ax[1].set_xlim(-0.5, 3.5)
ax[1].set_ylim(-0.5, 1.5)
ax[1].set_xlabel("x")
ax[1].set_ylabel("u(x,t)")
ax[1].set_title("Solution u(x,t)")

ax[2].set_xlim(-0.5, 3.5)
ax[2].set_ylim(-0.25, 1)
ax[2].set_xlabel("x")
ax[2].set_ylabel("f(u(x))")
ax[2].set_title("Flux F(u(x,t))")

ax[3].set_xlim(0, T)
ax[3].set_ylim(-0.25, 1)
ax[3].set_xlabel("time")
ax[3].set_ylabel("fu(x*,t)")
ax[3].set_title("Flux at x* = 2")

# Plot the initial condition
u_grid = np.linspace(0, 1, 50)
ax[0].plot(u_grid, flux(u_grid), 'r-', label='f(u) = u(1 - u)')
ax[1].plot(spatial_grid, u_initial, 'b--', label='Initial Condition')
ax[2].plot(spatial_grid, flux(u_initial), 'b--', label='Initial Condition')

# Create an empty line for the evolving solution
#line, = ax.plot([], [], 'r-', label='Exact Solution')
line2, = ax[1].plot([], [], 'g-', label='Numerical Solution, u(x,t)')
line3, = ax[2].plot([], [], 'g-', label='f(u(x,t))')
line4, = ax[3].plot([], [], 'c-', label='Flux at x* = 2')
#time_text = fig.text(0.5, 0.02, '', ha='center', va='bottom', fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
time_text = ax[1].text(1.0, 1.4, '', fontsize=10)

for i in range(len(ax)):
    ax[i].grid()
    ax[i].legend()

# Initialize list for ax4
flux_at_x = []
time_list = []

# Function to update the animation
def update(frame):
    global flux_at_x, time_list

    # Reset data at the start of the loop
    if frame == 0:
        flux_at_x = []
        time_list = []
        line4.set_data([], [])  # Clear line4

    t = time_steps[frame]
    #u_exact = uxt(spatial_grid, t)  # Compute solution u(x,t)
    #line.set_data(spatial_grid, u_exact)  # Update the solution curve
    line2.set_data(spatial_grid, u_numerical[frame])  
    line3.set_data(spatial_grid, flux(u_numerical[frame]))

    # Calculate flux at x* = 2
    time_list.append(t)
    flux_list = flux(u_numerical[frame])[int((2+0.5)/dx)]
    flux_at_x.append(flux_list)
    line4.set_data(time_list, flux_at_x)
    time_text.set_text(f"t = {t:.2f}")  # Update time label
    return line2, line3, line4, time_text

# # Function to save figures with 4 subplots at specific times
# def save_figure_at_time(t_target, u_numerical, time_steps, spatial_grid):
#     # Find the frame index closest to the target time
#     frame_index = np.argmin(np.abs(time_steps - t_target))
    
#     # Extract the numerical solution at the target time
#     u_at_t = u_numerical[frame_index]
    
#     # Create a figure with 4 subplots
#     fig, ax = plt.subplots(1, 4, figsize=(20, 5))
#     fig.suptitle(f'Modelling Traffic Flow: Initial Condition 1 at t = {t_target:.1f}')
    
#     # Subplot 1: Flux function
#     u_grid = np.linspace(0, 1, 50)
#     ax[0].plot(u_grid, flux(u_grid), 'r-', label='f(u) = u(1 - u)')
#     ax[0].set_xlabel("u")
#     ax[0].set_ylabel("f(u)")
#     ax[0].set_title("Flux Function")
#     ax[0].grid()
#     ax[0].legend()
    
#     # Subplot 2: Numerical solution u(x, t)
#     ax[1].plot(spatial_grid, u_at_t, 'g-', label='Numerical Solution, u(x,t)')
#     ax[1].plot(spatial_grid, u_initial, 'b--', label='Initial Condition')
#     ax[1].set_xlim(-0.5, 3.5)
#     ax[1].set_ylim(-0.5, 1.5)
#     ax[1].set_xlabel("x")
#     ax[1].set_ylabel("u(x,t)")
#     ax[1].set_title("Solution u(x,t)")
#     ax[1].grid()
#     ax[1].legend()
    
#     # Subplot 3: Flux of the solution f(u(x, t))
#     ax[2].plot(spatial_grid, flux(u_at_t), 'g-', label='f(u(x,t))')
#     ax[2].plot(spatial_grid, flux(u_initial), 'b--', label='Initial Condition')
#     ax[2].set_xlim(-0.5, 3.5)
#     ax[2].set_ylim(-0.25, 1)
#     ax[2].set_xlabel("x")
#     ax[2].set_ylabel("f(u(x,t))")
#     ax[2].set_title("Flux F(u(x,t))")
#     ax[2].grid()
#     ax[2].legend()
    
#     # Subplot 4: Flux at x* = 2 over time
#     time_list = time_steps[:frame_index + 1]
#     flux_at_x = [flux(u_numerical[i])[int((2 + 0.5) / dx)] for i in range(frame_index + 1)]
#     ax[3].plot(time_list, flux_at_x, 'c-', label='Flux at x* = 2')
#     ax[3].set_xlim(0, T)
#     ax[3].set_ylim(0, 1)
#     ax[3].set_xlabel("time")
#     ax[3].set_ylabel("f(u(x*,t))")
#     ax[3].set_title("Flux at x* = 2")
#     ax[3].grid()
#     ax[3].legend()
    
#     # Save the figure
#     plt.tight_layout()
#     plt.savefig(f'traffic_flow_1_t_{t_target:.0f}.png')
#     plt.close()

# # Save figures at T = 0, T = 5, T = 10
# save_figure_at_time(0, u_numerical, time_steps, spatial_grid)
# save_figure_at_time(5, u_numerical, time_steps, spatial_grid)
# save_figure_at_time(10, u_numerical, time_steps, spatial_grid)

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(time_steps), interval=25, blit=True)

# Show the animation
plt.legend()
plt.show()