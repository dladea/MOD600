# Project 3: Task 1
# Created by: Dea Lana Asri - 277575
# dl.asri@stud.uis.no
# Date: 21.03.2025

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.io 

# Define spatial grid and time points
L = 4  # Length of the domain
nx = 600  # Number of spatial points
dx = L / nx  # Spatial step
spatial_grid = np.linspace(0, L, nx)  # Spatial grid
T = 10  # Final time
nt = 4000  # Number of time steps
dt = T / nt  # Time step
time_steps = np.linspace(0, T, nt)  # Time grid

# Flux function
def flux(u):
    return u * (1 - u)

def dev_flux(u):
    return 1 - 2 * u

# Rusanov numerical flux
def rusanov(u_left, u_right):
    return 0.5 * (flux(u_left) + flux(u_right)) - 0.5 * (max(abs(dev_flux(u_left)), abs(dev_flux(u_right)))) * (u_right - u_left)

# CFL condition, put max kx = 1.75
def cfl(dev, dx, dt):
    return max(dev) * 1.75 * dt / dx

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
dev = [dev_flux(u) for u in u_initial]

# Check CFL condition
cfl_condition = cfl(dev, dx, dt)
if cfl_condition > 1:
    print("CFL condition is not satisfied")
    exit()
else:
    print("CFL condition is satisfied")

# Numerical solution
for n in range(1, nt):
    # Compute all fluxes first
    F = np.zeros(nx + 1)
    for i in range(nx + 1):
        u_left = u_old[i-1] if i > 0 else u_old[0]
        u_right = u_old[i] if i < nx else u_old[-1]
        F[i] = rusanov(u_left, u_right)

    # Update solution
    for i in range(1, nx-1):
        u[i] = u_old[i] - (dt / dx) * ((kx[i+1]+kx[i]) / 2 * F[i+1] - (kx[i]+kx[i-1]) / 2 * F[i])
    
    # Boundary conditions
    u[0] = u[1]  
    u[-1] = u[-2] 
    
    # Update u_old and save the result
    u_old = np.copy(u)
    u_numerical.append(np.copy(u))

# Set up the figure and axis
fig, ax = plt.subplots(1, 4, figsize=(20, 5), tight_layout=True)
fig.suptitle('Traffic Flow Simulation: Random generated k(x)')


ax[0].set_xlabel("u")
ax[0].set_ylabel("f(u)")
ax[0].set_title("Flux Function")

ax[1].set_xlim(0, 4)
ax[1].set_ylim(-0.25, 1)
ax[1].set_xlabel("x")
ax[1].set_ylabel("u(x,t)")
ax[1].set_title("Initial Distribution U0(x) and Solution u(x,t)")

ax[2].set_xlim(0, 4)
ax[2].set_ylim(0, 2)
ax[2].set_xlabel("x")
ax[2].set_ylabel("k(x)")
ax[2].set_title("k(x)")

ax[3].set_xlim(0, T)
ax[3].set_ylim(-0.25, 1)
ax[3].set_xlabel("time")
ax[3].set_ylabel("fu(x*,t)")
ax[3].set_title("Observation of u(Xi,t)")


# Plot the initial condition
u_grid = np.linspace(0, 1, 50)
ax[0].plot(u_grid, flux(u_grid), 'r-', label='f(u) = u(1 - u)')
ax[1].plot(spatial_grid, u_initial, 'b--', label='Initial distribution')
ax[2].plot(spatial_grid, kx, 'b--', label='kx')

# Create an empty line for the evolving solution
line2, = ax[1].plot([], [], 'g-', label='Numerical Solution, u(x,t)')
line3, = ax[3].plot([], [], 'r-', label='u(x,t) at x1')
line4, = ax[3].plot([], [], 'g-', label='u(x,t) at x2')
line5, = ax[3].plot([], [], 'b-', label='u(x,t) at x3')
line6, = ax[3].plot([], [], 'c-', label='u(x,t) at x4')
time_text = ax[1].text(1.0, 0.9, '', fontsize=10)

for i in range(len(ax)):
    ax[i].grid()
    ax[i].legend()

# Initialize list for observation spot
u_x1 = []
u_x2 = []
u_x3 = []
u_x4 = []
time_list = []

# Function to update the animation
def update(frame):
    global u_x1, u_x2, u_x3, u_x4, time_list

    # Reset data at the start of the loop
    if frame == 0:
        u_x1 = []
        u_x2 = []
        u_x3 = []
        u_x4 = []
        time_list = []
        line3.set_data([], [])
        line4.set_data([], [])
        line5.set_data([], [])
        line6.set_data([], []) 

    t = time_steps[frame]
    #u_exact = uxt(spatial_grid, t)  # Compute solution u(x,t)
    #line.set_data(spatial_grid, u_exact)  # Update the solution curve
    line2.set_data(spatial_grid, u_numerical[frame])  # Update the solution curve

    # Calculate flux at x1, x2, x3, x4
    time_list.append(t)
    u_x1.append(u_numerical[frame][int(0.75/dx)])
    u_x2.append(u_numerical[frame][int(1.5/dx)])
    u_x3.append(u_numerical[frame][int(2.25/dx)])
    u_x4.append(u_numerical[frame][int(3.25/dx)])
    line3.set_data(time_list, u_x1)
    line4.set_data(time_list, u_x2)
    line5.set_data(time_list, u_x3)
    line6.set_data(time_list, u_x4)
    time_text.set_text(f"t = {t:.2f}")  # Update time label
    return line2, line3, line4, line5, line6, time_text

# Function to save figures with 4 subplots at specific times
def save_figure_at_time(t_target, u_numerical, time_steps, spatial_grid):
    # Find the frame index closest to the target time
    frame_index = np.argmin(np.abs(time_steps - t_target))
    
    # Extract the numerical solution at the target time
    u_at_t = u_numerical[frame_index]
    
    # Create a figure with 4 subplots
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'Modelling Traffic Flow: Initial Condition 1 at t = {t_target:.1f}')
    
    # Subplot 1: Flux function
    u_grid = np.linspace(0, 1, 50)
    ax[0].plot(u_grid, flux(u_grid), 'r-', label='f(u) = u(1 - u)')
    ax[0].set_xlabel("u")
    ax[0].set_ylabel("f(u)")
    ax[0].set_title("Flux Function")
    ax[0].grid()
    ax[0].legend()
    
    # Subplot 2: Numerical solution u(x, t)
    ax[1].plot(spatial_grid, u_at_t, 'g-', label='Numerical Solution, u(x,t)')
    ax[1].plot(spatial_grid, u_initial, 'b--', label='Initial Condition')
    ax[1].set_xlim(0, 4)
    ax[1].set_ylim(-0.25, 1)
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("u(x,t)")
    ax[1].set_title("Solution u(x,t)")
    ax[1].grid()
    ax[1].legend()
    
    # Subplot 3: Flux of the solution f(u(x, t))
    ax[2].plot(spatial_grid, kx, 'b--', label='kx')
    ax[2].set_xlim(0, 4)
    ax[2].set_ylim(0, 2)
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("k(x)")
    ax[2].set_title("k(x)")
    ax[2].grid()
    ax[2].legend()
    
    # Subplot 4: Flux at x* = 2 over time
    time_list = time_steps[:frame_index + 1]
    flux_at_x1 = [u_numerical[i][int(0.75 / dx)] for i in range(frame_index + 1)]
    flux_at_x2 = [u_numerical[i][int(1.5 / dx)] for i in range(frame_index + 1)]
    flux_at_x3 = [u_numerical[i][int(2.25 / dx)] for i in range(frame_index + 1)]
    flux_at_x4 = [u_numerical[i][int(3.25 / dx)] for i in range(frame_index + 1)]
    ax[3].plot(time_list, flux_at_x1, 'r-', label='u(x,t) at x1 = 0.75')
    ax[3].plot(time_list, flux_at_x2, 'g-', label='u(x,t) at x2 = 1.5')
    ax[3].plot(time_list, flux_at_x3, 'b-', label='u(x,t) at x3 = 2.25')
    ax[3].plot(time_list, flux_at_x4, 'c-', label='u(x,t) at x4 = 3.25')
    ax[3].set_xlim(0, T)
    ax[3].set_ylim(-0.25, 1)
    ax[3].set_xlabel("time")
    ax[3].set_ylabel("fu(x*,t)")
    ax[3].set_title("Observation of u(Xi,t)")
    ax[3].grid()
    ax[3].legend()
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f'traffic_flow_1_t_{t_target:.0f}.png')
    plt.close()

# Save figures at T = 0, T = 5, T = 10
save_figure_at_time(5, u_numerical, time_steps, spatial_grid)
save_figure_at_time(10, u_numerical, time_steps, spatial_grid)


# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(time_steps), interval=10, blit=True)
# Show the animation
plt.legend()
plt.show()

# # Save picture for observation data vs predicted data
# fig, ax = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
# fig.suptitle('Observation of u(Xi,t) vs Predicted u(Xi,t)')

# # Predicted Data
# u_at_x1 = [u_numerical[i][int(0.75/dx)] for i in range(nt)]
# u_at_x2 = [u_numerical[i][int(1.5/dx)] for i in range(nt)]
# u_at_x3 = [u_numerical[i][int(2.25/dx)] for i in range(nt)]
# u_at_x4 = [u_numerical[i][int(3.25/dx)] for i in range(nt)]
# ax[0].plot(time_steps, u_at_x1, 'r-', label='Predicted u(x1,t)')
# ax[0].plot(time_steps, u_at_x2, 'g-', label='Predicted u(x2,t)')
# ax[0].plot(time_steps, u_at_x3, 'b-', label='Predicted u(x3,t)')
# ax[0].plot(time_steps, u_at_x4, 'c-', label='Predicted u(x4,t)')
# ax[0].set_xlim(0, T)
# ax[0].set_ylim(0, 1)
# ax[0].set_xlabel("time")
# ax[0].set_ylabel("u(x*,t)")
# ax[0].set_title("Predicted u(Xi,t)")
# ax[0].grid()
# ax[0].legend()

# # Observation data
# observed = scipy.io.loadmat('observation.mat')
# time_observed = observed['time_obs'].flatten() 
# uobs_x1 = observed['u_true_X1'][:, 0]
# uobs_x2 = observed['u_true_X2'][:, 0]
# uobs_x3 = observed['u_true_X3'][:, 0]
# uobs_x4 = observed['u_true_X4'][:, 0]
# ax[1].plot(time_observed, uobs_x1, 'ro', label='Observed u(x1,t)')
# ax[1].plot(time_observed, uobs_x2, 'go', label='Observed u(x2,t)')
# ax[1].plot(time_observed, uobs_x3, 'bo', label='Observed u(x3,t)')
# ax[1].plot(time_observed, uobs_x4, 'co', label='Observed u(x4,t)')
# ax[1].set_xlim(0, T)
# ax[1].set_ylim(0, 1)
# ax[1].set_xlabel("time")
# ax[1].set_ylabel("u(x*,t)")
# ax[1].set_title("Observed u(Xi,t)")
# ax[1].grid()
# ax[1].legend()

# # Save the figure
# plt.tight_layout()
# plt.savefig('observation_vs_predicted.png')
# plt.close()

# End of task 1
