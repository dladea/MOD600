# 3e

import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# Parameters
T = 100
time_steps = 2000
dt = T / time_steps
L = 20 # Axon length
D0 = 0.4 # Diffusion coefficient
N0 = 0.1 # Coefficient of free particles at x = 0
NL = 0.01 # Coefficient of free particles at x = L
kmin = 1 # Binding rate in minimum direction
sigma0 = 0.1 # degree of loading at x = 0
sigmaL = 0.1 # degree of loading at x = L
kp = 0.5 # Detachment rate for particles in + direction
kn = 0.5 # Detachment rate for particles in - direction
dx = 0.2 # grid spacing
current_time = 0

# Spatial grid
M = int(L / dx) # number of grid blocks
x = np.linspace(dx * 0.5, L - 0.5*dx, M)


# Initial condition
n0 = N0 + (NL - N0) * x / L
npos0 = 0
nneg0 = 0

# Calculate numerical solution for free particles
n_old = np.copy(n0)
n = np.zeros(M)

# Calculate numerical solution for loaded particles in + direction
npos_old = np.full(M, npos0)
npos = np.zeros(M)

# Calculate numerical solution for loaded particles in - direction
nneg_old = np.full(M, nneg0)
nneg = np.zeros(M)

# Flux
flux = np.zeros(M)

# Prepare the figure
fig, ax = plt.subplots(1, 4, figsize=(25, 7))
fig.suptitle("Task 3E", fontsize=16)
time_text = fig.text(0.5, 0.02, "", ha='center', va='bottom', fontsize=14, bbox=dict(facecolor='white', alpha=0.8))

ax[0].set_ylabel("Total Density")
ax[0].set_ylim(-0.05, 0.2)
ax[0].set_title("Total Density (n + npos + nneg)")
line_total, = ax[0].plot([], [], 'ob', label='n + npos + nneg')
total_flux, = ax[0].plot([], [], '--g', label='Total Flux', linewidth=3)
line_init_total, = ax[0].plot(x, n0, '-r', label='Initial Condition')

ax[1].set_ylabel("n")
ax[1].set_ylim(-0.01, 0.2)
ax[1].set_title("Concentration of Free Particles")
line_num, = ax[1].plot([], [], 'ob', label='Numerical Solution')
line_init, = ax[1].plot(x, n0, '-r', label='Initial Condition')

ax[2].set_ylabel("n positive")
ax[2].set_ylim(-0.01, 0.06)
ax[2].set_title("Concentration of Loaded Particles in + direction")
line_num_pos, = ax[2].plot([], [], 'ob', label='Numerical Solution')
line_init_pos, = ax[2].plot(x, np.zeros(M), '-r', label='Initial Condition')

ax[3].set_ylabel("n negative")
ax[3].set_ylim(-0.01, 0.06)
ax[3].set_title("Concentration of Loaded Particles in - direction")
line_num_neg, = ax[3].plot([], [], 'ob', label='Numerical Solution')
line_init_neg, = ax[3].plot(x, np.zeros(M), '-r', label='Initial Condition')

for i in range(len(ax)):
    ax[i].set_xlim(0, L)
    ax[i].set_xlabel("x-axis")
    ax[i].grid()
    ax[i].legend()

def stability_check(D0, dx, dt):
    return D0 * dt / dx**2
def CFL(dt, dx):
    return dt/dx

if stability_check(D0, dx, dt) > 0.5 or CFL(dt, dx) > 1:
    print("Stability condition is not satisfied")
    sys.exit()


# Initialize animation
def init():
    line_num.set_data([], [])
    line_num_pos.set_data([], [])
    line_num_neg.set_data([], [])
    line_total.set_data([], [])
    total_flux.set_data([], [])
    time_text.set_text("")
    return line_num, line_num_pos, line_num_neg, line_total, total_flux, time_text,

# Update function for animation
def update(frame):
    global n_old, n, npos_old, npos, nneg_old, nneg, flux, current_time
    current_time = (frame+1) * dt
    for j in range(1, M - 1):
        n[j] = n_old[j] + dt * D0 / dx**2 * (n_old[j+1] - 2 * n_old[j] + n_old[j-1]) + dt * (kp * npos_old[j] + kn * nneg_old[j])
        npos[j] = npos_old[j] - dt / dx * (npos_old[j]- npos_old[j-1]) - dt * kp * npos_old[j]
        nneg[j] = nneg_old[j] + dt / dx * (nneg_old[j+1] - nneg_old[j]) - dt * kn * nneg_old[j]
    # Set the boundary conditions
    n[0] = N0
    n[-1] = NL
    n_old = np.copy(n)
    npos[0] = N0 * sigma0
    npos_old = np.copy(npos)
    nneg[-1] = NL * sigmaL
    nneg_old = np.copy(nneg)
    flux[1:]= npos[1:] - nneg[1:] - D0 * (n[1:] - n[:-1]) / dx
    flux[0] = flux[1]
    flux[-1] = flux[-2]
    
    # Update data for animation
    line_num.set_data(x, n)
    line_num_pos.set_data(x, npos)
    line_num_neg.set_data(x, nneg)
    line_total.set_data(x, n + npos + nneg)
    total_flux.set_data(x, flux)
    time_text.set_text(f"Time: {current_time:.2f}")
    return line_num, line_num_pos, line_num_neg, line_total, total_flux, time_text,

# Create animation
anim = FuncAnimation(fig, update, frames=time_steps, init_func=init, blit=False)

# Save as GIF
anim.save("3edettachment100.gif", writer=PillowWriter(fps=30))
print("Animation saved as 3edettachment100.gif")