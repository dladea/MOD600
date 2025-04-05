# Project 3: Task 1
# Created by: Dea Lana Asri - 277575
# dl.asri@stud.uis.no
# Date: 21.03.2025
# Code is developed with aid of AI

import numpy as np
import matplotlib.pyplot as plt
import sys
import time

# Set random seed for reproducibility
np.random.seed(42)

plt.ion()
fig_live, axs_live = plt.subplots(1, 2, figsize=(14, 6))

# Load Data 
observation1 = np.load("data-2/True_Observation_data_1_upd.npz")
observation2 = np.load("data-2/True_Observation_data_2_upd.npz")

# Extract parameters from observation data
x = observation1["x"]
xh = observation1["xh"]
N = observation1["M"].item()
Ldom = observation1["Ldom"].item()
T = observation1["T"].item()
time_obs = observation1["t_obspoint"][observation1["t_obspoint"] <= T]
x_k = observation1["x_k"]
Mr = observation1["Mr"].item()
X_positions = observation1["X_position"]

# Observation Data
u_true_1 = [observation1["u_true_X1"], observation1["u_true_X2"],
            observation1["u_true_X3"], observation1["u_true_X4"]]
u_true_2 = [observation2["u_true_X1"], observation2["u_true_X2"],
            observation2["u_true_X3"], observation2["u_true_X4"]]

# Initial Conditions
initial_condition1 = observation1["u0"]
initial_condition2 = observation2["u0"]

# MCMC Parameters
add_unc = 0.03
error_val = 500
kk_tolerance = 1000
extend_iterations = 1000
tau = 20
beta = 0.05
mu = 1.0
sigma = 0.25
plot_update_frequency = 5 * tau

def flux(u):
    return u * (1 - u)

def flux_derivative(u):
    return 1 - 2 * u

def rusanov_flux(u_left, u_right):
    dl = flux_derivative(u_left) if u_left is not None else 0
    dr = flux_derivative(u_right) if u_right is not None else 0
    a_max = max(abs(dl), abs(dr))
    fl = flux(u_left) if u_left is not None else 0
    fr = flux(u_right) if u_right is not None else 0
    diff_term = (u_right - u_left) if u_right is not None and u_left is not None else 0
    return 0.5 * (fl + fr) - 0.5 * a_max * diff_term

def create_k_function(k_values, x_grid):
    global Ldom
    k_i = np.concatenate(([1.0], k_values, [1.0]))
    x_k_interp_nodes = np.linspace(0, Ldom, len(k_i))
    return np.interp(x_grid, x_k_interp_nodes, k_i)

def solver(k_values, initial_conditions, x_k_ctrl, xh_grid, N_cells, L_domain, time_obs_points, X_obs_positions):
    u_obs = []
    num_observation_points = len(X_obs_positions)
    num_time_steps = len(time_obs_points)

    if not isinstance(initial_conditions, list):
        local_initial_conditions = [initial_conditions]
    else:
        local_initial_conditions = initial_conditions

    cell_centers = 0.5 * (xh_grid[:-1] + xh_grid[1:])
    obs_indices = np.array([np.argmin(np.abs(cell_centers - pos)) for pos in X_obs_positions], dtype=int)

    kx = create_k_function(k_values, xh_grid)

    dx = L_domain / N_cells
    max_k_runtime = max(1.0, np.max(kx))
    dt = 0.95 * dx / (max_k_runtime * 1.0)

    for ic_index, ic in enumerate(local_initial_conditions):
        u_old = np.copy(ic)
        u = np.zeros(N_cells)
        current_sim_time = 0.0
        time_obs_idx = 0
        u_obs_single_ic = np.zeros((num_observation_points, num_time_steps))

        while time_obs_idx < num_time_steps:
            fluxes = np.zeros(N_cells + 1)
            u_padded = np.pad(u_old, (1, 1), mode='edge')
            for i in range(N_cells + 1):
                u_left = u_padded[i]
                u_right = u_padded[i+1]
                fluxes[i] = rusanov_flux(u_left, u_right)

            for i in range(N_cells):
                idx_k_right = min(i + 1, len(kx) - 1)
                idx_k_left = max(i, 0)
                k_right_interface = kx[idx_k_right]
                k_left_interface = kx[idx_k_left]
                idx_f_right = min(i + 1, N_cells)
                idx_f_left = max(i, 0)
                term = (dt / dx) * (k_right_interface * fluxes[idx_f_right] - k_left_interface * fluxes[idx_f_left])
                u[i] = u_old[i] - term

            if N_cells > 1:
                 u[0] = u[1]
                 u[N_cells-1] = u[N_cells-2]

            u_old = u.copy()
            current_sim_time += dt

            while time_obs_idx < num_time_steps and current_sim_time >= time_obs_points[time_obs_idx] - 1e-9:
                 for i in range(num_observation_points):
                     u_idx = obs_indices[i]
                     safe_u_idx = max(0, min(u_idx, N_cells - 1))
                     u_obs_single_ic[i, time_obs_idx] = u_old[safe_u_idx]
                 time_obs_idx += 1
                 if time_obs_idx == num_time_steps:
                     break
        u_obs.append(u_obs_single_ic)
    return u_obs

def calculate_loss(predicted_data, observed_data, gamma):
    diff = predicted_data - observed_data
    loss = 0.5 * np.sum(diff**2) / (gamma**2)
    return loss

def update_live_plot(fig, axs, current_iter, loss_history, ensemble,
                     initial_theta, x_plot_grid, run_label, total_iters):
    axs[0].clear()
    axs[1].clear()
    iterations_sampled = (np.arange(len(loss_history)) + 1) * tau
    axs[0].plot(iterations_sampled, loss_history, marker='.', linestyle='-', markersize=3)
    axs[0].set_xlabel("MCMC Iteration")
    axs[0].set_ylabel("Total Loss Function Value")
    axs[0].set_title(f"Loss History ({run_label} - Iter {current_iter}/{total_iters})")
    axs[0].grid(True)
    finite_losses = [l for l in loss_history if np.isfinite(l)] 
    if finite_losses:
         min_loss = np.min(finite_losses)
         if min_loss > 0:
              axs[0].set_yscale('log')

    ensemble_np = np.array(ensemble)
    num_ensemble_members = len(ensemble_np)
    if num_ensemble_members > 0:
        max_lines_to_plot = 100
        plot_indices = np.random.choice(num_ensemble_members, size=min(num_ensemble_members, max_lines_to_plot), replace=False)
        for i in plot_indices:
            k_vals = ensemble_np[i]
            k_func = create_k_function(k_vals, x_plot_grid)
            axs[1].plot(x_plot_grid, k_func, color='gray', alpha=0.1)
        mean_k_values = np.mean(ensemble_np, axis=0)
        mean_k_func = create_k_function(mean_k_values, x_plot_grid)
        axs[1].plot(x_plot_grid, mean_k_func, color='red', linewidth=2.0, label=f'Mean k(x) (N={num_ensemble_members})')

    initial_k_func = create_k_function(initial_theta, x_plot_grid)
    axs[1].plot(x_plot_grid, initial_k_func, color='blue', linestyle='--', linewidth=1.5, label='Initial k(x)')
    axs[1].set_xlabel("Position x")
    axs[1].set_ylabel("k(x)")
    axs[1].set_title(f"k(x) Ensemble ({run_label})")
    axs[1].legend(fontsize='small')
    axs[1].grid(True)
    axs[1].set_ylim(0, 2.0)
    fig.suptitle(f"MCMC Progress ({run_label}) - Iteration {current_iter}/{total_iters}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.draw()
    plt.pause(0.01)

def run_mcmc(initial_theta, initial_conditions, observed_data_input,
             n_iterations, beta, sigma_prior, gamma_likelihood, thinning_interval,
             x_k_nodes, xh_grid, N_grid, L_domain, time_observation, X_observation_positions,
             k_min=0.5, k_max=1.5,
             fig=None, axs=None, plot_freq=100, run_label=""):

    Mr = len(initial_theta)
    current_theta = np.clip(initial_theta, k_min, k_max)
    observed_data_structured = []
    if isinstance(observed_data_input[0], list):
        for obs_list_for_ic in observed_data_input:
             observed_data_structured.append(np.stack(obs_list_for_ic, axis=0))
    elif isinstance(observed_data_input[0], np.ndarray):
        observed_data_structured.append(np.stack(observed_data_input, axis=0))

    def calculate_total_loss(predictions_list, observations_list, gamma_loss):
        total_loss = 0.0
        for i, (pred, obs) in enumerate(zip(predictions_list, observations_list)):
             loss_i = calculate_loss(pred, obs, gamma_loss)
             total_loss += loss_i
        return total_loss

    current_predictions_list = solver(current_theta, initial_conditions, x_k_nodes, xh_grid, N_grid, L_domain, time_observation, X_observation_positions)
    current_loss = calculate_total_loss(current_predictions_list, observed_data_structured, gamma_likelihood)
    best_loss_so_far = current_loss
    best_theta_so_far = current_theta.copy()

    ensemble = []
    loss_history = []
    accepted_count = 0

    print(f"Starting MCMC ({run_label}). Initial Total Loss: {current_loss:.4f}")

    for k in range(n_iterations):
        epsilon = np.random.normal(0, sigma_prior, size=Mr)
        proposed_theta_unclipped = np.sqrt(1.0 - beta**2) * current_theta + beta * epsilon
        proposed_theta = np.clip(proposed_theta_unclipped, k_min, k_max)

        proposed_predictions_list = solver(
            proposed_theta, initial_conditions, x_k_nodes, xh_grid, N_grid, L_domain,
            time_observation, X_observation_positions
        )
        proposed_loss = calculate_total_loss(proposed_predictions_list, observed_data_structured, gamma_likelihood)

        alpha = 0.0
        if np.isfinite(proposed_loss) and np.isfinite(current_loss):
            log_alpha = min(0.0, current_loss - proposed_loss)
            alpha = np.exp(log_alpha)

        u_rand = np.random.uniform(0, 1)
        if u_rand < alpha:
            current_theta = proposed_theta
            current_loss = proposed_loss
            accepted_count += 1
            if current_loss < best_loss_so_far:
                 best_loss_so_far = current_loss
                 best_theta_so_far = current_theta.copy()

        if (k + 1) % thinning_interval == 0:
            ensemble.append(current_theta.copy())
            loss_history.append(current_loss)
            if fig is not None and axs is not None and (k + 1) % plot_freq == 0:
                 update_live_plot(fig, axs, k + 1, loss_history, ensemble,
                                  initial_theta, xh_grid, run_label, n_iterations)
            if (k + 1) % (thinning_interval * 10) == 0 or (k + 1) == n_iterations:
                 current_acceptance_rate = accepted_count / (k + 1) if (k + 1) > 0 else 0
                 print(f"Iter {k+1}/{n_iterations} ({run_label}), Loss: {current_loss:.4f}, Acc. Rate: {current_acceptance_rate:.3f}")

    acceptance_rate = accepted_count / n_iterations if n_iterations > 0 else 0
    final_loss = loss_history[-1] if loss_history else current_loss
    print(f"MCMC ({run_label}) finished. Final Loss: {final_loss:.4f}, Overall Acc. Rate: {acceptance_rate:.3f}")
    print(f"  Best Loss Found ({run_label}): {best_loss_so_far:.4f}")

    if fig is not None and axs is not None:
         update_live_plot(fig, axs, n_iterations, loss_history, ensemble,
                          initial_theta, xh_grid, run_label, n_iterations)

    return ensemble, loss_history, acceptance_rate, best_theta_so_far, best_loss_so_far

# Main Execution
best_k_results = {}
best_k_results['initial_k_partA'] = None
best_k_results['best_k_partA_run1'] = None
best_k_results['best_k_partA_run2'] = None
best_k_results['best_k_partA_overall'] = None
best_k_results['initial_k_partB'] = None
best_k_results['best_k_partB_initial'] = None
best_k_results['best_k_partB_extension'] = None
best_k_results['best_k_partB_final'] = None


print(f"Generating initial k_values from N(mean={mu}, std_dev={sigma}), size={Mr}")
initial_k_values = np.random.normal(mu, sigma, size=Mr)
initial_k_values = np.clip(initial_k_values, 0.5, 1.5)
best_k_results['initial_k_partA'] = initial_k_values.copy()
print("Initial randomly generated k_values (theta_0):", np.round(initial_k_values, 3))

sim_params = {
    "x_k_nodes": x_k,
    "xh_grid": xh,
    "N_grid": N,
    "L_domain": Ldom,
    "time_observation": time_obs,
    "X_observation_positions": X_positions
}

# Part A: Use only the first observation dataset
print("\n--- Running MCMC Part (a), Run 1 ---")
ensemble_a1, loss_history_a1, acc_rate_a1, best_k_a1, best_loss_a1 = run_mcmc(
    initial_theta=initial_k_values.copy(),
    initial_conditions=initial_condition1,
    observed_data_input=u_true_1,
    n_iterations=kk_tolerance,
    beta=beta,
    sigma_prior=sigma,
    gamma_likelihood=add_unc,
    thinning_interval=tau,
    k_min=0.5, k_max=1.5,
    fig=fig_live, axs=axs_live, plot_freq=plot_update_frequency, run_label="Part A Run 1",
    **sim_params
)
best_k_results['best_k_partA_run1'] = best_k_a1.copy()


print("\n--- Running MCMC Part (a), Run 2 (Continuing) ---")
if len(ensemble_a1) > 0:
    last_theta_a1 = ensemble_a1[-1]
    print("Starting Run 2 from the last state of Run 1.")
    ensemble_a2, loss_history_a2, acc_rate_a2, best_k_a2, best_loss_a2 = run_mcmc(
        initial_theta=last_theta_a1.copy(),
        initial_conditions=initial_condition1,
        observed_data_input=u_true_1,
        n_iterations=kk_tolerance,
        beta=beta,
        sigma_prior=sigma,
        gamma_likelihood=add_unc,
        thinning_interval=tau,
        k_min=0.5, k_max=1.5,
        fig=fig_live, axs=axs_live, plot_freq=plot_update_frequency, run_label="Part A Run 2",
        **sim_params
    )
    best_k_results['best_k_partA_run2'] = best_k_a2.copy()
else:
    print("Warning: Ensemble from Run 1 is empty. Skipping Run 2.")
    ensemble_a2, loss_history_a2, acc_rate_a2, best_k_a2, best_loss_a2 = [], [], 0.0, initial_k_values, np.inf

ensemble_a1_np = np.array(ensemble_a1) if ensemble_a1 else np.empty((0, Mr))
ensemble_a2_np = np.array(ensemble_a2) if ensemble_a2 else np.empty((0, Mr))
ensemble_a_combined_np = np.concatenate((ensemble_a1_np, ensemble_a2_np), axis=0)
loss_history_a_combined = loss_history_a1 + loss_history_a2
best_k_a_overall = initial_k_values
best_loss_a_overall = np.inf


if best_loss_a1 < best_loss_a_overall:
    best_loss_a_overall = best_loss_a1
    best_k_a_overall = best_k_results['best_k_partA_run1']
if best_loss_a2 < best_loss_a_overall: 
    best_loss_a_overall = best_loss_a2
    best_k_a_overall = best_k_results['best_k_partA_run2']

if best_k_a_overall is not initial_k_values:
    best_k_results['best_k_partA_overall'] = best_k_a_overall.copy()
    print(f"\nOverall best k(x) for Part A found with Loss = {best_loss_a_overall:.4f}")
else:
    print("\nCould not find a valid best k(x) for Part A (possibly Run 1 & 2 failed or didn't improve).")
final_loss_a = loss_history_a_combined[-1] if loss_history_a_combined else np.inf
print(f"Final Loss after executed iterations (Part A): {final_loss_a:.4f}")

# Part B: Use both observation datasets
print("\n--- Running MCMC Part (b) (Combined Observations) ---")
combined_initial_conditions = [initial_condition1, initial_condition2]
combined_observed_data_input = [u_true_1, u_true_2]
initial_k_values_b = np.clip(np.random.normal(mu, sigma, size=Mr), 0.5, 1.5)
best_k_results['initial_k_partB'] = initial_k_values_b.copy()
print("Initial randomly generated k_values for Part (b):", np.round(initial_k_values_b, 3))
n_iterations_b_initial = 2 * kk_tolerance

ensemble_b, loss_history_b, acc_rate_b, best_k_b, best_loss_b = run_mcmc(
    initial_theta=initial_k_values_b.copy(),
    initial_conditions=combined_initial_conditions,
    observed_data_input=combined_observed_data_input,
    n_iterations=n_iterations_b_initial,
    beta=beta,
    sigma_prior=sigma,
    gamma_likelihood=add_unc,
    thinning_interval=tau,
    k_min=0.5, k_max=1.5,
    fig=fig_live, axs=axs_live, plot_freq=plot_update_frequency, run_label="Part B Initial",
    **sim_params
)
best_k_results['best_k_partB_initial'] = best_k_b.copy()
final_loss_b_initial = loss_history_b[-1] if loss_history_b else np.inf
print(f"Loss after initial {n_iterations_b_initial} iterations for Part (b): {final_loss_b_initial:.4f}")

# Conditional Extension for Part B if loss is not satisfactory
total_iterations_b = n_iterations_b_initial
best_k_b_ext_run = None
if np.isfinite(final_loss_b_initial) and final_loss_b_initial > error_val and extend_iterations > 0:
    print(f"\n--- Extending MCMC Part (b) as loss {final_loss_b_initial:.4f} > {error_val} ---")
    if len(ensemble_b) > 0:
        last_theta_b = ensemble_b[-1]
        print(f"Starting extension ({extend_iterations} iterations) from the last state.")

        ensemble_b_ext, loss_history_b_ext, acc_rate_b_ext, best_k_b_ext_run, best_loss_b_ext = run_mcmc(
            initial_theta=last_theta_b.copy(),
            initial_conditions=combined_initial_conditions,
            observed_data_input=combined_observed_data_input,
            n_iterations=extend_iterations,
            beta=beta,
            sigma_prior=sigma,
            gamma_likelihood=add_unc,
            thinning_interval=tau,
            k_min=0.5, k_max=1.5,
            fig=fig_live, axs=axs_live, plot_freq=plot_update_frequency, run_label="Part B Extended",
            **sim_params
        )
        ensemble_b = ensemble_b + ensemble_b_ext
        loss_history_b = loss_history_b + loss_history_b_ext
        total_iterations_b += extend_iterations
        best_k_results['best_k_partB_extension'] = best_k_b_ext_run.copy()

        current_overall_best_loss = best_loss_b if np.isfinite(best_loss_b) else np.inf
        if np.isfinite(best_loss_b_ext) and best_k_b_ext_run is not None and best_loss_b_ext < current_overall_best_loss:
             best_k_b = best_k_b_ext_run
             best_loss_b = best_loss_b_ext
             print(f"Extension found a new overall best k(x) for Part B (Loss={best_loss_b:.4f})")
    else:
        print("Warning: Cannot extend Part B run as the initial run produced no valid states.")
elif final_loss_b_initial <= error_val: print(f"\nLoss {final_loss_b_initial:.4f} <= {error_val}, no extension needed.")
else: print(f"\nInitial Part B loss was not finite ({final_loss_b_initial}), cannot determine extension.")


# Final results for Part B
final_loss_b_overall = loss_history_b[-1] if loss_history_b else np.inf
print(f"\n--- Final Results Part (b) ({total_iterations_b} total iterations) ---")
print(f"Final Loss (at end of chain): {final_loss_b_overall:.4f}")
best_k_results['best_k_partB_final'] = best_k_b.copy()
print(f"Overall Best Loss Found: {best_loss_b:.4f}")
print(f"Parameters for Overall Best k(x): {np.round(best_k_b, 3)}")


# Saving results
k_results_to_save = {key: value for key, value in best_k_results.items() if value is not None}
if k_results_to_save:
    save_filename = "best_k_results.npz"
    try:
        np.savez_compressed(save_filename, **k_results_to_save)
        print(f"\nBest k(x) parameter sets saved to '{save_filename}'")
        print("Saved keys:", list(k_results_to_save.keys()))
    except Exception as e:
        print(f"\nError saving best k results to '{save_filename}': {e}")
else:
    print("\nNo valid best k results were found to save.")

plt.ioff()
print("\nClose the plot window to exit the script.")
plt.show()