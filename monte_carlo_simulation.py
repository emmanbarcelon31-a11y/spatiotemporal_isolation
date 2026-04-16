import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 1. Variables and Log-uniform priors
n_sims = 1000000
N_safe = 10 ** np.random.uniform(8, 9, n_sims)
f_bio = 10 ** np.random.uniform(-8, -3, n_sims)
L = 10 ** np.random.uniform(3, 7, n_sims)

# Effective Detection Radius Prior (10 to 100,000 ly)
R_eff_prior = 10 ** np.random.uniform(1, 5, n_sims)

# Expansion Speed Prior (0.0001c to 0.01c)
v_exp = 10 ** np.random.uniform(-4, -2, n_sims)

# Physical Constants (Explicit for code readability)
c = 1.0  # Speed of light in ly/yr
t_present = 13.6e9

# 2. The Rigorous Spatiotemporal Math

# Hypoexponential Biological Delay (Varying 'Hard Step' Rates)
lambda_rates = np.array([1/0.5e9, 1/0.7e9, 1/0.9e9, 1/1.1e9, 1/1.3e9])
n_steps = len(lambda_rates)

def p_hypoexponential(delta_t):
    if delta_t < 0: return 0
    prob = 0
    for i in range(n_steps):
        C_i = 1.0
        for j in range(n_steps):
            if i != j:
                C_i *= lambda_rates[j] / (lambda_rates[j] - lambda_rates[i])
        prob += C_i * lambda_rates[i] * np.exp(-lambda_rates[i] * delta_t)
    return prob

# Star Formation Rate: Parametrized Snaith et al. (Peaks at 7 Gyr)
def sfr_shape(tau):
    a = 2.8
    b = a / 7.0e9
    return (tau**a) * np.exp(-b * tau)

sfr_norm, _ = quad(sfr_shape, 0, t_present)

def sfr_normalized(tau):
    return sfr_shape(tau) / sfr_norm

def integrand(tau):
    return sfr_normalized(tau) * p_hypoexponential(t_present - tau)

# Dynamic Concurrency
E_present, _ = quad(integrand, 0, t_present)
N_concurrent = N_safe * f_bio * E_present * L

# --- STATIONARY BOUNDARY ---
# Formatted to explicitly mirror Eq 5 to prevent reviewer confusion.
# Note: For concurrent civs, delta_t <= L, so R_inner mathematically collapses to 0.
R_outer_stat = np.minimum(c * L, R_eff_prior)
R_inner_stat = np.maximum(0, R_outer_stat - c * L)

V_shell_stat = (4/3) * np.pi * (R_outer_stat**3 - R_inner_stat**3)
V_GHZ = 2.5e11
V_shell_stat = np.minimum(V_shell_stat, V_GHZ) # Cap volume to galaxy size
P_spatial_stat = V_shell_stat / V_GHZ

# Calculate STATIONARY Expected Contacts
N_contact_stat = N_concurrent * P_spatial_stat

# --- SUB-LIGHT EXPANSION CAVEAT TEST ---
# v_exp is fractional c; L is in years; so R_col is in light-years (since c=1 ly/yr)
R_col = v_exp * L
R_outer_exp = R_col + np.minimum(c * L, R_eff_prior)
R_inner_exp = np.maximum(0, R_outer_exp - c * L)

V_shell_exp = (4/3) * np.pi * (R_outer_exp**3 - R_inner_exp**3)
V_shell_exp = np.minimum(V_shell_exp, V_GHZ)
P_spatial_exp = V_shell_exp / V_GHZ

# Calculate EXPANDING Expected Contacts
N_contact_exp = N_concurrent * P_spatial_exp

# Calculate Isolation Percentages
prob_isolation_stat = np.sum(N_contact_stat < 1) / n_sims * 100
prob_isolation_exp = np.sum(N_contact_exp < 1) / n_sims * 100

print(f"Stationary Isolation: {prob_isolation_stat:.3f}%")
print(f"Expanding Isolation: {prob_isolation_exp:.3f}%")

# --- VISUALIZATION PREP ---
# Filter out strict zeros before applying log10 to prevent artificial -12 spikes
valid_stat = N_contact_stat > 0
log_N_contact_stat = np.full(n_sims, -12.0) # Background floor for non-contacts
log_N_contact_stat[valid_stat] = np.log10(N_contact_stat[valid_stat])

valid_exp = N_contact_exp > 0
log_N_contact_exp = np.full(n_sims, -12.0)
log_N_contact_exp[valid_exp] = np.log10(N_contact_exp[valid_exp])


# 3. Plotting
plt.figure(figsize=(10,6))

# Plot the original Stationary Histogram (limiting range to avoid the -12 floor block)
bins = np.linspace(-10, np.max(log_N_contact_stat), 100)
counts, bins, patches = plt.hist(log_N_contact_stat, bins=bins, edgecolor='black', alpha=0.85, label='Stationary Civs')

for patch, bin_left in zip(patches, bins[:-1]):
    if bin_left < 0:
        patch.set_facecolor('darkslateblue') # Isolation
    else:
        patch.set_facecolor('darkorange') # Contact

# Overlay the Expanding Histogram as a clear outline
plt.hist(log_N_contact_exp, bins=bins, histtype='step', color='red', linewidth=2, linestyle='dotted', label='Expanding Civs (v = 0.0001c to 0.01c)')

plt.axvline(x=0, color='red', linestyle='-', linewidth=2.5, label='Contact Boundary (N >= 1)')
plt.title('Monte Carlo: Expected Spatiotemporal Contacts (Expansion Test)', fontsize=15, pad=15)
plt.xlabel('Log10(N_contact)', fontsize=13)
plt.ylabel('Frequency (10^6 Simulations)', fontsize=13)
plt.legend(fontsize=11, loc='upper left')
plt.grid(axis='y', alpha=0.33)

# Add a text box comparing the two percentages
info_text = (f"Stationary Isolation: {prob_isolation_stat:.3f}%\n"
             f"Expanding Isolation: {prob_isolation_exp:.3f}%")
plt.figtext(0.65, 0.5, info_text, fontsize=12, bbox=dict(facecolor='white', edgecolor='black', alpha=0.9))

plt.tight_layout()
plt.savefig('spatiotemporal_histogram_final.png', dpi=300)
plt.show()
