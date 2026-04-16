import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 1. Variables and Log-uniform priors
n_sims = 1000000
N_safe = 10 ** np.random.uniform(8, 9, n_sims)
f_bio = 10 ** np.random.uniform(-8, -3, n_sims)
L = 10 ** np.random.uniform(3, 7, n_sims)

# --- NEW: Expansion Speed Prior (0.0001c to 0.01c) ---
v_exp = 10 ** np.random.uniform(-4, -2, n_sims)

# 2. The Rigorous Spatiotemporal Math
t_present = 13.6e9

# UPGRADE 1: Hypoexponential Biological Delay (Varying 'Hard Step' Rates)
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

# UPGRADE 2: Dynamic Concurrency
E_present, _ = quad(integrand, 0, t_present)
N_concurrent = N_safe * f_bio * E_present * L

# UPGRADE 3: Relativistic Light Shell Volume (STATIONARY BOUNDARY)
R_eff_stat = np.minimum(L, 1000)
V_shell_stat = (4/3) * np.pi * (R_eff_stat**3)
V_GHZ = 2.5e11
P_spatial_stat = V_shell_stat / V_GHZ

# Calculate STATIONARY Expected Contacts
N_contact_stat = N_concurrent * P_spatial_stat
log_N_contact_stat = np.log10(N_contact_stat + 1e-12)

# --- UPGRADE 4: SUB-LIGHT EXPANSION CAVEAT TEST ---
# Total radius = Colonized space + 1000ly radio bleed from the outermost colony
R_col = v_exp * L
R_eff_exp = R_col + np.minimum(L, 1000)

# Calculate EXPANDING Volume (Capped at V_GHZ to prevent probabilities > 1)
V_shell_exp = (4/3) * np.pi * (R_eff_exp**3)
V_shell_exp = np.minimum(V_shell_exp, V_GHZ)
P_spatial_exp = V_shell_exp / V_GHZ

# Calculate EXPANDING Expected Contacts
N_contact_exp = N_concurrent * P_spatial_exp
log_N_contact_exp = np.log10(N_contact_exp + 1e-12)

# Calculate Isolation Percentages
prob_isolation_stat = np.sum(N_contact_stat < 1) / n_sims * 100
prob_isolation_exp = np.sum(N_contact_exp < 1) / n_sims * 100

print(f"Stationary Isolation: {prob_isolation_stat:.3f}%")
print(f"Expanding Isolation: {prob_isolation_exp:.3f}%")

# 3. Plotting
plt.figure(figsize=(10,6))

# Plot the original Stationary Histogram
counts, bins, patches = plt.hist(log_N_contact_stat, bins=100, edgecolor='black', alpha=0.85, label='Stationary Civs')

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
plt.savefig('spatiotemporal_histogram_expansion_test.png', dpi=300)
plt.show()
