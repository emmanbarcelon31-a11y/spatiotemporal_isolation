import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 1. Variables and Log-uniform priors
n_sims = 1000000
N_safe = 10 ** np.random.uniform(8, 9, n_sims)
f_bio = 10 ** np.random.uniform(-8, -3, n_sims)
L = 10 ** np.random.uniform(3, 7, n_sims)

# 2. The Rigorous Spatiotemporal Math
t_present = 13.6e9

# UPGRADE 1: Hypoexponential Biological Delay (Varying 'Hard Step' Rates)
# 5 steps summing to roughly 4.5 billion years, with distinct rates
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

# UPGRADE 2: Dynamic Concurrency (Fixed)
# Calculate Present Day Emergence Rate (E_present)
E_present, _ = quad(integrand, 0, t_present)

# Since L is very small compared to galactic history, integrating E(t)
# over the lifespan window L is accurately simplified to E_present * L.
N_concurrent = N_safe * f_bio * E_present * L

# UPGRADE 3: Relativistic Light Shell Volume (Spatial Contact)
# Signals expand at speed of light (c = 1 ly/yr).
# Effective signal radius is bounded by lifespan and max detectability limit (1000 ly)
R_eff = np.minimum(L, 1000)
V_shell = (4/3) * np.pi * (R_eff**3)
V_GHZ = 2.5e11
P_spatial = V_shell / V_GHZ

# Calculate actual expected CONTACTS
N_contact = N_concurrent * P_spatial
log_N_contact = np.log10(N_contact + 1e-12)

# 3. Plotting
plt.figure(figsize=(10,6))
counts, bins, patches = plt.hist(log_N_contact, bins=100, edgecolor='black', alpha=0.85)

for patch, bin_left in zip(patches, bins[:-1]):
    if bin_left < 0:
        patch.set_facecolor('darkslateblue') # Isolation
    else:
        patch.set_facecolor('darkorange') # Contact

plt.axvline(x=0, color='red', linestyle='--', linewidth=2.5, label='Contact Boundary (N_contact >= 1)')
plt.title('Monte Carlo: Expected Spatiotemporal Contacts', fontsize=15, pad=15)
plt.xlabel('Log10(N_contact)', fontsize=13)
plt.ylabel('Frequency (10^6 Simulations)', fontsize=13)
plt.legend(fontsize=12)
plt.grid(axis='y', alpha=0.33)

prob_isolation = np.sum(N_contact < 1) / n_sims * 100
plt.figtext(0.15, 0.75, f"Spatiotemporal Isolation:\n{prob_isolation:.3f}% yield N < 1",
            fontsize=12, bbox=dict(facecolor='white', edgecolor='black', alpha=0.9))

plt.tight_layout()
plt.savefig('spatiotemporal_histogram_upgraded.png', dpi=300)
plt.show()
