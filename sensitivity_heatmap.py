import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import quad

# 1. Parameter Space Setup (Updated for 2026 Consensus)
resolution = 50
f_bio_log = np.linspace(-15, -2, resolution) # Expanded range to match Monte Carlo
L_log = np.linspace(3, 7, resolution)
heatmap_data = np.zeros((resolution, resolution))

# 2. The Rigorous Spatiotemporal Math
t_present = 13.6e9

# Hypoexponential Biological Delay
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

def sfr_shape(tau):
    a = 2.8
    b = a / 7.0e9
    return (tau**a) * np.exp(-b * tau)

sfr_norm, _ = quad(sfr_shape, 0, t_present)

def sfr_normalized(tau):
    return sfr_shape(tau) / sfr_norm

def integrand(tau):
    return sfr_normalized(tau) * p_hypoexponential(t_present - tau)

# Spatial Constants (Updated 2026 Boundaries)
N_safe_avg = 5e8
V_GHZ = 7.8e12  # Updated modern 3D volume

# Calculate E_present once
E_present, _ = quad(integrand, 0, t_present)

for i, l_val in enumerate(L_log):
    for j, f_val in enumerate(f_bio_log):
        L = 10 ** l_val
        f_bio = 10 ** f_val

        # Calculate concurrent civilizations
        N_concurrent = N_safe_avg * f_bio * E_present * L

        # Relativistic Light Shell Volume (2026 Leakage Limit)
        R_eff = np.minimum(L, 200) # Capped at 200 ly median for unintentional leakage
        V_shell = (4/3) * np.pi * (R_eff**3)
        P_spatial = V_shell / V_GHZ

        # Calculate actual expected CONTACTS
        N_contact = N_concurrent * P_spatial
        prob_contact = 1 - np.exp(-N_contact)
        heatmap_data[i, j] = prob_contact

# 3. Plotting
plt.figure(figsize=(10, 8))
ax = sns.heatmap(heatmap_data, cmap='magma',
                 cbar_kws={'label': 'Probability of Contact P(N_contact >= 1)'},
                 xticklabels=np.round(f_bio_log, 1),
                 yticklabels=np.round(L_log, 1))

ax.invert_yaxis()
plt.xticks(np.arange(0, resolution, 10), np.round(f_bio_log[::10], 1))
plt.yticks(np.arange(0, resolution, 10), np.round(L_log[::10], 1))

plt.xlabel('Biological Bottleneck (Log10 f_bio)', fontsize=14)
plt.ylabel('Communicative Lifespan (Log10 L in years)', fontsize=14)
plt.title('Sensitivity Space: 2026 Spatiotemporal Contact Probability', fontsize=16, pad=20)

# The "Isolation Zone" is now so large the contour might barely show at the top right
plt.contour(heatmap_data, levels=[0.5], colors='white', linestyles='dashed', linewidths=2)
plt.text(5, 20, 'Spatiotemporal Isolation\nAbsolute Dominance', color='white', fontsize=14, alpha=0.9)

plt.tight_layout()
plt.savefig('heatmap.png', dpi=300)
plt.show()
