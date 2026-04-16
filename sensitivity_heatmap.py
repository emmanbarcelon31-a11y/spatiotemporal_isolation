import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import quad

# ==========================================
# --- Evolutionary Timeline Setting ---
# ==========================================
TARGET_EVOLUTION_GYR = 8.0

# Parameter Space Setup
resolution = 50
f_bio_log = np.linspace(-8, -3, resolution)
L_log = np.linspace(3, 7, resolution)
heatmap_data = np.zeros((resolution, resolution))

# The Spatiotemporal Math
t_present = 13.6e9

# --- Dynamic Hypoexponential Scaling ---
def generate_lambda_rates(target_gyr):
    base_times_gyr = np.array([0.5, 0.7, 0.9, 1.1, 1.3])
    scaling_factor = target_gyr / np.sum(base_times_gyr)
    scaled_times_yr = (base_times_gyr * scaling_factor) * 1e9
    return 1.0 / scaled_times_yr

lambda_rates = generate_lambda_rates(TARGET_EVOLUTION_GYR)
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

# Spatial Constants
N_safe_avg = 5e8
V_GHZ = 2.5e11

# Calculate E_present for the new target timeline
E_present, _ = quad(integrand, 0, t_present)

# Calculate NEW algebraic threshold for the text label
# N_contact = N_safe * f_bio * E_present * L * P_spatial = 1
# P_spatial for 1000ly radius is (4/3 * pi * 1000^3) / 2.5e11
V_shell_fixed = (4/3) * np.pi * (1000**3)
P_spatial_fixed = V_shell_fixed / V_GHZ
new_threshold = 1 / (N_safe_avg * E_present * P_spatial_fixed)

for i, l_val in enumerate(L_log):
    for j, f_val in enumerate(f_bio_log):
        L = 10 ** l_val
        f_bio = 10 ** f_val
        N_concurrent = N_safe_avg * f_bio * E_present * L
        R_eff = np.minimum(L, 1000)
        V_shell = (4/3) * np.pi * (R_eff**3)
        P_spatial = V_shell / V_GHZ
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
plt.title(f'Sensitivity Space ({TARGET_EVOLUTION_GYR} Gyr Evolution)', fontsize=16, pad=20)

plt.contour(heatmap_data, levels=[0.5], colors='white', linestyles='dashed', linewidths=2)
plt.text(5, 32, 'Spatiotemporal Isolation\nAbsolute Dominance', color='white', fontsize=12, alpha=0.8)
plt.text(5, 29, f'Contact Threshold: f_bio x L ≈ {new_threshold:,.0f}', color='white', fontsize=11, alpha=0.9, style='italic')

plt.tight_layout()
plt.show()
