# Global comparison of MCMC vs Fisher uncertainties
import matplotlib.pyplot as plt
import numpy as np

# Collect all uncertainty pairs across all events
fisher_sigmas = []
mcmc_sigmas = []
hdf5_sigmas = []
prior_sigmas = []
param_names = []
event_names = []

for prm_file, event_data in event_list.items():
    event_name = f"{event_data['field']}_{event_data['subrun']}_{event_data['event_id']}"
    parameter_labels = event_data['parameter_labels']
    fisher_unc = event_data.get('fisher_uncertainties', pd.Series())
    hdf5_unc = event_data.get('hdf5_uncertainties', pd.Series())
    mcmc_unc = event_data.get('mcmc_uncertainties', {})
    prior_unc = event_data.get('prior_uncertainties', {})
    
    for label in parameter_labels:
        label_key = f"log_{label}_err" if label in ["s", "q", "rho", "tE"] else f"{label}_err"
        
        # Get Fisher uncertainty
        if label_key in fisher_unc.index and not np.isnan(fisher_unc[label_key]):
            fisher_sig = fisher_unc[label_key]
            
            # Get MCMC uncertainty (average of + and -)
            if label_key in mcmc_unc:
                p50, err_minus, err_plus = mcmc_unc[label_key]
                mcmc_sig = (err_minus + err_plus) / 2.0
                
                # Store the pair
                fisher_sigmas.append(fisher_sig)
                mcmc_sigmas.append(mcmc_sig)
                param_names.append(label_key.replace('_err', ''))
                event_names.append(event_name)
                
                # Also get HDF5 and Prior if available
                if label_key in hdf5_unc.index and not np.isnan(hdf5_unc[label_key]):
                    hdf5_sigmas.append(hdf5_unc[label_key])
                else:
                    hdf5_sigmas.append(np.nan)
                
                if label_key in prior_unc:
                    prior_sigmas.append(prior_unc[label_key])
                else:
                    prior_sigmas.append(np.nan)

fisher_sigmas = np.array(fisher_sigmas)
mcmc_sigmas = np.array(mcmc_sigmas)
hdf5_sigmas = np.array(hdf5_sigmas)
prior_sigmas = np.array(prior_sigmas)

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Scatter plot: MCMC vs Fisher
ax = axes[0, 0]
ax.scatter(fisher_sigmas, mcmc_sigmas, alpha=0.5, s=20, c='black')
# Add 1:1 line
lim_max = max(np.max(fisher_sigmas), np.max(mcmc_sigmas))
ax.plot([0, lim_max], [0, lim_max], 'r--', linewidth=2, label='1:1 line')
ax.set_xlabel('Fisher $\\sigma$', fontsize=12)
ax.set_ylabel('MCMC $\\sigma$', fontsize=12)
ax.set_title('MCMC vs Fisher Uncertainties', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')

# 2. Ratio histogram
ax = axes[0, 1]
ratios = mcmc_sigmas / fisher_sigmas
ax.hist(ratios, bins=30, color='black', alpha=0.7, edgecolor='black')
ax.axvline(1.0, color='r', linestyle='--', linewidth=2, label='MCMC = Fisher')
median_ratio = np.median(ratios)
ax.axvline(median_ratio, color='blue', linestyle='-', linewidth=2, label=f'Median = {median_ratio:.2f}')
ax.set_xlabel('MCMC $\\sigma$ / Fisher $\\sigma$', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Ratio Distribution', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 3. Cumulative fractional uncertainty comparison
ax = axes[1, 0]
sorted_ratios = np.sort(ratios)
cumulative = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
ax.plot(sorted_ratios, cumulative, color='black', linewidth=2, label='MCMC/Fisher')
ax.axvline(1.0, color='r', linestyle='--', linewidth=2, label='Equal uncertainties')
ax.axhline(0.5, color='gray', linestyle=':', linewidth=1)
ax.set_xlabel('MCMC $\\sigma$ / Fisher $\\sigma$', fontsize=12)
ax.set_ylabel('Cumulative Fraction', fontsize=12)
ax.set_title('Cumulative Distribution of Uncertainty Ratios', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Add quartile annotations
q25 = np.percentile(ratios, 25)
q50 = np.percentile(ratios, 50)
q75 = np.percentile(ratios, 75)
ax.text(0.05, 0.95, f'Q1: {q25:.2f}\\nMedian: {q50:.2f}\\nQ3: {q75:.2f}', 
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. Parameter-wise comparison (grouped by parameter type)
ax = axes[1, 1]
# Group by parameter name (not by event)
unique_params = sorted(set(param_names))
param_median_ratios = []
for param in unique_params:
    mask = np.array(param_names) == param
    param_ratios = ratios[mask]
    param_median_ratios.append(np.median(param_ratios))

x_pos = np.arange(len(unique_params))
colors = ['red' if r < 0.8 else 'orange' if r < 1.2 else 'green' for r in param_median_ratios]
bars = ax.bar(x_pos, param_median_ratios, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(1.0, color='black', linestyle='--', linewidth=2)
ax.set_xticks(x_pos)
ax.set_xticklabels(unique_params, rotation=45, ha='right')
ax.set_ylabel('Median MCMC $\\sigma$ / Fisher $\\sigma$', fontsize=12)
ax.set_title('Parameter-wise Median Ratios', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir + 'global_uncertainty_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\\n=== Global Uncertainty Comparison Summary ===")
print(f"Number of parameter measurements: {len(ratios)}")
print(f"\\nMCMC/Fisher Ratio Statistics:")
print(f"  Mean:   {np.mean(ratios):.3f}")
print(f"  Median: {np.median(ratios):.3f}")
print(f"  Std:    {np.std(ratios):.3f}")
print(f"  Min:    {np.min(ratios):.3f}")
print(f"  Max:    {np.max(ratios):.3f}")
print(f"\\n  Q1:     {np.percentile(ratios, 25):.3f}")
print(f"  Q3:     {np.percentile(ratios, 75):.3f}")
print(f"\\nFraction where MCMC < Fisher: {np.sum(ratios < 1.0) / len(ratios):.1%}")
print(f"Fraction where MCMC > Fisher: {np.sum(ratios > 1.0) / len(ratios):.1%}")
print(f"\\nParameter-wise median ratios:")
for param, ratio in zip(unique_params, param_median_ratios):
    status = "✓" if 0.8 < ratio < 1.2 else "⚠"
    print(f"  {status} {param:12s}: {ratio:.3f}")
