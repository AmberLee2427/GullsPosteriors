# Flux Parameter Priors

## Overview

Implemented three strict one-sided Gaussian priors on flux parameters to enforce physical constraints and proper normalization for relative flux ~1.0.

## Priors Implemented

### 1. Negative Blend Flux Prior (sigma_fb = 0.2)

**Applied when:** `fb < 0`

**Physical reasoning:** Small negative blend flux is plausible due to:
- Photometric calibration issues
- Faint lenses in crowded fields
- Measurement noise

**Penalty:**
```python
if fb < 0:
    lp += -0.5 * (fb / sigma_fb)**2
```

**Examples with sigma_fb = 0.2:**
- fb = -0.05: penalty = -0.03 (allowed, ~97% probability)
- fb = -0.1: penalty = -0.125 (mildly discouraged, ~88% probability)
- fb = -0.2: penalty = -0.5 (moderately penalized, ~61% probability)
- fb = -0.5: penalty = -3.125 (~4% probability)
- fb = -1.0: penalty = -12.5 (strongly rejected)

### 2. Negative Source Flux Prior (sigma_fs = 0.05)

**Applied when:** `fs < 0`

**Physical reasoning:** Source flux **must be positive** - negative source flux is completely unphysical (can't have negative light emission).

**Penalty:**
```python
if fs < 0:
    lp += -0.5 * (fs / self.sigma_fs)**2
```

**Examples with sigma_fs = 0.05 (stricter):**
- fs = -0.01: penalty = -0.02 (barely tolerated)
- fs = -0.05: penalty = -0.5 (~61% probability)
- fs = -0.1: penalty = -2.0 (~14% probability)
- fs = -0.2: penalty = -8.0 (~0.03% probability, strongly rejected)

This is **4× stricter** than the blend flux prior (sigma_fs = 0.05 vs sigma_fb = 0.2).

### 3. Excess Baseline Flux Prior (sigma_fs = 0.05)

**Applied when:** `fs + fb > 1.0`

**Physical reasoning:** Relative flux is normalized such that baseline flux ≈ 1.0. If `fs + fb` significantly exceeds 1, it indicates:
- Poor normalization/calibration
- Model failure or convergence issues
- Incorrect physical parameters

**Penalty:**
```python
fbaseline = fs + fb
if fbaseline > 1.0:
    lp += -0.5 * ((fbaseline - 1.0) / self.sigma_fs)**2
```

**Examples with sigma_fs = 0.05:**
- fbaseline = 1.01: penalty = -0.02 (tiny excess, allowed)
- fbaseline = 1.05: penalty = -0.5 (5% excess, ~61% probability)
- fbaseline = 1.10: penalty = -2.0 (10% excess, ~14% probability)
- fbaseline = 1.20: penalty = -8.0 (20% excess, ~0.03% probability, strongly rejected)

### 4. No Penalty on Positive Values

**Important:** These are **one-sided priors**. There is **no penalty** for:
- Positive blend flux (fb > 0) of any magnitude
- Source flux in range (0, ∞)
- Baseline flux in range (0, 1.0]

The data (likelihood) naturally constrains these values.

## Historical Context

### Original Problem (sigma_fb = 50.0)

The original blend flux prior used `sigma_fb = 50.0`, which was absurdly large for relative flux normalized to ~1.0.

**Example penalties with sigma_fb = 50.0 (TOO WEAK):**
- fb = -1.0: penalty = -0.0002 (no constraint!)
- fb = -10.0: penalty = -0.02 (still barely any penalty!)
- fb = -50.0: penalty = -0.5 (only moderate penalty!)

This provided **no practical constraint** on negative blend flux.

### Fixes Applied (December 2024)

1. Changed `sigma_fb = 50.0 → 0.2` to match relative flux scale
2. Added `sigma_fs = 0.05` for stricter priors on:
   - Negative source flux (fs < 0) - completely unphysical
   - Excess baseline flux (fs + fb > 1.0) - normalization violation

## Implementation Details

All three priors implemented in both LOM and non-LOM branches of `lnprior()` in `Fit/__init__.py`:

```python
# Get flux parameters from last chi2 calculation
fs, fb = self.get_fluxes(A, f, f_err**2)
lp = 0.0

# Negative blend flux (moderate penalty, sigma_fb = 0.2)
if fb < 0:
    lp += -0.5 * (fb / self.sigma_fb)**2
    print(f"fb: {fb}, sigma_fb: {self.sigma_fb}, lp: {lp}")

# Negative source flux (strict penalty, sigma_fs = 0.05)
if fs < 0:
    lp += -0.5 * (fs / self.sigma_fs)**2
    print(f"fs: {fs}, sigma_fs: {self.sigma_fs}, lp: {lp}")

# Excess baseline flux (strict penalty, sigma_fs = 0.05)
fbaseline = fs + fb
if fbaseline > 1.0:
    lp += -0.5 * ((fbaseline - 1.0) / self.sigma_fs)**2
    print(f"fbaseline: {fbaseline}, excess: {fbaseline-1.0:.3f}, sigma_fs: {self.sigma_fs}, lp: {lp}")
```

## Per-Observatory vs. Aggregated

Note that flux parameters are computed and saved **per-observatory** (different filters require different flux values). The priors check each observatory's flux independently.

Typical keys in blobs: `Fs_0`, `FB_0`, `Fbaseline_0`, `Fs_1`, `FB_1`, `Fbaseline_1`, etc.

## Testing

To verify priors are working:
1. Run MCMC with new code
2. Check terminal output for print statements when priors triggered
3. Load blobs in notebook, plot flux posteriors
4. Verify flux values stay within physically reasonable ranges:
   - **fs ≈ 0.5 to 1.5** (positive, near baseline)
   - **fb ≈ -0.2 to 0.5** (small negative allowed, not too positive)
   - **fbaseline ≈ 0.8 to 1.1** (near normalized baseline)

## Backward Compatibility

**Warning:** Old MCMC samples may be biased toward unphysical flux values due to weak/missing priors. Re-running with new priors will produce different (more physically reasonable) posteriors.

## Usage

The new defaults will automatically apply to all future MCMC runs. If you need different behavior:

**More conservative (tighter priors):**
```python
fit_obj = Fit(sigma_fb=0.1, sigma_fs=0.02)  # Stricter constraints
```

**More permissive (looser priors):**
```python
fit_obj = Fit(sigma_fb=0.5, sigma_fs=0.1)  # Allow larger deviations
```

**Effectively disable (for testing):**
```python
fit_obj = Fit(sigma_fb=10.0, sigma_fs=10.0)  # Very weak priors
```

## See Also

- `FLUX_PARAMETERS_IMPLEMENTATION.md` - How flux parameters are computed and saved via blobs
- `Fit/__init__.py` lines 382-402 (LOM) and 437-457 (non-LOM) - Prior implementation
- Original issue: "fix FB prior (50 is insane)" - sigma_fb was absurdly large
