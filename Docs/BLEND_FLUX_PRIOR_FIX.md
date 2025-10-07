# Flux Parameter Priors

## Overview

Implemented strict one-sided Gaussian priors on flux parameters to enforce physical constraints and proper normalization for relative flux ~1.0.

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

### 2. Negative Source Flux Prior (sigma_fs = 0.05) **NEW**

**Applied when:** `fs < 0`

**Physical reasoning:** Source flux **must be positive** - negative source flux is completely unphysical (can't have negative light emission).

**Penalty:**
```python
if fs < 0:
    lp += -0.5 * (fs / sigma_fs)**2
```

**Examples with sigma_fs = 0.05 (stricter):**
- fs = -0.01: penalty = -0.02 (barely tolerated)
- fs = -0.05: penalty = -0.5 (~61% probability)
- fs = -0.1: penalty = -2.0 (~14% probability)
- fs = -0.2: penalty = -8.0 (~0.03% probability, strongly rejected)

This is **4× stricter** than the blend flux prior (sigma_fs = 0.05 vs sigma_fb = 0.2).

### 3. Excess Baseline Flux Prior (sigma_fs = 0.05) **NEW**

**Applied when:** `fs + fb > 1.0`

**Physical reasoning:** Relative flux is normalized such that baseline flux ≈ 1.0. If `fs + fb` significantly exceeds 1, it indicates:
- Poor normalization/calibration
- Model failure or convergence issues
- Incorrect physical parameters

**Penalty:**
```python
fbaseline = fs + fb
if fbaseline > 1.0:
    lp += -0.5 * ((fbaseline - 1.0) / sigma_fs)**2
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

### Original Problem (Fixed)

## Historical Context

### Original Problem (sigma_fb = 50.0)

The original blend flux prior used `sigma_fb = 50.0`, which was absurdly large for relative flux normalized to ~1.0.

### Why This Was Broken

In the likelihood calculation, observed flux is modeled as:
```
F_observed = Fs * A + FB
```

where:
- `Fs` = source flux
- `A` = magnification 
- `FB` = blend flux

For the GULLS simulations, relative flux is normalized such that the baseline flux (no lensing, A=1) is approximately 1.0.

The prior applied a Gaussian penalty for negative blend flux:
```python
if fb < 0:
    lp += -0.5 * (fb / sigma_fb)**2
```

**With sigma_fb = 50.0:**
- fb = -1.0: penalty = -0.0002 (essentially free!)
- fb = -10.0: penalty = -0.02 (barely noticeable)
- fb = -50.0: penalty = -0.5 (still weak)

This means the prior was **not constraining negative blend at all**, allowing physically absurd values like fb = -10 (which would mean the source is 10× brighter than the entire field baseline).

### Physical Expectation

For dim lenses or photometric calibration issues, small negative blend flux is plausible:
- **Typical range:** -0.2 to +0.5 relative to baseline=1.0
- **Unreasonable:** |fb| > 1.0 (means blend dominates or source is invisible)

## Solution

Changed default `sigma_fb` from **50.0** to **0.2**.

### Impact of New Prior (sigma_fb = 0.2)

**Negative blend flux penalties:**
- fb = -0.05: penalty = -0.03 (allowed, ~97% probability vs peak)
- fb = -0.1: penalty = -0.125 (mildly discouraged, ~88% probability vs peak)
- fb = -0.2: penalty = -0.5 (moderately penalized, ~61% probability vs peak)
- fb = -0.5: penalty = -3.125 (~4% probability vs peak)
- fb = -1.0: penalty = -12.5 (strongly rejected, ~0.0003% probability vs peak)

This allows:
✅ Small negative blend (measurement noise, faint lenses)  
✅ Moderate blend flux (-0.2 to +0.5)  
❌ Large negative blend (unphysical)  
❌ Massive positive/negative blend (|fb| > 1)

## Files Changed

### `/Fit/__init__.py`

**Line 65:** Changed default parameter
```python
# Old:
sigma_fb=50.0,

# New:
sigma_fb=0.2,  # Prior width for negative blend flux (relative to baseline~1)
```

**Lines 95-101:** Updated docstring
```python
sigma_fb : float, optional
    Standard deviation for one-sided Gaussian prior on negative blend flux.
    Applied when fb < 0 to gently penalize large negative blend while allowing
    small negative values (realistic for faint lenses or photometric noise).
    Should be scaled to match relative flux units (typically ~0.1-0.2 for 
    relative flux normalized to 1.0). Default: 0.2
```

### `/m00_unc_check.ipynb`

Marked TODO items as completed:
```markdown
[x] add Fs and FB to the samples (now saved per-observatory via blobs)
[x] fix FB prior (changed sigma_fb from 50 to 0.2)
```

## Usage

The new default will automatically apply to all future MCMC runs. If you need different behavior for specific analyses:

**More conservative (tighter prior):**
```python
fit_obj = Fit(sigma_fb=0.1)  # Penalizes fb < -0.1 more strongly
```

**More permissive (looser prior):**
```python
fit_obj = Fit(sigma_fb=0.5)  # Allows larger negative blend
```

**Effectively disable (for testing):**
```python
fit_obj = Fit(sigma_fb=10.0)  # Very weak prior
```

## Backward Compatibility

⚠️ **This changes the prior distribution** ⚠️

Old runs with `sigma_fb=50.0` effectively had no constraint on negative blend. New runs will properly penalize unphysical values. This means:

1. **Posteriors will differ** for events with significant negative blend
2. **Old results may have been biased** toward unrealistic negative blend
3. **Re-running analysis is recommended** if blend flux is scientifically important

## Testing Recommendations

After this change, you should:

1. ✅ Run a test event and check that FB values are reasonable (-0.2 to +0.5 range)
2. ✅ Check for print statements about fb penalties in the output
3. ✅ Verify that events aren't getting stuck at fb boundaries
4. ✅ Compare posteriors before/after for a few test cases

## Related Issues

- Initial TODO: "fix FB prior (50 is insane)" 
- Flux parameters now saved via blobs (per-observatory)
- Future work: Consider implementing a proper asymmetric prior (allow positive blend more freely than negative) (It's probably fine how it is; penalty of sig = 0.05 for FS < 0 and FB+FS > 1 and penalty with sig = 0.2 for FB < 0)
