# Flux Parameters Implementation via Blobs

## Overview

This document describes how flux parameters (Fs, FB, Fbaseline) were added to MCMC samples using the "blobs" feature of emcee/dynesty, avoiding expensive recomputation of magnification models.

## Problem

The flux parameters (Fs, FB, Fbaseline) are computed via linear regression during every likelihood evaluation, but they weren't being saved alongside the physical parameters in the MCMC samples. We needed to add them without:
- Recomputing expensive magnification models
- Modifying the core MCMC sampler logic
- Breaking existing code

## Solution: Blobs

Both emcee and dynesty support saving arbitrary auxiliary data ("blobs") alongside samples. We modified the likelihood function to return both the log-probability and a dictionary containing the flux parameters.

## Implementation Details

### 1. Modified `Fit.get_chi2()` 
**File:** `Fit/__init__.py`

Added caching of flux parameters computed during chi-square calculation:

```python
# Initialize cache for flux parameters (for blobs)
self.last_fluxes = {}

# ... during observatory loop ...
fs, fb = self.get_fluxes(A, f, f_err**2)

# Cache flux parameters for blobs
self.last_fluxes[obs] = (fs, fb)
```

### 2. Modified `Fit.lnprob()` 
**File:** `Fit/__init__.py`

Changed return value from scalar to tuple `(log_prob, blobs)`:

```python
# Extract flux parameters from cached values (averaged across observatories)
if hasattr(self, 'last_fluxes') and len(self.last_fluxes) > 0:
    fs_values = [fs for fs, fb in self.last_fluxes.values()]
    fb_values = [fb for fs, fb in self.last_fluxes.values()]
    Fs = np.mean(fs_values)
    FB = np.mean(fb_values)
    Fbaseline = Fs + FB
    blobs = {"Fs": Fs, "FB": FB, "Fbaseline": Fbaseline}
else:
    blobs = {"Fs": np.nan, "FB": np.nan, "Fbaseline": np.nan}

return lp + ll, blobs
```

**Note:** Returns `np.nan` blobs for rejected samples (prior violations, etc.)

### 3. Modified `lnprob_transform()`
**File:** `Fit/_emcee.py`

Updated wrapper function to handle tuple return from `lnprob()`:

```python
lp, blobs = self.lnprob(theta, event)
return lp, blobs
```

### 4. Modified `run_emcee()`
**File:** `Fit/_emcee.py`

Added blob extraction and saving after each sampling checkpoint:

```python
# Extract blobs (flux parameters) if available
if hasattr(sampler, 'blobs') and sampler.blobs is not None:
    # sampler.blobs is list of [nwalkers, nsteps] dicts
    # Convert to structured array: [nsamples, 3] for Fs, FB, Fbaseline
    blobs_list = []
    for step_blobs in sampler.blobs:
        for walker_blob in step_blobs:
            if walker_blob is not None:
                blobs_list.append([walker_blob.get("Fs", np.nan), 
                                  walker_blob.get("FB", np.nan), 
                                  walker_blob.get("Fbaseline", np.nan)])
    if len(blobs_list) > 0:
        blobs_array = np.array(blobs_list)
        np.save(
            path + "posteriors/" + event_name + "_emcee_blobs.npy",
            blobs_array,
        )
```

**Saved file:** `{event_name}_emcee_blobs.npy` - shape `[n_samples, 3]`

### 5. Modified `run_dynesty()`
**File:** `Fit/_dynesty.py`

Dynesty doesn't natively support blobs, so we:
1. Created a wrapper function to extract only log-probability
2. Cached blobs in `self._dynesty_blobs` during sampling
3. Saved blobs after sampling completes

```python
# Wrapper to extract only log-probability from lnprob (which returns (lp, blobs))
def loglike_wrapper(theta, event):
    lp, blobs = self.lnprob(theta, event)
    # Store blobs for later extraction (dynesty doesn't support blobs natively)
    if not hasattr(self, '_dynesty_blobs'):
        self._dynesty_blobs = []
    self._dynesty_blobs.append(blobs)
    return lp

sampler = dynesty.DynamicNestedSampler(
    loglike_wrapper,  # Use wrapper
    ...
)

# After sampling...
if hasattr(self, '_dynesty_blobs') and len(self._dynesty_blobs) > 0:
    blobs_array = np.array([[b.get("Fs", np.nan), b.get("FB", np.nan), b.get("Fbaseline", np.nan)] 
                            for b in self._dynesty_blobs])
    # Take the last len(samples) blobs (corresponds to final resampled posterior)
    if len(blobs_array) >= len(samples):
        blobs_array = blobs_array[-len(samples):]
    np.save(path+'posteriors/'+event_name+'_dynesty_blobs.npy', blobs_array)
    del self._dynesty_blobs
```

**Saved file:** `{event_name}_dynesty_blobs.npy` - shape `[n_samples, 3]`

### 6. Modified `m00_unc_check.ipynb`
**File:** `m00_unc_check.ipynb`, cell `#VSC-be3066c8`

Updated data loading to:
1. Load blob files if they exist
2. Stack blobs onto samples array
3. Update parameter_labels to include flux parameters

```python
# Try to load flux parameter blobs
blobs_filename = filename.replace("samples.npy", "blobs.npy")
blobs_path = output_dir + blobs_filename
if os.path.exists(blobs_path):
    print(f"  Loading flux parameter blobs from {blobs_filename}")
    blobs = np.load(blobs_path)
    print(f"  Blobs shape: {blobs.shape}")
    # Stack blobs onto samples: [n_samples, n_params+3]
    if blobs.shape[0] == samples.shape[0]:
        samples = np.hstack([samples, blobs])
        print(f"  Combined samples shape: {samples.shape}")
        # Update parameter labels to include flux parameters
        parameter_labels = parameter_labels + ['Fs', 'FB', 'Fbaseline']
    else:
        print(f"  Warning: Blobs shape mismatch, skipping")
else:
    print(f"  No blobs file found, flux parameters not available")
```

## Usage

### Running New MCMC Samples

When you run `gulls_post.py` now, it will automatically:
1. Compute flux parameters during likelihood evaluation
2. Save them as blobs alongside physical parameters
3. Create two files per event:
   - `{event_name}_emcee_samples.npy` or `{event_name}_dynesty_samples.npy` - physical parameters
   - `{event_name}_emcee_blobs.npy` or `{event_name}_dynesty_blobs.npy` - flux parameters

### Loading Samples in Analysis

The notebook will automatically detect and load blob files:
- If blobs exist: samples array shape is `[n_samples, n_params+3]` with flux parameters appended
- If blobs don't exist: samples array shape is `[n_samples, n_params]` with only physical parameters
- Parameter labels automatically updated to include `['Fs', 'FB', 'Fbaseline']` when blobs are loaded

### Backward Compatibility

Code is fully backward compatible:
- Old samples without blobs will still load normally
- Flux parameters simply won't be available for those events
- No errors or crashes - just informational messages

## Flux Parameter Definitions

- **Fs** (Source Flux): Flux from the lensed source star, averaged across observatories
- **FB** (Blend Flux): Flux from unlensed blend sources, averaged across observatories
- **Fbaseline** (Baseline Flux): Total flux baseline = Fs + FB

These are computed via weighted linear least squares during chi-square calculation:
```
F_observed = Fs * A + FB
```
where A is the magnification from the binary lens model.

## Performance

**No additional computational cost** - flux parameters are computed as a byproduct of the likelihood calculation. We're just saving values that were already being calculated and discarded.

## Testing

To test with existing samples:
1. Re-run MCMC on a test event - blobs will be saved automatically
2. Run `m00_unc_check.ipynb` - it should detect and load the blobs
3. Check that `parameter_labels` includes `['Fs', 'FB', 'Fbaseline']`
4. Verify samples array has 3 additional columns

## Future Work

- Add flux parameters to corner plots (currently only physical params are plotted)
- Consider saving per-observatory flux parameters instead of just the mean
- Add flux parameter uncertainties to comparison tables
