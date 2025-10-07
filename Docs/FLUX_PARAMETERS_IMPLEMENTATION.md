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
# Extract flux parameters from cached values (per-observatory)
# Each observatory has different filters, so we save them separately
blobs = {}
if hasattr(self, 'last_fluxes') and len(self.last_fluxes) > 0:
    for obs, (fs, fb) in self.last_fluxes.items():
        blobs[f"Fs_{obs}"] = fs
        blobs[f"FB_{obs}"] = fb
        blobs[f"Fbaseline_{obs}"] = fs + fb
else:
    # Return empty dict if no fluxes computed
    pass

return lp + ll, blobs
```

**Note:** Returns empty dict `{}` for rejected samples (prior violations, etc.)

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
if hasattr(sampler, 'blobs') and sampler.blobs is not None and len(sampler.blobs) > 0:
    # sampler.blobs is list of [step][walker] dicts
    # Each dict has keys like: Fs_0, FB_0, Fbaseline_0, Fs_1, FB_1, etc.
    # First, collect all unique keys from all blobs to determine columns
    all_keys = set()
    for step_blobs in sampler.blobs:
        for walker_blob in step_blobs:
            if walker_blob is not None and isinstance(walker_blob, dict):
                all_keys.update(walker_blob.keys())
    
    if len(all_keys) > 0:
        # Sort keys for consistent ordering (Fs_0, FB_0, Fbaseline_0, Fs_1, ...)
        sorted_keys = sorted(all_keys)
        
        # Extract values for each sample
        blobs_list = []
        for step_blobs in sampler.blobs:
            for walker_blob in step_blobs:
                if walker_blob is not None and isinstance(walker_blob, dict):
                    # Extract values in sorted key order, use NaN for missing keys
                    row = [walker_blob.get(key, np.nan) for key in sorted_keys]
                    blobs_list.append(row)
                else:
                    # Rejected sample - all NaN
                    blobs_list.append([np.nan] * len(sorted_keys))
        
        if len(blobs_list) > 0:
            blobs_array = np.array(blobs_list)
            # Save both the array and the column names
            np.save(
                path + "posteriors/" + event_name + "_emcee_blobs.npy",
                blobs_array,
            )
            # Save column names as separate file for easier loading
            np.save(
                path + "posteriors/" + event_name + "_emcee_blobs_keys.npy",
                np.array(sorted_keys),
            )
```

**Saved files:** 
- `{event_name}_emcee_blobs.npy` - shape `[n_samples, n_flux_params]` (3 per observatory)
- `{event_name}_emcee_blobs_keys.npy` - column names like `['Fs_0', 'FB_0', 'Fbaseline_0', ...]`

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
    # Collect all unique keys from all blobs
    all_keys = set()
    for blob in self._dynesty_blobs:
        if blob is not None and isinstance(blob, dict):
            all_keys.update(blob.keys())
    
    if len(all_keys) > 0:
        sorted_keys = sorted(all_keys)
        blobs_list = []
        for blob in self._dynesty_blobs:
            if blob is not None and isinstance(blob, dict):
                row = [blob.get(key, np.nan) for key in sorted_keys]
                blobs_list.append(row)
            else:
                blobs_list.append([np.nan] * len(sorted_keys))
        
        blobs_array = np.array(blobs_list)
        if len(blobs_array) >= len(samples):
            blobs_array = blobs_array[-len(samples):]
        
        np.save(path+'posteriors/'+event_name+'_dynesty_blobs.npy', blobs_array)
        np.save(path+'posteriors/'+event_name+'_dynesty_blobs_keys.npy', np.array(sorted_keys))
    
    del self._dynesty_blobs
```

**Saved files:** 
- `{event_name}_dynesty_blobs.npy` - shape `[n_samples, n_flux_params]`
- `{event_name}_dynesty_blobs_keys.npy` - column names

### 6. Modified `m00_unc_check.ipynb`
**File:** `m00_unc_check.ipynb`, cell `#VSC-be3066c8`

Updated data loading to:
1. Load blob files and their column names if they exist
2. Stack blobs onto samples array
3. Update parameter_labels to include per-observatory flux parameters

```python
# Try to load flux parameter blobs
blobs_filename = filename.replace("samples.npy", "blobs.npy")
blobs_keys_filename = filename.replace("samples.npy", "blobs_keys.npy")
blobs_path = output_dir + blobs_filename
blobs_keys_path = output_dir + blobs_keys_filename

if os.path.exists(blobs_path) and os.path.exists(blobs_keys_path):
    print(f"  Loading flux parameter blobs from {blobs_filename}")
    blobs = np.load(blobs_path)
    blobs_keys = np.load(blobs_keys_path, allow_pickle=True)
    print(f"  Blobs shape: {blobs.shape}")
    print(f"  Blobs columns: {list(blobs_keys)}")
    
    # Stack blobs onto samples: [n_samples, n_params + n_flux_params]
    if blobs.shape[0] == samples.shape[0]:
        samples = np.hstack([samples, blobs])
        print(f"  Combined samples shape: {samples.shape}")
        # Update parameter labels to include flux parameters
        parameter_labels = parameter_labels + list(blobs_keys)
    else:
        print(f"  Warning: Blobs shape mismatch, skipping")
else:
    print(f"  No blobs files found, flux parameters not available")
```

## Usage

### Running New MCMC Samples

When you run `gulls_post.py` now, it will automatically:
1. Compute flux parameters during likelihood evaluation
2. Save them as blobs alongside physical parameters
3. Create files per event:
   - `{event_name}_emcee_samples.npy` or `{event_name}_dynesty_samples.npy` - physical parameters
   - `{event_name}_emcee_blobs.npy` or `{event_name}_dynesty_blobs.npy` - flux parameters (per-observatory)
   - `{event_name}_emcee_blobs_keys.npy` or `{event_name}_dynesty_blobs_keys.npy` - column names

### Loading Samples in Analysis

The notebook will automatically detect and load blob files:
- If blobs exist: samples array shape is `[n_samples, n_params + n_flux_params]` with per-observatory flux parameters appended
- If blobs don't exist: samples array shape is `[n_samples, n_params]` with only physical parameters
- Parameter labels automatically updated to include per-observatory flux parameters like `['Fs_0', 'FB_0', 'Fbaseline_0', 'Fs_1', ...]` when blobs are loaded

### Backward Compatibility

Code is fully backward compatible:
- Old samples without blobs will still load normally
- Flux parameters simply won't be available for those events
- No errors or crashes - just informational messages

## Flux Parameter Definitions

**Per-observatory flux parameters** (where N is the observatory code 0, 1, 2, etc.):

- **Fs_N** (Source Flux): Flux from the lensed source star at observatory N
- **FB_N** (Blend Flux): Flux from unlensed blend sources at observatory N
- **Fbaseline_N** (Baseline Flux): Total flux baseline at observatory N = Fs_N + FB_N

These are computed via weighted linear least squares during chi-square calculation:
```
F_observed = Fs * A + FB
```
where A is the magnification from the binary lens model.

**Why per-observatory?** Each observatory uses different filters (W146, Z087, K213), so flux values are **not comparable** across observatories and should not be averaged. Saving them separately preserves the physical meaning.

## Performance

**No additional computational cost** - flux parameters are computed as a byproduct of the likelihood calculation. We're just saving values that were already being calculated and discarded.

## Testing

To test with existing samples:
1. Re-run MCMC on a test event - blobs will be saved automatically (both .npy and _keys.npy files)
2. Run `m00_unc_check.ipynb` - it should detect and load the blobs
3. Check that `parameter_labels` includes per-observatory flux parameters like `['Fs_0', 'FB_0', 'Fbaseline_0', ...]`
4. Verify samples array has additional columns (3 per observatory)

## Future Work

- Add per-observatory flux parameters to corner plots (currently only physical params are plotted)
- Add flux parameter uncertainties to comparison tables
- Consider adding flux parameter priors to the sampling (DONE)
