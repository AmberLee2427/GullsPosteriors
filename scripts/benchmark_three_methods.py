#!/usr/bin/env python
"""
Run three approaches (Dynesty, fast-emcee, Laplace) on a small list of events
and print wall-times plus basic covariance diagnostics.
"""
import subprocess, time, json, shutil, sys
from pathlib import Path
import os

EVENT_DIR = Path("../overguide_m00_Fisher/")  # sampler script runs on all events in the directory

SAMPLER_SCRIPT = Path("../gulls_post_emcee_adaptive_BI.py")  # your main driver

TIMINGS = {}

# ----------------------------------------------------------------------
# helper
def run(cmd, label):
    t0 = time.time()
    print(f"\n→  {label}: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    dt = time.time() - t0
    TIMINGS[label] = TIMINGS.get(label, 0.0) + dt
# ----------------------------------------------------------------------

path = EVENT_DIR.as_posix() + "/"       # gulls_post_emcee_adaptive_BI expects a trailing /

# 1. Adaptive burn-in  with emcee
run([sys.executable, SAMPLER_SCRIPT,
        "3", path, "-s", "emcee", "-noLOM", "-t", "0", 
        "-nbimin", "500", "-nbimax", "1000", "-nbistep", "200",
        "-n", "1000", "-nstep", "100",
        "-prior", "normal-unit-cube", "-adapt",
        "-f", "icpf"], "adaptive burn-in")

os.remove(f"{EVENT_DIR}/emcee_run_list.txt")
os.rename(f"{EVENT_DIR}/posteriors/", f"{EVENT_DIR}/posteriors_full_adaptive_burnin_emcee/")

# 1. Dynesty
run([sys.executable, SAMPLER_SCRIPT,
        "3", path, "-s", "dynesty", "-noLOM", "-t", "0", 
        "-nbimin", "500", "-nbimax", "1000", "-nbistep", "200",
        "-n", "1000", "-nstep", "100",
        "-prior", "normal-unit-cube", "-f", "itpf"], f"dynesty")

os.remove(f"{EVENT_DIR}/emcee_run_list.txt")
os.rename(f"{EVENT_DIR}/posteriors/", f"{EVENT_DIR}/posteriors_dynesty/")

# 2. fast-emcee  (half burn-in, larger priors: uses -f flag to skip plots)
run([sys.executable, SAMPLER_SCRIPT,
        "3", path, "-s", "emcee", "-f", "n", "-noLOM", "-t", "0", 
        "-n", "1000", "-nstep", "100",
        "-prior", "normal", "-f", "icpf"], f"fast_emcee")

os.remove(f"{EVENT_DIR}/emcee_run_list.txt")
os.rename(f"{EVENT_DIR}/posteriors/", f"{EVENT_DIR}/posteriors_fast_emcee/")

# 3. Laplace
run([sys.executable, "scripts/laplace_estimate.py",
        path], f"laplace")

# ----------------------------------------------------------------------
print("\n======== Timing summary ========")
for k, v in TIMINGS.items():
    print(f"{k:25s}: {v:6.1f} s")