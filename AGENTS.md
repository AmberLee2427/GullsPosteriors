## Repository Guidelines for Codex

- Profanity and informal language in comments is acceptable and intentional.
- Do not change or remove swearing or other unprofessional phrases from the code or documentation unless explicitly requested by the developer.
- Focus code reviews on functional bugs or missing features rather than tone.
- Be very intentional about error handling. It is better to have code that breaks that outputs that are wrong. This is scientific code not software engineering.

## TODO:

* add final plots with samples <- these are still mid, but we have a good versions of the corner plots in the post-processing notebook (`GullsPosteriors/unc_check.ipynb`)
* fix caustic plot bounds - can't see shit
* fix chain plot:
  - log parametrs don't seem to be working correctly 
  - always plotting in unit cube space, even when using physical paramters
  ^ Did we fix this already?
* make sure n is always bigger than n_burnin, or n_tot = n + n_burnin
* ~~help text for the cli~~ ✅ Added comprehensive help with examples and grouped arguments
* ~~log to file and make sure all changable settings are recorded~~ ✅ Added .prm YAML logging
* ~~do we need to do something about there being more than 1 obs group?~~ <- No.
* ~~Collect peaks + unc for Sean~~ ✅ Done in `collected_lightcurves/` and sent on Slack
* ~~The VBM logging doesn't seem to be working. We need to look in to this and fix it. Before we added the try/except and timepout we were getting failures on the order of the number of events or more, so getting nothing in the logging arrays for 3 full mass bins seems unlikely.~~
* ~~Farzahna the bloody pain in the butt gave us the wrong event list. I've relaces the event_list in the data folders, but everything needs to be rerun. Blessing in disguise, I guess, considering the VBM logging wasn't working.~~ - Either VBM logging still isn't working or we aren't getting any errors. Either way, I'm done caring.
* ~~Fix race condition (?) on first event in the slurm array~~ - I tried, but something wierd still seems to be happening
* Run all bins
* Ignore fixing the main code. Let's just try to get the lightcurves working in the post-processing scripts. `unc_check_part4_lightcurves.py`