## Repository Guidelines for Codex

- Profanity and informal language in comments is acceptable and intentional.
- Do not change or remove swearing or other unprofessional phrases from the code or documentation unless explicitly requested by the developer.
- Focus code reviews on functional bugs or missing features rather than tone.
- Be very intentional about error handling. It is better to have code that breaks that outputs that are wrong. This is scientific code not software engineering.

## TODO:

* add final plots with samples
* fix caustic plot bounds
* fix chain plot:
  - log parametrs don't seem to be working correctly 
  - always plotting in unit cudem space, even when using physical paramters
* make sure n is always bigger than n_burnin, or n_tot = n + n_burnin
* ~~help text for the cli~~ ✅ Added comprehensive help with examples and grouped arguments
* ~~log to file and make sure all changable settings are recorded~~ ✅ Added .prm YAML logging
* do we need to do something about there being more than 1 obs group?