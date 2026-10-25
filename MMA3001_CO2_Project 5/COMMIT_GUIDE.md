# Suggested authentic GitHub commit sequence

**Only commit work that is actually present, tested or reviewed. Do not manufacture earlier development history.**

1. `chore: initialise reproducible project structure` — Added README, Python dependencies, ignore rules and directory organisation.
2. `feat: parse CO2 observations from nested sensor JSON` — Implemented archive reading, timestamp conversion and physical plausibility checks.
3. `feat: construct leakage-aware time-series examples` — Added resampling, lag features, future target and chronological partitions.
4. `feat: compare persistence linear and tree regression` — Implemented baselines, candidate models and validation-based selection.
5. `test: cover target alignment metrics and invalid inputs` — Added automated unit tests for data preparation and numerical correctness.
6. `feat: export held-out metrics and prediction figure` — Added reproducible test evaluation, CSV/JSON results and visualisation.
7. `docs: generate HTML API reference and test report` — Generated and checked source documentation and captured pytest results.
8. `docs: add engineering report and presentation materials` — Added verified findings, limitations, AI reflection and demonstration notes.

For each commit: `git add <files>` then `git commit -m "message" -m "description"` and `git push`. Use only the messages corresponding to work completed in that commit. The initial package may already contain several features, so a single truthful initial import commit is preferable to pretending the features were developed separately.
