# MMA3001 — One-hour indoor CO₂ forecasting

An individual numerical methods / machine learning engineering project using Monash-supplied smart-building environmental measurements. Compare a persistence baseline, ordinary least squares linear regression and decision-tree regression. Predict CO₂ (ppm) **one hour ahead** using the current and previous four 15-minute CO₂ bins. A chronological 60/20/20 split with purged boundaries separates training, model selection and final testing.

## Quick start

1. Install Python 3.10+ and create an environment: `python -m venv .venv`.
2. Activate it: macOS/Linux `source .venv/bin/activate`; Windows ` .venv\Scripts\activate`.
3. Run `python -m pip install -r requirements.txt`.
4. Place the university-supplied environmental ZIP in `data/raw/`. **Do not commit raw data without checking its licence/permissions.**
5. Run `python -m src.pipeline --zip "data/raw/5EnvSensor_MayToDec2024_180kRows (2)(2).zip"` (adjust to your actual filename).
6. Run tests: `python -m pytest -v > test_report.txt` (to see output on screen as well, run `python -m pytest -v`).
7. Generate HTML API documentation: `python -m pdoc src -o docs/api`.
8. Review `outputs/results.json`, `outputs/validation_metrics.csv`, and `outputs/test_predictions.png`.

## Reproducibility and design

- Environmental CSV is embedded inside a ZIP; its `jsondata` field contains nested measurement records. The code extracts the `Carbon dioxide` variable and its timestamp.
- The 250–10,000 ppm filter is a **broad engineering plausibility screen**, not sensor calibration; evaluate its sensitivity before making strong claims.
- The most frequently sampled environmental sensor is selected by default. This is a *data availability* choice, not a claim of building representativeness.
- 15-minute bins are median-aggregated. Missing bins are preserved; complete lag/target examples are selected. No test information is used to impute features.
- The target is one hour ahead, not current CO₂. Persistence is the current value. All methods use only available historical readings.
- The validation set selects among baseline, linear regression and three decision-tree depths. The held-out test is evaluated once after selection.
- Outputs include error in ppm (MAE, RMSE), training time and a visual comparison. This is an observational forecasting study, not a ventilation-control system.

## Repository structure

- `src/pipeline.py`: extraction, preparation, model comparison, evaluation, plots.
- `tests/test_pipeline.py`: automated correctness and edge-case tests.
- `docs/api/`: generated HTML source-code documentation (commit generated files).
- `report/`: project report and presentation preparation.
- `test_report.txt`: captured pytest report (commit after generating).
- `outputs/`: generated results (not committed by default; select reproducible summary figures for report).

## Limitations and responsible use

Potential limitations: clock or calibration errors, sensor selection bias, nonstationary conditions, missing intervals, seasonal shift, and limited physical interpretation. Occupancy mapping is intentionally excluded because the supplied workbook labels the occupancy sheet as incorrect. Never infer occupancy causation from this analysis. Validate on a full data run before interpreting results.

## AI use

AI assisted with project scaffolding, implementation and initial documentation. The student must inspect outputs, run tests, confirm dataset permissions, make/defend engineering choices, maintain authentic commit history and write an accurate personal AI reflection. See `report/AI_REFLECTION_TEMPLATE.md`.
