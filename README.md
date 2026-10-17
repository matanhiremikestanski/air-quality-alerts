# Urban Air-Quality Anomaly Alert Service (Practical 4)

## Contents
| File | Purpose |
|---|---|
| `pipeline.py` | Loader, causal features, Isolation Forest + LOF detector, alert rule, explanations, evaluation helpers, `AlertService` |
| `train.py` | Fit, calibrate, save `models/detector.joblib` and `data/history.csv` |
| `api.py` | FastAPI: `POST /score`, `GET /health` (docs at `/docs`) |
| `app.py` | Streamlit dashboard: severity, explanation, 72 h window plot |
| `AirQuality_Anomaly_Notebook.ipynb` | Analysis, model comparison, validation, sample alerts |
| `tools/make_synthetic.py` | Smoke-test data only; never use for results |

## Run
```bash
pip install -r requirements.txt
# download AirQualityUCI.csv from https://archive.ics.uci.edu/dataset/387/air+quality into data/
python train.py data/AirQualityUCI.csv
jupyter nbconvert --to notebook --execute AirQuality_Anomaly_Notebook.ipynb --inplace   # produces evidence outputs
uvicorn api:app --port 8000        # API
streamlit run app.py               # dashboard
```
Deploy: push this folder (including `models/` and `data/history.csv`) to GitHub, then Streamlit Community Cloud (main file `app.py`) or Render/Railway (`uvicorn api:app --host 0.0.0.0 --port $PORT`). Put the public URL in your submission.

Example request:
```bash
curl -X POST localhost:8000/score -H "Content-Type: application/json" -d '{"timestamp":"2005-03-01 18:00:00","values":{"PT08.S1(CO)":1300,"PT08.S2(NMHC)":1000,"PT08.S3(NOx)":900,"PT08.S4(NO2)":1500,"PT08.S5(O3)":1100,"T":14,"RH":45,"AH":0.8}}'
```

## Design decisions
* **Missing values:** `-200` -> NaN, hourly grid via `asfreq("h")`. Hours with missing inputs/history are `NO_DATA`, never imputed into a score, so a logger outage cannot masquerade as pollution.
* **Inputs:** 5 metal-oxide sensors + T, RH, AH. Reference analysers (GT columns) are excluded from the model and used only for validation. NMHC(GT) is ~90 % missing in the real file and is not used.
* **Drift and seasonality:** features are deviations from the sensor's own trailing 24 h and 168 h medians, plus hour/weekday cycles. All windows end at the current hour (the notebook asserts causality by truncating future data).
* **Models:** Isolation Forest (global isolation) and LOF with `novelty=True` (local density). They capture different anomaly types; requiring both to agree removes most single-model false positives. Scores are converted to percentiles of an out-of-sample calibration month so thresholds mean "rarer than X % of normal hours".
* **Alert rule:** WATCH both >= p99; ALERT both >= p99.5 and persistent (2 of last 3 h); CRITICAL both >= p99.9 and >= 3 sensors moved together; SENSOR_FAULT flatline (>= 6 identical hours) routed to maintenance; NO_DATA. Thresholds are constructor arguments.
* **Event vs fault heuristic:** several sensors moving together suggests a real change in air; one sensor moving alone suggests a fault. This is a hypothesis for the reviewer, not a diagnosis.
* **Explanations:** occlusion. Each variable's features are replaced by training medians; the drop in the ensemble score is that variable's contribution. Output lists the top variables with current value, deviation from the 24 h median and 3 h change.

## Validation rationale (labels are limited)
1. **Proxy rule:** hours where a reference analyser exceeds its training-period 95th percentile. Reported as precision, recall and *lift over base rate*, not as accuracy, because proxy events include ordinary rush-hour peaks and exclude faults.
2. **Visual inspection** of the top-ranked windows.
3. **Stability:** rolling-origin refits (90 d train, 28 d calibrate, 28 d test). Check that flag rates stay near the calibrated 1 %, and that IF and LOF overlap (Jaccard) consistently.
4. **False-alarm load:** alerts per week under the rule versus a naive single-model top-1 % rule.
Interpret the numbers from the real data in the notebook; do not quote numbers from the synthetic smoke test.

## Monitoring procedure
* **Daily:** review ALERT/CRITICAL queue; confirm with a calibrated instrument, nearby reference station or site check; record outcome (true event / sensor fault / false alarm). SENSOR_FAULT goes to maintenance.
* **Weekly:** alerts per week by severity; share of NO_DATA hours; flag rate per sensor. Targets: WATCH+ <= a staffing-defined cap, NO_DATA small.
* **Monthly:** compare p99 flag rate to the 1 % design value. Persistently > 2 % or < 0.3 % means drift/season shift: recalibrate (refit on the latest 90 d, calibrate on the next 28 d), versioning the model.
* **Feedback:** the review log becomes the verified labels the dataset lacks, enabling supervised evaluation later.
* **Change control:** re-run the notebook validation before replacing the deployed model; keep the previous model for rollback.

## Responsible use and uncertainty
* The sensors are low-cost, cross-sensitive to temperature/humidity, and drift; the model learns what is *unusual*, not what is *harmful*. No health thresholds are applied.
* **Missed events** (false negatives) can leave people exposed to pollution without warning; **false alarms** cause alert fatigue, which also leads to missed events. The p99 design trades these off and the numbers will not be perfect: a slow, sustained rise that the 168 h baseline absorbs can be missed.
* One year of data covers each season once; the model has seen no full seasonal cycle twice, and the real events in it are unverified.
* Therefore alerts are a **triage aid for human reviewers**. They must **not** replace calibrated monitoring equipment, regulatory measurements or public health decisions.
