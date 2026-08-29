# Predictive Maintenance: Aircraft Engine RUL Estimation

## Why Predictive Maintenance

1. **Avoiding the financial catastrophe of unplanned downtime (AOG):** In aviation, when an engine fails unexpectedly and an aircraft is grounded (a state known as Aircraft on Ground), the cost to the airline ranges between $10,000 and $150,000 per day per aircraft (flight delays, passenger compensation, and schedule disruption). Predictive maintenance reduces this unplanned downtime by 30% to 50%.
2. **Stopping the waste of Remaining Useful Life (RUL):** Traditional scheduled (preventive) maintenance forces engineers to replace parts or overhaul engines after a fixed number of flight hours, even if the part is still in excellent condition and could keep operating safely. This wastes roughly 15% to 20% of the useful life of components that can cost millions of dollars. This project computes RUL accurately in order to extract that wasted 20% down to the last safe operating cycle.
3. **Reducing overall maintenance cost:** Companies that apply machine learning models to predict engine degradation reduce maintenance costs by 10% to 20%. The reason is simple: maintenance is performed only on the component that is actually about to fail, and spare parts are ordered just-in-time instead of freezing millions of dollars in "just in case" inventory.

## Data Source & Scope

The project uses the **N-CMAPSS (NASA Commercial Modular Aero-Propulsion System Simulation, 2nd generation)** turbofan engine degradation dataset, developed collaboratively by NASA PCoE, ETH Zurich, and PARC.

Dataset link: https://www.kaggle.com/datasets/bishals098/nasa-cmapss-2-engine-degradation

The dataset provides run-to-failure trajectories for a fleet of aircraft engines, including multivariate sensor readings, operating condition variables (altitude, Mach number, throttle resolver angle), flight class labels, and auxiliary/virtual sensor signals, in addition to the true RUL for each cycle in the development (training) set.

## Problem Framing

### Problem Type: Regression

This problem is framed as a **Regression problem**, not Classification. The target (RUL) is a **discrete count quantity** with a meaningful order and distance between values — the difference between RUL=44 and RUL=45 is small and practically insignificant, while the difference between RUL=5 and RUL=95 is large and critical. Treating this as Multiclassification (100 separate classes) would discard this ordinal relationship and weaken the model's ability to leverage the closeness between values.

### Target Definition: Piecewise Linear RUL

An engine remains in a stable, healthy condition for a long period at the start of its life before actual degradation becomes visible in sensor readings. Training the model on raw, unclipped RUL values (which can reach hundreds of cycles early in an engine's life) forces the model to try to distinguish between large values (e.g., RUL=300 vs. RUL=280) that carry no practical meaning, which dilutes learning.

**Solution: clip the RUL to a maximum threshold `R_early` during training only:**

```
RUL_train = min(actual_RUL, R_early)
```

`R_early` is the threshold above which the engine is considered "fully healthy," with no need to distinguish between values beyond it. While values of 125–130 are commonly cited in C-MAPSS literature, direct inspection of this dataset (`train_df["RUL"].max()`) confirmed that RUL is already pre-clipped at **99**. Accordingly, `R_early = 99` is adopted for this project, and no additional clipping is required during training since the source data is already bounded.

### Output Constraints

A regression model has no built-in bounds, so it may output values outside the valid range (negative, or above the maximum threshold). This is handled post-prediction:

```
predicted_RUL = max(0, predicted_RUL)
```

Values above `R_early` are not a real concern, since they simply indicate the engine is still far from failure.

## Success Metrics

### 1. RMSE (Root Mean Squared Error)

The primary error metric, measured in engine cycles, penalizing large errors more heavily than small ones:

```
RMSE = sqrt( (1/n) * Σ (RUL_predicted_i - RUL_actual_i)^2 )
```

### 2. PHM08 Scoring Function

RMSE treats an early prediction and a late prediction of equal magnitude as equally bad. In a maintenance context, this is not realistic: predicting a **later** failure than the true one (i.e., telling an engineer "you still have time" when the engine is actually close to failing) is far more dangerous than predicting an **earlier** failure (which only costs an unnecessary early inspection). The PHM08 scoring function used in the original NASA/PHM08 Prognostics Challenge captures this asymmetry with an exponential penalty:

```
d = RUL_predicted - RUL_actual

s_i = exp(-d / 13) - 1,   if d < 0   (early prediction)
s_i = exp( d / 10) - 1,   if d >= 0  (late prediction)

Score = Σ s_i
```

Because the exponent denominator for late predictions (10) is smaller than for early predictions (13), the penalty grows faster for late (dangerous) predictions than for early (conservative) ones. Lower total score is better.

## Project Structure

```
RUL-Plain/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── ML-DL-Pipeline.png
│   └── figs/
│       └── all needed figures
├── notebooks/
│   └── all needed notebooks
├── src/
│   └── all code scripts, pipelines, and functions
├── api.py            # All API endpoints
├── interface.py       # Streamlit interface
└── README.md
```

## Assumptions

* **Independent units:** Engine #1 is assumed to be fully independent of engine #2. The degradation or failure of one engine gives no prior indication of another engine's degradation rate, since each engine has a different flight history and manufacturing history.
* **Monotonic degradation:** Engine condition is assumed to always worsen over time (as cycles increase, RUL decreases). An engine does not mechanically "heal" itself; any sudden improvement in sensor efficiency readings is treated as noise, not genuine health improvement.
* **End of trajectory = failure:** The last recorded cycle for each engine in the training set is assumed to be the actual failure point (RUL = 0). The model is trained on this assumption.
* **Sensor reliability:** Changes in sensor readings are assumed to reflect real changes in the engine's physical (thermodynamic) state, not sensor faults or drift.
* **Initial wear:** Engines in N-CMAPSS are assumed not to all start from an identical "zero degradation" (100% health) state. Each engine starts with a different initial wear level, which explains the variation in unit trajectory lengths.

## Constraints

* **Operating conditions impact:** The model is constrained by the fact that sensor readings reflect not only degradation but also instantaneous changes in altitude, Mach number, and throttle resolver angle (TRA). Accurate prediction from raw data is not possible without addressing this through normalization.
* **No maintenance logs:** We are constrained by the absence of any data on intermediate maintenance actions performed during an engine's life. We do not know whether a filter or minor part was replaced, which could have slightly altered the degradation trajectory.
* **Censored test data (revised):** Unlike many published C-MAPSS variants where the test set trajectories are truncated before failure, direct inspection of this dataset (`test_df["RUL"].describe()`) showed that RUL is provided in full for every row in the test set (range 0–89 across ~2.7M rows). This constraint therefore does not apply to this specific dataset release, and RUL-based evaluation metrics (RMSE, PHM08 score) can be computed directly against the full test set rather than only at a single truncation point per unit.
