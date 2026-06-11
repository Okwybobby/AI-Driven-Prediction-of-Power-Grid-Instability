# Nigeria Grid Stability Dataset — Research Notes & ML Guide

**Project:** AI-Driven Prediction of Power Grid Instability in Nigeria
**Dataset File:** `nigeria_grid_stability_dataset.csv`
**Date Compiled:** 2026-06-11
**Compiled By:** Research data extraction from published Nigerian power systems papers

---

## 1. Project Overview

The goal of this project is to build a machine learning model that can predict power grid instability in the Nigerian 330kV transmission network **before** it occurs, giving operators time to intervene and prevent cascading failures or total blackouts.

Nigeria's national grid operates at a nominal frequency of **50 Hz** and transmission voltage of **330 kV**. The grid has historically suffered from severe instability due to insufficient generation capacity, inadequate transmission infrastructure, aging equipment, and load-shedding policies. The system frequently experiences partial or total collapses.

The ML model targets a **three-class classification** problem:
- **Class 0 — Stable:** Normal operating conditions; frequency near 50 Hz, voltage near nominal
- **Class 1 — Unstable/Warning:** System is stressed or oscillating; intervention required
- **Class 2 — Collapse/Blackout:** System has lost synchronism or experienced a blackout

The dataset compiled here draws from dynamic stability simulations (ETAP software), static load-flow analysis, and national grid operating samples. It is a **research-grade seed dataset** intended to be augmented with synthetic data before model training.

---

## 2. Dataset Summary Table

| Section | Source Paper | Records | Sim Type | Key Features Available | Label Distribution |
|---|---|---|---|---|---|
| ETAP_S1_Stable | Oluseyi et al., NIJOTECH 2017 | 40 | Dynamic (transient) | Frequency, Voltage, Rotor_Angle, Active_Power | 40×Label=0 |
| ETAP_S1_Unstable | Oluseyi et al., NIJOTECH 2017 | 30 | Dynamic (transient) | Frequency, Voltage, Rotor_Angle, Active_Power | 10×Label=0, 19×Label=1, 1×Label=2 |
| ETAP_S2_Unstable | Oluseyi et al., NIJOTECH 2017 | 17 | Dynamic (transient) | Frequency, Voltage, Rotor_Angle, Active_Power | 1×Label=0, 15×Label=1, 1×Label=2 |
| Nigeria31Bus_Static | Oluseyi et al., AZOJETE 2018 | 31 | Static load flow | Frequency, Voltage, Angle, Active_Power, Reactive_Power | 22×Label=0, 9×Label=1 |
| Sample_NationalGrid | Project Description Sample | 5 | Operational sample | Frequency, Voltage, Active_Power, Reactive_Power, Temp | 2×Label=0, 1×Label=1, 2×Label=2 |
| **TOTAL** | | **123** | | | **75×Label=0, 36×Label=1, 4×Label=2** |

**Important:** The dataset is heavily imbalanced (75 stable vs 4 collapse). Augmentation is essential before training. See Section 7.

---

## 3. Key Thresholds for the Nigerian Grid

These are the critical operating boundaries used to define stability labels. Use these to inform feature engineering and to validate synthetic data generation.

### Frequency Thresholds (Nominal: 50 Hz)

| Condition | Frequency Range | Action / Label |
|---|---|---|
| Normal | 49.5 – 50.5 Hz | Stable (0) |
| Under-frequency warning | 48.5 – 49.5 Hz | Unstable (1) — load shedding initiated |
| Over-frequency warning | 50.5 – 51.5 Hz | Unstable (1) — generation tripping |
| Under-frequency critical | < 48.5 Hz | Collapse imminent (2) |
| Over-frequency critical | > 51.5 Hz | Collapse imminent (2) |
| Collapse / blackout | 0.0 Hz | Collapse (2) |

*Nigerian Grid Code specifies operating band of 49.5–50.5 Hz; the under-frequency load-shedding relay typically trips at 48.5 Hz.*

### Voltage Thresholds (Nominal: 330 kV = 1.0 pu)

| Condition | Voltage (pu) | Voltage (kV) | Label |
|---|---|---|---|
| Normal | 0.97 – 1.05 pu | 320.1 – 346.5 kV | Stable (0) |
| Warning (under-voltage) | 0.95 – 0.97 pu | 313.5 – 320.1 kV | Warning (1) |
| Stressed | < 0.95 pu | < 313.5 kV | Unstable (1) |
| Voltage collapse | < 0.50 pu | < 165.0 kV | Collapse (2) |
| Blackout | 0.0 pu | 0.0 kV | Collapse (2) |

### Rotor Angle Thresholds (Dynamic Stability)

| Condition | Rotor Angle | Interpretation |
|---|---|---|
| Stable oscillation | –30° to +60° | Normal post-fault recovery |
| Warning | 60° – 90° | Machine approaching stability limit |
| Loss of synchronism | > ±90° sustained, or continuously increasing | Unstable (1) |
| Out of step | Angle rotating through 360° | Collapse (2) |

### Critical Clearing Time (CCT)

| Fault Location | CCT | Consequence if Exceeded |
|---|---|---|
| Egbin–Ikeja West line fault | ~1.0 s (CCT proven at 0.2s) | Loss of synchronism at ~13s |
| Ikeja-West busbar fault | ~0.45 s | Loss of synchronism at ~21.5s |

---

## 4. Power Station Capacities — Ikeja-West 330kV Sub-Network

These are the generation sources modeled in the ETAP simulation (Source 1). The Egbin/AES unit served as the swing (reference) bus.

| Power Station | Role in Model | Installed Capacity (MW) | Available Capacity (MW) | Notes |
|---|---|---|---|---|
| Egbin / AES | Swing bus (reference generator) | 1504 | 1000 | Largest in sub-network; modeled at 443.239 MW dispatch in simulations |
| Omotosho NIPP | PV bus (voltage-controlled) | 250 | 200 | Gas turbine plant |
| Olorunsogo NIPP | PV bus (voltage-controlled) | 200 | 160 | Gas turbine plant |
| **Sub-network Total** | | **1954** | **1360** | 330kV system |

**Mechanical Power Reference:** The ETAP simulation consistently shows Mech_Power_MW = 443.239 MW for the Egbin/AES generator across all scenarios. This is the governor setpoint for the simulation. Electrical power fluctuates around this value as the system oscillates.

---

## 5. Nigeria-31 Bus System — Critical Transmission Lines

From the fragility analysis in Source 2. The Line Quality Potential (LQP) factor and the Lmn voltage stability index both approach 1.0 as a line nears its stability limit. A value above ~0.9 is considered critical.

| From Bus | To Bus | Line Description | LQP Factor | Lmn Index | Fragility Level |
|---|---|---|---|---|---|
| Bus 11 (Oshogbo) | Bus 10 (Ikeja-West) | Oshogbo–Ikeja-West | 0.9924 | 0.8813 | **CRITICAL** — most fragile line in network |
| Bus 24 (Ayede) | Bus 10 (Ikeja-West) | Ayede–Ikeja-West | 0.4379 | 0.4550 | HIGH |
| Bus 18 (Kano) | Bus 12 (Kaduna) | Kano–Kaduna | 0.3339 | 0.3051 | MODERATE-HIGH |
| Bus 11 (Oshogbo) | Bus 8 (Benin) | Oshogbo–Benin | 0.3132 | 0.2297 | MODERATE |
| Bus 10 (Ikeja-West) | Bus 8 (Benin) | Ikeja-West–Benin | 0.2737 | 0.1379 | MODERATE |
| Bus 26 (Sepele_TS) | Bus 2 (Jebba_HT) | Sepele–Jebba | 0.0213 | 0.4642 | MODERATE (anomalous LQP) |
| Bus 31 (Jebba_TS2) | Bus 7 (Jebba_TS) | Jebba loop | 0.0253 | 0.6231 | MODERATE-HIGH (anomalous LQP) |

**Key Insight:** The Oshogbo–Ikeja-West corridor is the single most critical line in the Nigerian 330kV network with an LQP of 0.9924 (threshold = 1.0). A fault on this line is the scenario most likely to cascade into a system-wide collapse. This line should be a priority feature in the ML model.

**Buses Below Voltage Threshold (V < 0.97 pu) in Static Analysis:**

| Bus | Name | Voltage (pu) | Voltage (kV) | Status |
|---|---|---|---|---|
| Bus 14 | Egbin_TS | 0.9777 | 322.6 | Warning |
| Bus 15 | Onitsha | 0.9730 | 321.1 | Warning |
| Bus 17 | Ajaokuta | 0.9575 | 315.9 | **Stressed** |
| Bus 22 | Alaoji | 0.9775 | 322.6 | Warning |
| Bus 28 | Afam_TS | 0.9810 | 323.7 | Warning |

---

## 6. Reliability Statistics by City

From Ajenikoko et al. (2010), based on Customer Average Interruption Duration Index (CAIDI) measurements. Lower Relative CAIDI indicates worse reliability (more frequent, longer outages relative to the average).

| City | Avg Relative CAIDI | Reliability Level | Practical Implication |
|---|---|---|---|
| Ibadan | 0.7186 | Average | Frequent short interruptions |
| Port Harcourt | 0.5279 | Average | Moderate interruption frequency |
| Benin | 0.7579 | Average | Best of the sampled cities |
| Ilorin | 0.1195 | **Poor** | Near-constant outages; very low reliability |
| Ikeja | 0.3976 | **Poor** | Highly unreliable despite industrial importance |
| Kaduna | 0.4017 | **Poor** | Significant industrial load at risk |
| Kano | 0.4108 | **Poor** | High commercial demand, poor supply |

**Note:** Ilorin's Relative CAIDI of 0.1195 is the lowest in the dataset, meaning customers there experience outages that are both very frequent and of long duration relative to national norms. This aligns with the bus-level analysis: Ilorin is electrically distant from generation centers and served by a single corridor.

---

## 7. Data Gaps and Augmentation Suggestions

### What Is Missing from This Dataset

| Gap | Impact on ML Model | Priority |
|---|---|---|
| Real-time SCADA measurements (not simulated) | Model may not generalize to actual grid noise | HIGH |
| Reactive power measurements for ETAP records | All Q values are estimated (×0.6012 ratio) | HIGH |
| Temperature variation data (only 32–33°C present) | Cannot model seasonal effects | MEDIUM |
| Load demand variation over 24h cycles | No day/night or peak/off-peak patterns | HIGH |
| Multiple fault types (line-to-ground, 3-phase, etc.) | Only line and busbar faults simulated | MEDIUM |
| Generator trip events (sudden loss of generation) | Common cause of Nigerian grid collapse | HIGH |
| Intermediate collapse progression (t=13s to blackout) | Only 4 collapse records in entire dataset | CRITICAL |
| Data from all 31 buses during dynamic events | Dynamic data only from Egbin/Ikeja-West | MEDIUM |
| Harmonic distortion / power quality features | Not captured in these papers | LOW |

### Synthetic Data Generation Strategy

The dataset currently has 123 records. A minimum of **1,000–5,000 records** is recommended for meaningful ML training. The following methods can be used to augment:

#### Method 1: Physics-Based Interpolation
For the ETAP dynamic scenarios, the simulation outputs at ~0.1–0.2s intervals. Interpolate between known time steps using:
- **Linear interpolation** for slowly varying quantities (Mech_Power, Ambient_Temp)
- **Cubic spline interpolation** for oscillating quantities (Rotor_Angle, Frequency)
- This can generate ~5–10× more records while maintaining physical realism

#### Method 2: SMOTE (Synthetic Minority Oversampling Technique)
- Apply SMOTE specifically to the **Collapse (Label=2)** class (currently only 4 records)
- Use Borderline-SMOTE or ADASYN variants which are better for rare-event detection
- Python: `from imblearn.over_sampling import SMOTE, BorderlineSMOTE`
- Target: generate at least 50–100 synthetic collapse records

#### Method 3: Parametric Variation of Bus Conditions
Using the 31-bus static data as a base, vary:
- Load demand ±20% across all buses simultaneously (morning/evening peak vs. off-peak)
- Apply load growth factor (Nigeria grid load grows ~6% per year; simulate 2020–2030 conditions)
- For each load level, re-compute expected voltage using the linearized power flow equations (DC approximation)

#### Method 4: Gaussian Noise Injection
For stable operating records:
- Add Gaussian noise (σ = 0.01 to 0.05) to Frequency, Voltage_pu, and Active_Power_MW
- This simulates real SCADA measurement noise
- Preserves the label of the original record (noise-augmented stable = still stable)

#### Method 5: Scenario Combination
Combine features from different bus states to simulate scenarios not in the papers:
- Bus 17 (Ajaokuta, V=0.9575 pu) with frequency from unstable ETAP records
- This generates realistic stressed-but-not-collapsing scenarios that are underrepresented

#### Recommended Augmentation Target

| Label | Current Count | Augmentation Target | Method |
|---|---|---|---|
| 0 — Stable | 75 | 400 | Gaussian noise + parametric variation |
| 1 — Unstable | 36 | 300 | SMOTE + interpolation |
| 2 — Collapse | 4 | 150 | Borderline-SMOTE + physics interpolation |
| **Total** | **123** | **850+** | |

---

## 8. ML Model Recommendations

### Feature Engineering

The raw features in the CSV should be supplemented with the following engineered features before training:

| Engineered Feature | Formula / Method | Rationale |
|---|---|---|
| Frequency_Deviation | abs(Frequency_Hz - 50.0) | How far from nominal; more informative than raw Hz |
| Voltage_Deviation_pu | abs(Voltage_pu - 1.0) | Distance from nominal voltage |
| Power_Imbalance_MW | Active_Power_MW - 443.239 (or grid average) | Mechanical vs. electrical power mismatch drives instability |
| Rate_of_Change_of_Frequency (ROCOF) | dFrequency/dt between consecutive records | ROCOF > 0.5 Hz/s is a critical instability indicator |
| Voltage_Rate_of_Change | dVoltage_pu/dt | Rapidly falling voltage = collapse precursor |
| Rotor_Angle_Velocity | dAngle/dt | Angular acceleration indicates generator going out of step |
| P_Q_Ratio | Active_Power_MW / Reactive_Power_MVAr | Power factor proxy; stressed systems show high Q demand |
| Frequency_×_Voltage | Frequency_Hz × Voltage_pu | Interaction term capturing combined deterioration |
| Time_Since_Fault | Sim_Time_s - fault_onset_time | Proximity to fault event; useful for time-series models |

**Note:** ROCOF is likely the single most predictive feature for pre-collapse detection. Nigerian grid collapses typically show ROCOF > 1.0 Hz/s in the 2–5 seconds before full blackout.

### Recommended Algorithms

| Algorithm | Pros | Cons | Recommended Use |
|---|---|---|---|
| **Random Forest** | Handles class imbalance well with class_weight; robust to noise; interpretable feature importances | Not ideal for time-series patterns | First baseline model; good for static features |
| **XGBoost / LightGBM** | Best performance on tabular data in practice; fast; scale_pos_weight for imbalance | Requires careful hyperparameter tuning | Primary production model candidate |
| **LSTM (Long Short-Term Memory)** | Captures temporal dependencies in time-series; can learn ROCOF patterns implicitly | Requires sequential ordering; needs more data | Best for deployment on streaming SCADA data |
| **1D CNN** | Can detect oscillation patterns in frequency/voltage waveforms | Less interpretable | Useful for waveform-based input |
| **Isolation Forest** | Unsupervised; can detect anomalies without labeled collapse data | Does not output class probabilities | Pre-screening / anomaly detection layer |
| **Logistic Regression** | Fully interpretable; fast; useful as baseline | Cannot capture non-linear stability boundaries | Baseline comparison only |

**Recommended Pipeline:**
1. Train Random Forest baseline (fast, interpretable)
2. Train XGBoost with SMOTE-augmented data (expected best performance on this dataset)
3. If real-time streaming data becomes available, deploy LSTM on sliding 10-second windows

### Handling Class Imbalance

Do **not** simply oversample without strategy. The collapse class is physically rare but operationally critical. A false negative on a collapse event is far more costly than a false positive.

```python
# Recommended approach in scikit-learn
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Apply BorderlineSMOTE only on training set (never on test set)
smote = BorderlineSMOTE(k_neighbors=3, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Step 2: Use class_weight='balanced' as additional protection
clf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',  # further penalizes collapse misclassification
    random_state=42
)
clf.fit(X_train_res, y_train_res)

# Step 3: Evaluate with appropriate metrics
# Do NOT use accuracy alone — a model that always predicts "Stable"
# achieves ~61% accuracy on this dataset but is completely useless
print(classification_report(y_test, clf.predict(X_test),
      target_names=['Stable', 'Unstable', 'Collapse']))
```

**Evaluation Metrics to Prioritize:**
- **Recall for Class 2 (Collapse):** Must be as high as possible (> 0.90 target). Missing a collapse is catastrophic.
- **F1-Score (macro-averaged):** Balances precision and recall across all three classes
- **AUC-ROC (one-vs-rest):** Useful for threshold tuning
- **Confusion Matrix:** Always inspect manually; a model that confuses Stable with Unstable is acceptable; one that confuses Unstable with Collapse is not

### Suggested Train/Test Split

Given the small dataset size, use **stratified k-fold cross-validation** (k=5) rather than a single train/test split. This ensures each fold has proportional representation of all three classes.

```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**Do not use time-based splits** until the dataset has proper timestamps from a single continuous SCADA stream. The current dataset mixes records from different simulation scenarios.

---

## 9. Data Quality Notes

### Estimated vs. Measured Values

| Field | Status | Notes |
|---|---|---|
| Frequency_Hz | **Measured** (from simulation) | Direct ETAP output for ETAP records; reported values for static/sample records |
| Rotor_Angle_deg | **Measured** (from simulation) | Direct ETAP output; not available for static or national grid records |
| Active_Power_MW (Elec_Power) | **Measured** (from simulation) | Direct ETAP output; load MW from load-flow for static records |
| Mech_Power_MW | **Measured** (from simulation) | Governor setpoint; constant at 443.239 MW — not included as ML feature (leakage risk) |
| Voltage_pu / Voltage_kV | **ESTIMATED** for ETAP records | ETAP scenarios only reported fault-clearing voltage (97.02% = 320.2 kV for stable). Voltage trajectory during unstable scenarios was estimated using published descriptions of voltage collapse behavior from the same paper. |
| Reactive_Power_MVAr | **ESTIMATED** for ETAP records | Computed as Active_Power_MW × 0.6012. This ratio was derived from the Nigeria-31 bus load flow where Q/P ratios average approximately 0.60 across load buses. This is an approximation; actual Q values depend on power factor correction equipment. |
| Reactive_Power_MVAr | **MEASURED** for 31-bus records | Direct load-flow output from Source 2 |
| Ambient_Temp_C | **Estimated** | Set to 33°C uniformly (within Nigeria's typical 30–35°C range); no time-of-day or seasonal variation |
| Voltage_kV for 31-bus | **Computed** from V_pu × 330 | Exact within rounding; V_pu is direct simulation output |

### Caveats and Limitations

1. **Scale mismatch:** ETAP records are for a **single generator** (443 MW Egbin unit) in a sub-network. Nigeria31Bus and Sample_NationalGrid records are at **system level** (4,000+ MW total). These should ideally be modeled separately or normalized.

2. **Voltage estimation uncertainty:** The voltage profiles assigned to ETAP unstable records are based on the paper's qualitative description of voltage behavior. The actual voltage trajectory during loss-of-synchronism can differ significantly from these estimates. The `Voltage_Estimated = TRUE` flag identifies these records.

3. **Missing intermediate time steps:** Scenario 2 (Appendix D) has a large time gap between t=17.55s and t=21.35s. The system behavior during this 3.8-second gap is unknown. Do not interpolate blindly across this gap without domain knowledge.

4. **Static vs. dynamic features:** The Nigeria-31 bus records are from static load flow (no dynamics), while ETAP records are from dynamic simulations. A model trained on both must not treat the static Angle_deg as equivalent to the dynamic Rotor_Angle_deg. Consider adding a `Data_Type` feature (static/dynamic) or training separate models.

5. **Reactive power for Collapse records:** Records with Stability_Label=2 and Power=0 correctly show Reactive_Power=0. This is the post-collapse state, not a useful predictor feature — the collapse has already occurred.

6. **No real-world validation:** All ETAP data is simulation-derived. The model trained on this data must be validated against real Nigerian grid event logs (obtainable from TCN — Transmission Company of Nigeria) before deployment.

---

## 10. References

1. **Oluseyi, P.O., Ogunjuyigbe, A., Jimoh, A.A., and Okakwu, I.K.** (2017). "Analysis of the Transient Stability Limit of Nigeria's 330kV Transmission Sub-Network." *Nigerian Journal of Technology (NIJOTECH)*, Vol. 36, No. 3, pp. 928–941. *(Source for ETAP_S1_Stable, ETAP_S1_Unstable, ETAP_S2_Unstable sections. Appendices B, C, and D.)*

2. **Oluseyi, P.O., Akinbulire, T.O., Abdulkareem, A., and Awosope, C.O.A.** (2018). "Assessment of Electrical Grid Fragility in Nigeria-31 Bus System." *Arid Zone Journal of Engineering, Technology and Environment (AZOJETE)*, Vol. 14, No. 3, pp. 494–510. *(Source for Nigeria31Bus_Static section. Tables 1 and 2.)*

3. **Ajenikoko, G.A., Ojerinde, A.I., and Adeleke, B.** (2010). "Power Quality and Reliability Assessment of the Nigerian Power Grid." *(Source for reliability indices by city — Section 6 of this document.)*

4. **Project Description — AI-Driven Prediction of Power Grid Instability in Nigeria** (2026). Internal project sample dataset. *(Source for Sample_NationalGrid section. Five national-grid-level sample records.)*

---

## Appendix A: Column Definitions for CSV

| Column | Type | Description |
|---|---|---|
| Record_ID | Integer | Sequential unique identifier |
| Dataset_Section | String | Which source section this record belongs to |
| Source_Paper | String | Reference to originating paper |
| Sim_Time_s | Float / String | Simulation time in seconds, or timestamp string for national grid, or "static" |
| Bus_ID | String | Unique identifier for the bus or measurement point |
| Bus_Name | String | Human-readable bus/location name |
| Frequency_Hz | Float | System frequency at measurement point |
| Voltage_pu | Float | Bus voltage in per-unit (1.0 = nominal) |
| Voltage_kV | Float | Bus voltage in kilovolts |
| Angle_deg | Float | Voltage angle (static) or rotor angle (dynamic) in degrees |
| Active_Power_MW | Float | Active power at bus (load MW for static; electrical power for dynamic) |
| Reactive_Power_MVAr | Float | Reactive power at bus |
| Ambient_Temp_C | Float | Ambient temperature in degrees Celsius |
| Grid_State | String | Descriptive label (Stable / Pre_Fault / Fault_Onset / Unstable / Warning / Stressed / Collapse / Blackout) |
| Stability_Label | Integer | **Target variable:** 0=Stable, 1=Unstable/Warning, 2=Collapse/Blackout |
| Voltage_Estimated | Boolean | TRUE if Voltage_pu/kV was estimated rather than directly measured or simulated |
| Reactive_Estimated | Boolean | TRUE if Reactive_Power_MVAr was estimated using the P×0.6012 approximation |

---

## 11. Compatibility Audit — Existing CSV vs. Project Work.pdf Requirements

**Date of audit:** 2026-06-11
**Against document:** Project Work.pdf — *AI-Driven Prediction of Power Grid Instability*

### Project Requirements Summary (from Chapter 3 & 4)

| Requirement | Source in Document | Details |
|---|---|---|
| Time-series input data | Ch. 3 — Data Collection | Sequential measurements from TCN or simulation |
| Frequency + Voltage features | Ch. 3 — Data Preprocessing | "Normalize voltage/frequency readings" |
| Stability classification labels | Ch. 4 — Evaluation | "Predicting a Collapse vs. Stable state" |
| Python-compatible format | Ch. 4 — Implementation | "Use Python (Scikit-learn, TensorFlow/Keras)" |
| Support for LSTM / time-series models | Ch. 3 — Model Selection | "LSTM networks are great for time-series power data" |
| Alert-system output (T-minutes prediction) | Ch. 5 — Conclusion | "Can it predict a collapse 15 minutes ahead?" |
| Handle missing values / Nigerian data quality | Ch. 3 — Data Preprocessing | "Handle missing values (common in Nigerian data)" |

### Compatibility Verdict for `nigeria_grid_stability_dataset.csv` (123 rows)

| Requirement | Status | Notes |
|---|---|---|
| Frequency feature present | ✅ PASS | `Frequency_Hz` column — direct simulation output |
| Voltage feature present | ✅ PASS | `Voltage_pu` and `Voltage_kV` — pu is preferred for ML normalisation |
| Active/Reactive Power features | ✅ PASS | Present; reactive is estimated for ETAP records (flag provided) |
| Stability labels (0/1/2) | ✅ PASS | `Stability_Label` column; maps to Stable/Unstable/Collapse |
| Time-series structure | ⚠️ PARTIAL | ETAP sections are sequential (0.1s steps). Static 31-bus section has no time axis. Mixed sections break temporal continuity. |
| CSV format for Python | ✅ PASS | Standard CSV; readable with `pandas.read_csv()` directly |
| LSTM-ready sequential ordering | ⚠️ PARTIAL | Records within each ETAP scenario are sequential. Sections from different papers are not part of a single continuous stream; cannot be fed as one LSTM sequence without sectioning. |
| Sufficient volume for training | ❌ FAIL | 123 rows is far below the minimum (~1000–5000) for reliable ML training; ~5000–10000 for LSTM |
| Collapse class representation | ❌ FAIL | Only 4 collapse records (3.3%). LSTM/RF cannot learn collapse patterns from 4 examples. |
| ROCOF feature (key predictor) | ❌ MISSING | Not computed in original CSV. ROCOF (dF/dt) is the primary early-warning indicator; must be engineered or included. |
| Consistent spatial scale | ⚠️ WARNING | Mixed generator-level, bus-level, and national-level records in the same file. Normalisation alone cannot fully resolve this. |
| Missing value handling | ✅ PASS | No NaN values present; estimated fields flagged with Boolean columns. |

**Overall compatibility: PARTIAL — structurally sound, insufficient in volume and temporal density.**

The original CSV is valid as a research reference and seed dataset. It is **not suitable as a standalone training dataset** for the ML pipeline described in Project Work.pdf due to insufficient size and class imbalance. The augmented file (`nigeria_grid_stability_augmented.csv`) was created specifically to address these gaps.

---

## 12. Augmented Dataset — `nigeria_grid_stability_augmented.csv`

**Generated:** 2026-06-11
**Script:** `generate_augmented.py` (same directory)
**Method:** Physics-informed synthetic generation using Python stdlib only (no external dependencies for generation)

### Generation Methodology

Synthetic data was generated using first-principles power system physics:

- **Mean-reverting random walk** (Ornstein-Uhlenbeck-style): frequency and voltage evolve as `x[t+1] = x[t] + θ(μ - x[t])·dt + σ·ε` where θ is the reversion rate, μ is the nominal target, and ε ~ N(0,1). This produces realistic oscillatory behaviour around nominal values rather than uncorrelated random noise.
- **ROCOF (Rate of Change of Frequency)** computed as `(f[t] - f[t-1]) / Δt` at each time step. This is a derived feature directly available in the augmented dataset and is the primary short-term collapse predictor.
- **Reactive power** estimated as `Q = P × 0.6012` (derived from Nigeria-31 bus Q/P ratio). No change from original methodology.
- **Label assignment** follows the Nigerian grid thresholds established in Section 3 of this document:
  - Label 0 (Stable): 49.5 ≤ f ≤ 50.5 Hz **AND** V ≥ 0.95 pu
  - Label 1 (Warning): 49.0 ≤ f < 49.5 OR f > 50.5 OR V < 0.95 pu (mild deviation)
  - Label 1 (Unstable): f < 49.0 OR f > 51.5 OR V < 0.90 pu (severe deviation)
  - Label 2 (Collapse): f < 48.5 Hz OR V < 0.80 pu OR (f ≈ 0 AND V ≈ 0)

### Dataset Schema (Augmented File)

The augmented CSV uses a streamlined, ML-ready schema. It **replaces** the metadata-heavy columns of the seed file with features more directly useful for model training:

| Column | Type | Notes vs. Original CSV |
|---|---|---|
| Record_ID | Integer | Same |
| Scenario | String | Replaces `Dataset_Section` + `Source_Paper` — ML-irrelevant metadata removed |
| Source | String | Constant: `Synthetic_PhysicsInformed` |
| Sim_Time_s | Float | Same (in seconds; daily cycle rows use 60s intervals) |
| Bus_ID | String | Same concept |
| Bus_Name | String | Same concept |
| Frequency_Hz | Float | Same — **primary ML feature** |
| Voltage_pu | Float | Same — **primary ML feature** |
| Voltage_kV | Float | Same (= Voltage_pu × 330) |
| Angle_deg | Float | Same |
| Active_Power_MW | Float | Same |
| Reactive_Power_MVAr | Float | Same |
| **ROCOF_Hz_per_s** | Float | **NEW** — Rate of Change of Frequency; computed from consecutive time steps |
| Ambient_Temp_C | Float | Same; now includes temperature variation by scenario |
| Grid_State | String | Same enumeration |
| Stability_Label | Integer | Same — **target variable** |

*Removed from augmented schema: `Voltage_Estimated`, `Reactive_Estimated` (all synthetic data has known provenance; no per-row quality flags needed).*

### Scenario Breakdown and Label Distribution

| Scenario | Rows | Dominant Label | Description |
|---|---|---|---|
| Normal_Operation | 4,000 | 0 — Stable | Nominal grid; f ~ 50Hz ±0.3, V ~ 1.0pu ±0.02 |
| LightLoad_Night | 1,000 | 0 — Stable | Off-peak; f slightly above nominal, P ~270 MW |
| HeavyLoad_Peak | 1,200 | 0–1 — Stable/Warning | Peak demand; f dips to ~49.5Hz, V sags to ~0.97pu |
| Fault_PreFault | 400 | 0 — Stable | 20 fault events × 20 pre-fault steps; nominal conditions |
| Fault_Onset | 160 | 1 — Warning/Unstable | 20 fault events × 8 steps; sudden V drop and f deviation |
| Fault_Unstable | 500 | 1 — Unstable | 20 fault events × 25 steps; oscillating/diverging |
| Fault_Collapse | 200 | 2 — Collapse | 8 collapse events × 25 steps; f and V → 0 |
| Fault_Recovery | 600 | 0–1 — Recovery | 12 recovery events × 50 steps; f/V return to nominal |
| Voltage_Instability | 650 | 1 — Unstable | 10 events; Q surge collapses V while f stays near 50Hz |
| Voltage_Collapse | 200 | 2 — Collapse | Continuation of voltage instability events |
| GenLoss_FreqDrop | 600 | 1–2 — Unstable | 12 events; sudden generation loss; f drops sharply |
| DailyLoadCycle | 1,500 | 0–1 — Stable/Warning | 24h load profile; f follows load; 60s time steps |
| UFLS_Declining | 200 | 1 — Warning | Under-frequency load shedding pre-shed phase |
| UFLS_Shedding | 100 | 0–1 | Moment of load shed; f rebounds |
| UFLS_Recovery | 200 | 0 — Stable | Post-shedding stabilisation |
| **TOTAL** | **11,510** | | |

**Label distribution:**

| Label | Count | Percentage | Notes |
|---|---|---|---|
| 0 — Stable | 8,493 | 73.8% | Realistic: grids are stable most of the time |
| 1 — Unstable/Warning | 1,585 | 13.8% | Stressed/oscillating conditions |
| 2 — Collapse | 1,432 | 12.4% | Intentionally elevated vs. real world for model training |

> **Note on Collapse ratio:** Real-world grids experience collapse far less than 12% of the time. The 12.4% figure is a deliberate overrepresentation to ensure the model has enough collapse examples to learn from. When deploying, use class weights or decision threshold tuning to re-calibrate output probabilities to real-world base rates.

### How to Load and Use in Python

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load augmented dataset
df = pd.read_csv('nigeria_grid_stability_augmented.csv')

# Select ML features (drop metadata and derived-but-redundant columns)
FEATURES = [
    'Frequency_Hz', 'Voltage_pu', 'Angle_deg',
    'Active_Power_MW', 'Reactive_Power_MVAr',
    'ROCOF_Hz_per_s', 'Ambient_Temp_C'
]
TARGET = 'Stability_Label'

X = df[FEATURES].values
y = df[TARGET].values

# Train/test split — stratified to preserve class ratios
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalise (IMPORTANT: fit scaler on training data only)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
```

**For LSTM sequential input**, reshape per scenario:
```python
# Group by scenario-event and create sequences of fixed window length
# Example: 20-step sliding window with step=1
WINDOW = 20
sequences, labels_seq = [], []
for i in range(len(X_train) - WINDOW):
    sequences.append(X_train[i:i+WINDOW])
    labels_seq.append(y_train[i+WINDOW])  # predict state at end of window

import numpy as np
X_lstm = np.array(sequences)   # shape: (N, 20, 7)
y_lstm = np.array(labels_seq)  # shape: (N,)
```

> **Caution:** Do not build LSTM sequences across scenario boundaries (e.g., across Normal_Operation and a Fault event). Filter by Scenario before windowing. The `Scenario` column can be used to group sequences correctly.

### Differences from Seed Dataset (`nigeria_grid_stability_dataset.csv`)

| Aspect | Seed (123 rows) | Augmented (11,510 rows) |
|---|---|---|
| Source | Published research papers + ETAP simulation | Physics-informed synthetic generation |
| Time resolution | 0.1s (ETAP dynamic), static, or 1-min operational | 0.1s (most), 60s (daily cycle) |
| ROCOF feature | Not present | ✅ Included as computed column |
| Collapse examples | 4 (3.3%) | 1,432 (12.4%) |
| Voltage quality | Mixed (estimated for ETAP) | All consistent (synthetic) |
| Temporal continuity | Broken between sections | Continuous within each scenario |
| Temperature variation | Uniform 33°C | Varies 24–39°C by scenario/season |
| Recommended use | Reference / validation against real papers | Primary training dataset |

**Recommended workflow:** Train on `nigeria_grid_stability_augmented.csv`, validate findings against the physics-derived values in `nigeria_grid_stability_dataset.csv` (the seed data acts as a cross-check sanity set).

---
