# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Nigeria Grid Instability Prediction — Notebook 2
# ## SVM · KNN · MLP · Stacking Ensemble
#
# **Purpose:** Extend the baseline models from Notebook 1 (LR, RF, XGBoost, BiLSTM)
# with kernel-based (SVM), distance-based (KNN), neural (MLP), and meta-learning
# (Stacking) approaches.  We also add PCA visualisation, feature selection,
# probability calibration, and learning-curve diagnostics.
#
# **Grid context:** Nigerian 330 kV transmission network, nominal 50 Hz / 1.0 pu.
# Labels — 0: Stable, 1: Unstable / Warning, 2: Collapse.

# %% [markdown]
# ## 0. Setup

# %%
import os
import sys
import warnings
import pickle
import ast
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts & notebooks
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  — registers the 3-D projection

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, learning_curve
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    ConfusionMatrixDisplay, f1_score
)
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV

try:
    from sklearn.inspection import LearningCurveDisplay
    HAS_LCD = True
except ImportError:
    HAS_LCD = False
    print("[INFO] LearningCurveDisplay not available — using manual learning curve plot.")

# ── imbalanced-learn (optional) ──────────────────────────────────────────────
try:
    from imblearn.over_sampling import BorderlineSMOTE
    HAS_SMOTE = True
    print("[OK] imbalanced-learn found — BorderlineSMOTE will be used.")
except ImportError:
    HAS_SMOTE = False
    print("[WARN] imbalanced-learn not installed.  Falling back to class_weight='balanced'.")
    print("       To install: pip install imbalanced-learn")

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
# Change DATA_DIR to the folder containing the CSV files if they are not in the
# current working directory.
DATA_DIR    = "."
MODEL_DIR   = os.path.join(DATA_DIR, "models")
FIGURES_DIR = os.path.join(DATA_DIR, "nb2_figures")

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Global constants ──────────────────────────────────────────────────────────
RANDOM_STATE   = 42
AUGMENTED_CSV  = os.path.join(DATA_DIR, "nigeria_grid_stability_augmented.csv")
SEED_CSV       = os.path.join(DATA_DIR, "nigeria_grid_stability_dataset.csv")

# Class label mapping used throughout
LABEL_MAP = {0: "Stable", 1: "Unstable", 2: "Collapse"}
CLASS_NAMES = ["Stable", "Unstable", "Collapse"]

print("Notebook 2 — SVM / KNN / MLP / Stacking")
print(f"  DATA_DIR    = {os.path.abspath(DATA_DIR)}")
print(f"  MODEL_DIR   = {os.path.abspath(MODEL_DIR)}")
print(f"  FIGURES_DIR = {os.path.abspath(FIGURES_DIR)}")

# %% [markdown]
# ## 1. Data Loading

# %%
df = pd.read_csv(AUGMENTED_CSV)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print("\nColumn dtypes:")
print(df.dtypes)

print("\nFirst 3 rows:")
print(df.head(3).to_string())

# Sanity-check expected columns
EXPECTED_COLS = [
    "Record_ID", "Scenario", "Source", "Sim_Time_s",
    "Bus_ID", "Bus_Name", "Frequency_Hz", "Voltage_pu", "Voltage_kV",
    "Angle_deg", "Active_Power_MW", "Reactive_Power_MVAr",
    "ROCOF_Hz_per_s", "Ambient_Temp_C", "Grid_State", "Stability_Label"
]
missing = [c for c in EXPECTED_COLS if c not in df.columns]
if missing:
    print(f"[WARN] Missing expected columns: {missing}")
else:
    print("\n[OK] All expected columns present.")

print("\nLabel distribution:")
label_counts = df["Stability_Label"].value_counts().sort_index()
for lbl, cnt in label_counts.items():
    print(f"  {lbl} ({LABEL_MAP.get(lbl, '?'):>10s}): {cnt:>6,}  ({cnt/len(df)*100:.1f}%)")

# %% [markdown]
# ## 2. Feature Engineering
#
# We create derived features that encode domain knowledge about power-system
# physics.  Frequency deviation and voltage deviation are fundamental stability
# indicators; their squares amplify large excursions.  Apparent power, power
# factor, and P/Q ratio capture the loading condition of each bus.  FxV is a
# cross-term that flags simultaneous frequency and voltage stress.

# %%
def engineer_features(data):
    """
    Add physics-motivated derived features to the raw grid telemetry.

    Returns a new DataFrame — the original is not modified.
    """
    d = data.copy()
    eps = 1e-6  # prevents division-by-zero while keeping magnitudes meaningful

    # Absolute deviation from nominal values (50 Hz, 1.0 pu)
    d["Freq_Dev"]     = (d["Frequency_Hz"] - 50.0).abs()
    d["V_Dev"]        = (d["Voltage_pu"]   - 1.0).abs()

    # Squared deviations — penalise large excursions more heavily
    d["Freq_sq"]      = d["Freq_Dev"] ** 2
    d["V_sq"]         = d["V_Dev"]   ** 2

    # Power-flow quantities
    d["Apparent_S"]   = (d["Active_Power_MW"]**2 + d["Reactive_Power_MVAr"]**2) ** 0.5
    d["PQ_Ratio"]     = d["Active_Power_MW"] / (d["Reactive_Power_MVAr"].abs() + eps)
    d["Power_Factor"] = d["Active_Power_MW"] / (d["Apparent_S"] + eps)

    # Cross-term: simultaneous frequency × voltage stress
    d["FxV"]          = d["Frequency_Hz"] * d["Voltage_pu"]

    # ROCOF features (Rate of Change of Frequency — key collapse predictor)
    d["ROCOF_abs"]    = d["ROCOF_Hz_per_s"].abs()
    d["ROCOF_sq"]     = d["ROCOF_Hz_per_s"] ** 2

    # Clean any inf / NaN produced by edge cases
    d.replace([float("inf"), float("-inf")], 0.0, inplace=True)
    d.fillna(0.0, inplace=True)
    return d


ENG_FEATURES = [
    "Frequency_Hz", "Voltage_pu", "Angle_deg",
    "Active_Power_MW", "Reactive_Power_MVAr", "ROCOF_Hz_per_s",
    "Ambient_Temp_C",
    "Freq_Dev", "V_Dev", "Freq_sq", "V_sq",
    "Apparent_S", "PQ_Ratio", "Power_Factor", "FxV",
    "ROCOF_abs", "ROCOF_sq",
]

df = engineer_features(df)
print(f"Feature matrix shape after engineering: {df[ENG_FEATURES].shape}")
print(f"Features ({len(ENG_FEATURES)}): {ENG_FEATURES}")

# %% [markdown]
# ## 3. Preprocessing
#
# ### Split strategy
# We use a **stratified 80/20 train/test split** to preserve the class ratio in
# both partitions — critical because the Collapse class is rare.
#
# ### Scaling
# `StandardScaler` is fitted **only on the training set** and then applied to the
# test set.  Fitting on the full dataset would cause data leakage.  All
# sklearn and SVM-based models require standardised input.
#
# ### Class imbalance
# BorderlineSMOTE generates synthetic minority-class samples near the decision
# boundary (the hardest region to classify).  If imbalanced-learn is absent we
# fall back to `class_weight='balanced'` inside each estimator.

# %%
X = df[ENG_FEATURES].values
y = df["Stability_Label"].values

# Stratified split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {X_train_raw.shape[0]:,}  |  Test: {X_test_raw.shape[0]:,}")

# Standard scaling — fit on train only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled  = scaler.transform(X_test_raw)

scaler_path = os.path.join(MODEL_DIR, "nb2_scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"Scaler saved → {scaler_path}")

# Class imbalance handling
if HAS_SMOTE:
    print("\nApplying BorderlineSMOTE ...")
    bsmote = BorderlineSMOTE(random_state=RANDOM_STATE, kind="borderline-1")
    X_train_res, y_train_res = bsmote.fit_resample(X_train_scaled, y_train)
    print("After SMOTE label distribution:")
    for lbl in np.unique(y_train_res):
        cnt = (y_train_res == lbl).sum()
        print(f"  {lbl} ({LABEL_MAP[lbl]:>10s}): {cnt:>6,}")
else:
    # Without SMOTE: use the scaled training set as-is; each model will receive
    # class_weight='balanced' where the API supports it.
    X_train_res, y_train_res = X_train_scaled, y_train
    print("\nUsing original training distribution (class_weight='balanced' per model).")

print(f"\nFinal training set: {X_train_res.shape[0]:,} samples × {X_train_res.shape[1]} features")

# %% [markdown]
# ## 4. PCA Analysis
#
# Principal Component Analysis reduces the 17-dimensional feature space for
# visualisation.  We first examine how many components are needed to explain
# 95 % of variance (scree plot), then project onto the first 2 and 3 components
# to visualise class separation geometry.

# %%
# Fit PCA on the (possibly resampled) training data
pca_full = PCA(random_state=RANDOM_STATE)
pca_full.fit(X_train_res)

explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)
n_95 = int(np.searchsorted(cumulative, 0.95)) + 1

print(f"Components to explain 95% variance: {n_95}")
print(f"Variance explained by PC1+PC2: {cumulative[1]:.3f}")
print(f"Variance explained by PC1+PC2+PC3: {cumulative[2]:.3f}")

# ── Scree plot ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(range(1, len(explained) + 1), explained * 100, alpha=0.6,
       label="Individual", color="#2196F3")
ax.plot(range(1, len(explained) + 1), cumulative * 100, "o-",
        color="#E91E63", label="Cumulative")
ax.axhline(95, color="grey", linestyle="--", linewidth=1, label="95 % threshold")
ax.axvline(n_95, color="orange", linestyle="--", linewidth=1,
           label=f"PC {n_95}")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Explained Variance (%)")
ax.set_title("PCA Scree Plot — Nigeria Grid Features")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "pca_scree.png"), dpi=150)
plt.close(fig)
print("Saved: pca_scree.png")

# ── 2-D scatter ───────────────────────────────────────────────────────────────
pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
# Project TRAINING set for scatter
X_tr_2d = pca2.fit_transform(X_train_res)

colors = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}
markers = {0: "o", 1: "s", 2: "^"}

fig, ax = plt.subplots(figsize=(8, 6))
for lbl in [0, 1, 2]:
    mask = y_train_res == lbl
    ax.scatter(X_tr_2d[mask, 0], X_tr_2d[mask, 1],
               c=colors[lbl], marker=markers[lbl], s=10, alpha=0.4,
               label=LABEL_MAP[lbl])
ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
ax.set_title("PCA 2-D Projection — Grid Stability Classes")
ax.legend(markerscale=2)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "pca_2d_scatter.png"), dpi=150)
plt.close(fig)
print("Saved: pca_2d_scatter.png")

# ── 3-D scatter ───────────────────────────────────────────────────────────────
pca3 = PCA(n_components=3, random_state=RANDOM_STATE)
X_tr_3d = pca3.fit_transform(X_train_res)

fig = plt.figure(figsize=(9, 7))
ax3 = fig.add_subplot(111, projection="3d")
for lbl in [0, 1, 2]:
    mask = y_train_res == lbl
    ax3.scatter(X_tr_3d[mask, 0], X_tr_3d[mask, 1], X_tr_3d[mask, 2],
                c=colors[lbl], marker=markers[lbl], s=8, alpha=0.3,
                label=LABEL_MAP[lbl])
ax3.set_xlabel("PC1"); ax3.set_ylabel("PC2"); ax3.set_zlabel("PC3")
ax3.set_title("PCA 3-D Projection — Grid Stability Classes")
ax3.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "pca_3d_scatter.png"), dpi=150)
plt.close(fig)
print("Saved: pca_3d_scatter.png")

# %% [markdown]
# ## 5. Feature Selection — SelectKBest (ANOVA F-test)
#
# `SelectKBest` with the ANOVA F-statistic ranks features by how well their
# variance separates the three classes.  The top-10 features will be noted for
# discussion, but we train all models on the **full** 17-feature set to avoid
# discarding potentially useful interactions.

# %%
selector = SelectKBest(score_func=f_classif, k=10)
selector.fit(X_train_res, y_train_res)

scores  = selector.scores_
indices = np.argsort(scores)[::-1]

print("Top-10 features by ANOVA F-score:")
for rank, idx in enumerate(indices[:10], 1):
    print(f"  {rank:2d}. {ENG_FEATURES[idx]:<22s}  F = {scores[idx]:.1f}")

# Bar chart
top10_names   = [ENG_FEATURES[i] for i in indices[:10]]
top10_scores  = [scores[i] for i in indices[:10]]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(top10_names[::-1], top10_scores[::-1], color="#3F51B5", alpha=0.8)
ax.set_xlabel("ANOVA F-score")
ax.set_title("Top-10 Most Discriminative Features (SelectKBest / F-classif)")
# Annotate values
for bar, val in zip(bars, top10_scores[::-1]):
    ax.text(val + max(top10_scores) * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}", va="center", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "feature_selection_kbest.png"), dpi=150)
plt.close(fig)
print("Saved: feature_selection_kbest.png")

# Store top-10 feature names for reference
TOP10_FEATURES = top10_names
print(f"\nTop-10: {TOP10_FEATURES}")

# %% [markdown]
# ## 6. Support Vector Machine (SVM)
#
# ### Why RBF-kernel SVM?
# The RBF (radial basis function) kernel maps input features into an infinite-
# dimensional Hilbert space, allowing the SVM to find non-linear decision
# boundaries — ideal for grid states that are non-linearly separable in the
# original feature space.  `class_weight='balanced'` ensures the minority
# Collapse class influences the margin equally.
#
# ### Scalability note
# SVM training complexity is O(n²) to O(n³) in memory and time.  For 9,000+
# training samples this is slow.  We **sub-sample 5,000 rows** for the grid
# search while preserving class proportions (stratified), then refit the best
# estimator on the full resampled set.

# %%
# Sub-sample for GridSearchCV — preserves class proportions
N_SVM_GRID = min(5000, len(X_train_res))
idx_sub = []
for lbl in np.unique(y_train_res):
    lbl_idx = np.where(y_train_res == lbl)[0]
    n_take  = int(N_SVM_GRID * len(lbl_idx) / len(y_train_res))
    rng     = np.random.RandomState(RANDOM_STATE)
    idx_sub.extend(rng.choice(lbl_idx, n_take, replace=False))
idx_sub = np.array(idx_sub)
X_svm_sub = X_train_res[idx_sub]
y_svm_sub = y_train_res[idx_sub]
print(f"SVM grid-search sub-sample: {len(idx_sub):,} rows")

# Hyperparameter grid
# C controls regularisation strength — higher C = tighter fit.
# gamma controls the RBF kernel width — 'scale' = 1/(n_feat * X.var()).
svm_param_grid = {
    "C":     [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.01, 0.1],
}

svm_base = SVC(
    kernel="rbf",
    class_weight="balanced",   # corrects for class imbalance in the sub-sample
    probability=True,          # needed for calibration and ROC curves
    random_state=RANDOM_STATE,
    cache_size=1000,           # MB — speeds up training on repeated kernel calls
)

cv_svm = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
svm_grid = GridSearchCV(
    svm_base,
    svm_param_grid,
    cv=cv_svm,
    scoring="f1_macro",        # macro-F1 treats all classes equally
    n_jobs=-1,
    verbose=0,
    refit=True,
)

print("Running GridSearchCV for SVM (this may take a few minutes) ...")
svm_grid.fit(X_svm_sub, y_svm_sub)
print(f"Best params : {svm_grid.best_params_}")
print(f"Best CV F1  : {svm_grid.best_score_:.4f}")

# Refit best estimator on FULL resampled training set for better generalisation
best_svm = SVC(
    kernel="rbf",
    C=svm_grid.best_params_["C"],
    gamma=svm_grid.best_params_["gamma"],
    class_weight="balanced",
    probability=True,
    random_state=RANDOM_STATE,
    cache_size=1000,
)
print("\nRefitting best SVM on full training set ...")
best_svm.fit(X_train_res, y_train_res)

# Evaluation
y_pred_svm = best_svm.predict(X_test_scaled)
print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=CLASS_NAMES))

# Confusion matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm_svm, display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False)
ax.set_title("SVM — Confusion Matrix")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "svm_confusion_matrix.png"), dpi=150)
plt.close(fig)
print("Saved: svm_confusion_matrix.png")

# ROC curves (one-vs-rest)
y_prob_svm  = best_svm.predict_proba(X_test_scaled)
y_test_bin  = label_binarize(y_test, classes=[0, 1, 2])

fig, ax = plt.subplots(figsize=(7, 5))
roc_colors = ["#4CAF50", "#FF9800", "#F44336"]
for i, (cls_name, col) in enumerate(zip(CLASS_NAMES, roc_colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob_svm[:, i])
    roc_auc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=col, lw=2, label=f"{cls_name} (AUC={roc_auc_val:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("SVM — ROC Curves (One-vs-Rest)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "svm_roc.png"), dpi=150)
plt.close(fig)
print("Saved: svm_roc.png")

# %% [markdown]
# ## 7. K-Nearest Neighbours (KNN)
#
# ### Why KNN?
# KNN is a non-parametric, instance-based learner with no training phase — it
# memorises the data and classifies by majority vote of the k nearest neighbours
# (Euclidean distance in scaled feature space).  It provides a useful baseline
# for how much structure exists in the feature space near each query point.
#
# ### Choosing k
# Small k → high variance (overfit); large k → high bias (underfit).  We sweep
# k ∈ {1, 3, 5, 7, 10, 15, 20, 30} and pick the value that maximises macro-F1
# on a held-out validation fold.
#
# ### k-distance graph
# Plotting the distance to the 5th nearest neighbour (sorted ascending) helps
# identify natural cluster density: a sharp "elbow" indicates where data
# transitions from dense clusters to sparse regions — useful for understanding
# class separability.

# %%
k_values = [1, 3, 5, 7, 10, 15, 20, 30]
knn_f1_scores = []

cv_knn = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

print("Sweeping k for KNN ...")
for k in k_values:
    knn_tmp = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    fold_scores = []
    for tr_idx, val_idx in cv_knn.split(X_train_res, y_train_res):
        knn_tmp.fit(X_train_res[tr_idx], y_train_res[tr_idx])
        y_val_pred = knn_tmp.predict(X_train_res[val_idx])
        fold_scores.append(f1_score(y_train_res[val_idx], y_val_pred, average="macro"))
    mean_f1 = np.mean(fold_scores)
    knn_f1_scores.append(mean_f1)
    print(f"  k={k:2d} → macro-F1 = {mean_f1:.4f}")

best_k = k_values[int(np.argmax(knn_f1_scores))]
print(f"\nOptimal k = {best_k}  (macro-F1 = {max(knn_f1_scores):.4f})")

# k vs F1 chart
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(k_values, knn_f1_scores, "o-", color="#9C27B0", lw=2, ms=7)
ax.axvline(best_k, color="#E91E63", linestyle="--", label=f"Best k={best_k}")
ax.set_xlabel("k (number of neighbours)")
ax.set_ylabel("CV Macro-F1")
ax.set_title("KNN — k vs Macro-F1")
ax.legend()
ax.set_xticks(k_values)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "knn_k_sweep.png"), dpi=150)
plt.close(fig)
print("Saved: knn_k_sweep.png")

# Fit optimal KNN
best_knn = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
best_knn.fit(X_train_res, y_train_res)

y_pred_knn = best_knn.predict(X_test_scaled)
print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn, target_names=CLASS_NAMES))

cm_knn = confusion_matrix(y_test, y_pred_knn)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm_knn, display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False)
ax.set_title(f"KNN (k={best_k}) — Confusion Matrix")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "knn_confusion_matrix.png"), dpi=150)
plt.close(fig)
print("Saved: knn_confusion_matrix.png")

# k-distance graph — sorted distance to 5th nearest neighbour
# Reveals cluster density and the natural "elbow" separating inliers from outliers
N_KDIST = min(3000, len(X_train_res))
rng_kd  = np.random.RandomState(RANDOM_STATE)
kdist_idx = rng_kd.choice(len(X_train_res), N_KDIST, replace=False)
nn_model = NearestNeighbors(n_neighbors=6, n_jobs=-1)   # 6 because index 0 = self
nn_model.fit(X_train_res[kdist_idx])
distances, _ = nn_model.kneighbors(X_train_res[kdist_idx])
dist_5th = np.sort(distances[:, 5])[::-1]   # distance to 5th neighbour, descending

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(dist_5th, color="#009688", lw=1.2)
ax.set_xlabel("Points (sorted by decreasing distance)")
ax.set_ylabel("Distance to 5th Nearest Neighbour")
ax.set_title("k-Distance Graph — Cluster Density Structure")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "knn_kdistance.png"), dpi=150)
plt.close(fig)
print("Saved: knn_kdistance.png")

# %% [markdown]
# ## 8. MLP — Multilayer Perceptron
#
# ### Architecture rationale
# `(128 → 64 → 32)` is a funnel architecture: wider layers near the input capture
# many potential feature interactions; narrower layers force the network to distil
# them into a compact representation.  ReLU avoids the vanishing gradient problem
# common with sigmoid/tanh in deeper networks.  Early stopping with
# `validation_fraction=0.1` prevents overfitting.

# %%
best_mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",          # rectified linear unit — fast, no vanishing gradient
    solver="adam",              # adaptive moment estimation — robust to learning rate
    alpha=1e-4,                 # L2 regularisation weight — prevents over-reliance on any feature
    max_iter=300,
    early_stopping=True,        # stop when validation loss stops improving
    validation_fraction=0.1,    # 10 % of training data used for validation
    n_iter_no_change=15,        # patience: 15 epochs with no improvement
    random_state=RANDOM_STATE,
    verbose=False,
)

print("Fitting MLP ...")
best_mlp.fit(X_train_res, y_train_res)
print(f"MLP stopped after {best_mlp.n_iter_} iterations.")

y_pred_mlp = best_mlp.predict(X_test_scaled)
print("\nMLP Classification Report:")
print(classification_report(y_test, y_pred_mlp, target_names=CLASS_NAMES))

cm_mlp = confusion_matrix(y_test, y_pred_mlp)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm_mlp, display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False)
ax.set_title("MLP — Confusion Matrix")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "mlp_confusion_matrix.png"), dpi=150)
plt.close(fig)
print("Saved: mlp_confusion_matrix.png")

# Loss curve — shows training convergence behaviour
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(best_mlp.loss_curve_, label="Training loss", color="#3F51B5")
if best_mlp.validation_scores_ is not None:
    ax.plot(best_mlp.validation_scores_, label="Validation accuracy", color="#E91E63")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss / Accuracy")
ax.set_title("MLP — Training Loss and Validation Accuracy")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "mlp_loss_curve.png"), dpi=150)
plt.close(fig)
print("Saved: mlp_loss_curve.png")

# %% [markdown]
# ## 9. Stacking Ensemble
#
# ### Why stacking?
# Each base model (SVM, KNN, MLP) captures different aspects of the data:
# SVM finds global maximum-margin boundaries; KNN captures local density;
# MLP learns non-linear feature combinations.  A meta-learner (Logistic
# Regression) learns how much to trust each base model's output for each class.
# This typically outperforms any single base model.
#
# ### Stack design
# - **Level-0 (base):** SVM, KNN, MLP trained on 4-fold cross-val to produce
#   out-of-fold predictions (sklearn handles this automatically with
#   `StackingClassifier`).
# - **Level-1 (meta):** Logistic Regression with C=1 (mild regularisation) fits
#   on the stacked probability vectors.
#
# The meta-learner's coefficients reveal which base model the ensemble trusts
# most for each class.

# %%
stack_clf = StackingClassifier(
    estimators=[
        ("svm", best_svm),
        ("knn", best_knn),
        ("mlp", best_mlp),
    ],
    final_estimator=LogisticRegression(
        C=1,
        max_iter=500,
        multi_class="multinomial",
        random_state=RANDOM_STATE,
    ),
    cv=4,               # 4-fold CV for generating out-of-fold meta-features
    stack_method="predict_proba",  # use probability estimates as meta-features
    passthrough=False,  # meta-learner sees ONLY base-model outputs, not raw features
    n_jobs=-1,
)

print("Fitting Stacking Ensemble (patience required) ...")
stack_clf.fit(X_train_res, y_train_res)
print("Stacking ensemble fitted.")

y_pred_stack = stack_clf.predict(X_test_scaled)
print("\nStacking Classification Report:")
print(classification_report(y_test, y_pred_stack, target_names=CLASS_NAMES))

cm_stack = confusion_matrix(y_test, y_pred_stack)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm_stack, display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False)
ax.set_title("Stacking Ensemble — Confusion Matrix")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "stacking_confusion_matrix.png"), dpi=150)
plt.close(fig)
print("Saved: stacking_confusion_matrix.png")

# Meta-learner coefficients — which base model does the stacker trust?
meta_lr = stack_clf.final_estimator_
coef_df  = pd.DataFrame(
    meta_lr.coef_,
    index=CLASS_NAMES,
    columns=["svm_stable", "svm_unstable", "svm_collapse",
             "knn_stable", "knn_unstable", "knn_collapse",
             "mlp_stable", "mlp_unstable", "mlp_collapse"],
)
print("\nMeta-learner coefficients (higher = more trust from that base model):")
print(coef_df.to_string())

# %% [markdown]
# ## 10. Calibration Analysis
#
# ### Why calibration matters for grid protection
# A model that outputs `P(collapse) = 0.8` should actually collapse 80 % of the
# time.  If it is poorly calibrated, alert thresholds chosen from probability
# outputs will give either too many false alarms (operators stop trusting them)
# or miss genuine collapse events.
#
# ### Expected Calibration Error (ECE)
# ECE divides the probability range into bins and measures the weighted average
# gap between mean predicted probability and actual fraction of positives.  Lower
# ECE = better calibrated.

# %%
def compute_ece(y_true_binary, y_prob, n_bins=10):
    """
    Compute Expected Calibration Error for a binary outcome.

    Parameters
    ----------
    y_true_binary : array-like of {0, 1}
    y_prob        : predicted probability for the positive class
    n_bins        : number of equal-width bins
    """
    bins      = np.linspace(0, 1, n_bins + 1)
    ece       = 0.0
    n_samples = len(y_true_binary)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true_binary[mask].mean()   # fraction of actual positives in bin
        conf = y_prob[mask].mean()          # mean predicted probability in bin
        ece += (mask.sum() / n_samples) * abs(acc - conf)
    return ece


# Binary problem: Collapse (label=2) vs Rest
y_test_collapse = (y_test == 2).astype(int)

# Collect models and their predicted probabilities for the Collapse class
calib_models = {}

# SVM
y_prob_svm_c  = best_svm.predict_proba(X_test_scaled)[:, 2]
calib_models["SVM"]      = y_prob_svm_c

# KNN
y_prob_knn_c  = best_knn.predict_proba(X_test_scaled)[:, 2]
calib_models["KNN"]      = y_prob_knn_c

# MLP
y_prob_mlp_c  = best_mlp.predict_proba(X_test_scaled)[:, 2]
calib_models["MLP"]      = y_prob_mlp_c

# Stacking
y_prob_stack_c = stack_clf.predict_proba(X_test_scaled)[:, 2]
calib_models["Stacking"] = y_prob_stack_c

# Reliability diagram + ECE
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
axes = axes.ravel()

for ax, (model_name, probs) in zip(axes, calib_models.items()):
    CalibrationDisplay.from_predictions(
        y_test_collapse, probs,
        n_bins=10, ax=ax, name=model_name,
        color="#3F51B5",
    )
    ece_val = compute_ece(y_test_collapse, probs)
    ax.set_title(f"{model_name}  (ECE = {ece_val:.4f})")
    ax.text(0.05, 0.90, f"ECE = {ece_val:.4f}", transform=ax.transAxes,
            fontsize=9, color="red")

fig.suptitle("Calibration Reliability Diagrams — Collapse Class", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "calibration_reliability.png"), dpi=150)
plt.close(fig)
print("Saved: calibration_reliability.png")

print("\nECE Summary (Collapse class):")
for model_name, probs in calib_models.items():
    ece_val = compute_ece(y_test_collapse, probs)
    print(f"  {model_name:<12s}: ECE = {ece_val:.4f}")

# %% [markdown]
# ## 11. Learning Curves
#
# Learning curves show train and cross-validation performance as a function of
# training-set size.  They answer:
# - **High train-val gap** → overfitting; more data would help.
# - **Both curves plateau early** → data saturation; more architecture/features
#   needed, not more data.
# - **Both curves still rising** → more data would improve performance.

# %%
def plot_manual_learning_curve(estimator, X, y, title, save_name,
                                train_sizes=None, cv=3):
    """
    Plot learning curve manually (compatible with all sklearn versions).
    Uses macro-F1 as the scoring metric.
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 8)

    from sklearn.model_selection import learning_curve as lc_fn

    sizes, train_scores, val_scores = lc_fn(
        estimator, X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, train_mean, "o-", color="#2196F3", label="Train F1")
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color="#2196F3")
    ax.plot(sizes, val_mean, "s-", color="#FF5722", label="Val F1")
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.2, color="#FF5722")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Macro-F1")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, save_name), dpi=150)
    plt.close(fig)
    print(f"Saved: {save_name}")
    return sizes, val_mean


# Use a smaller CV and sub-sample to keep runtime reasonable
LC_SUBSAMPLE = min(4000, len(X_train_res))
rng_lc       = np.random.RandomState(RANDOM_STATE)
lc_idx       = rng_lc.choice(len(X_train_res), LC_SUBSAMPLE, replace=False,)
X_lc         = X_train_res[lc_idx]
y_lc         = y_train_res[lc_idx]

# ── Stacking learning curve ───────────────────────────────────────────────────
# Re-build a fresh stacking estimator so sklearn can clone it internally
stack_for_lc = StackingClassifier(
    estimators=[
        ("svm",  SVC(kernel="rbf", C=svm_grid.best_params_["C"],
                     gamma=svm_grid.best_params_["gamma"],
                     class_weight="balanced", probability=True,
                     random_state=RANDOM_STATE)),
        ("knn",  KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)),
        ("mlp",  MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                               max_iter=200, random_state=RANDOM_STATE)),
    ],
    final_estimator=LogisticRegression(C=1, max_iter=300,
                                       random_state=RANDOM_STATE),
    cv=3, stack_method="predict_proba", n_jobs=1,
)

print("Computing Stacking learning curve ...")
_, stack_lc_vals = plot_manual_learning_curve(
    stack_for_lc, X_lc, y_lc,
    title="Stacking Ensemble — Learning Curve (Macro-F1)",
    save_name="learning_curve_stacking.png",
    cv=3,
)

# ── Random Forest learning curve (fast reference) ────────────────────────────
rf_for_lc = RandomForestClassifier(
    n_estimators=100, class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1,
)
print("Computing Random Forest learning curve ...")
_, rf_lc_vals = plot_manual_learning_curve(
    rf_for_lc, X_lc, y_lc,
    title="Random Forest — Learning Curve (Macro-F1)",
    save_name="learning_curve_rf.png",
    cv=3,
)

# %% [markdown]
# ## 12. Model Comparison
#
# Collect metrics for all four models trained in this notebook, and optionally
# load results from Notebook 1 for a cross-notebook comparison.

# %%
def eval_model(model, X_te, y_te, model_name, y_prob=None):
    """Return a dict of key metrics for one model."""
    y_pred = model.predict(X_te)
    report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    collapse_recall = report.get("2", {}).get("recall", float("nan"))
    macro_f1        = report["macro avg"]["f1-score"]
    macro_prec      = report["macro avg"]["precision"]
    macro_rec       = report["macro avg"]["recall"]
    acc             = report["accuracy"]

    # ROC-AUC for Collapse class
    if y_prob is not None:
        y_bin = (y_te == 2).astype(int)
        try:
            fpr, tpr, _ = roc_curve(y_bin, y_prob[:, 2])
            collapse_auc = auc(fpr, tpr)
        except Exception:
            collapse_auc = float("nan")
    else:
        collapse_auc = float("nan")

    return {
        "Model":           model_name,
        "Accuracy":        round(acc, 4),
        "Macro_Precision": round(macro_prec, 4),
        "Macro_Recall":    round(macro_rec, 4),
        "Macro_F1":        round(macro_f1, 4),
        "Collapse_Recall": round(collapse_recall, 4),
        "Collapse_AUC":    round(collapse_auc, 4),
    }


results_nb2 = []
results_nb2.append(eval_model(best_svm,   X_test_scaled, y_test, "SVM",
                               y_prob=best_svm.predict_proba(X_test_scaled)))
results_nb2.append(eval_model(best_knn,   X_test_scaled, y_test, "KNN",
                               y_prob=best_knn.predict_proba(X_test_scaled)))
results_nb2.append(eval_model(best_mlp,   X_test_scaled, y_test, "MLP",
                               y_prob=best_mlp.predict_proba(X_test_scaled)))
results_nb2.append(eval_model(stack_clf,  X_test_scaled, y_test, "Stacking",
                               y_prob=stack_clf.predict_proba(X_test_scaled)))

# Load Notebook-1 results if available
nb1_rf_path = os.path.join(MODEL_DIR, "random_forest.pkl")
nb1_gb_path = os.path.join(MODEL_DIR, "gradient_boost.pkl")
nb1_scaler  = os.path.join(MODEL_DIR, "scaler.pkl")

nb1_scaler_obj = None
if os.path.exists(nb1_scaler):
    with open(nb1_scaler, "rb") as f:
        nb1_scaler_obj = pickle.load(f)
    print(f"Loaded nb1 scaler from {nb1_scaler}")

def load_nb1_model(path, name, scaler_obj):
    """Load a nb1 model and evaluate it on the same test set."""
    if not os.path.exists(path):
        print(f"  [SKIP] {name} not found at {path}")
        return None
    with open(path, "rb") as f:
        mdl = pickle.load(f)
    # nb1 models were fit on nb1-scaled features
    if scaler_obj is not None:
        X_te = scaler_obj.transform(X_test_raw)
    else:
        X_te = X_test_scaled
    print(f"  Loaded {name} from {path}")
    return eval_model(mdl, X_te, y_test, name)

nb1_rf_result = load_nb1_model(nb1_rf_path, "RF (nb1)", nb1_scaler_obj)
nb1_gb_result = load_nb1_model(nb1_gb_path, "GradBoost (nb1)", nb1_scaler_obj)

all_results = results_nb2.copy()
if nb1_rf_result:
    all_results.append(nb1_rf_result)
if nb1_gb_result:
    all_results.append(nb1_gb_result)

results_df = pd.DataFrame(all_results)
print("\n===== MODEL COMPARISON TABLE =====")
print(results_df.to_string(index=False))

# Bar chart: Collapse Recall and Macro F1
model_names    = results_df["Model"].tolist()
collapse_recs  = results_df["Collapse_Recall"].tolist()
macro_f1s      = results_df["Macro_F1"].tolist()
x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 1.4), 5))
bars1 = ax.bar(x - width / 2, collapse_recs, width, label="Collapse Recall",
               color="#F44336", alpha=0.8)
bars2 = ax.bar(x + width / 2, macro_f1s,     width, label="Macro F1",
               color="#2196F3", alpha=0.8)

for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=20, ha="right")
ax.set_ylabel("Score")
ax.set_title("Model Comparison — Collapse Recall & Macro F1")
ax.legend()
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "model_comparison_bar.png"), dpi=150)
plt.close(fig)
print("Saved: model_comparison_bar.png")

# %% [markdown]
# ## 13. Save Models and Results

# %%
# Save all four nb2 models
models_to_save = {
    "nb2_svm.pkl":      best_svm,
    "nb2_knn.pkl":      best_knn,
    "nb2_mlp.pkl":      best_mlp,
    "nb2_stacking.pkl": stack_clf,
}

for fname, model in models_to_save.items():
    path = os.path.join(MODEL_DIR, fname)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    size_kb = os.path.getsize(path) / 1024
    print(f"Saved {fname:<24s}  ({size_kb:,.0f} KB)")

# Save results CSV
results_path = os.path.join(MODEL_DIR, "nb2_results.csv")
results_df.to_csv(results_path, index=False)
print(f"\nResults saved → {results_path}")

print("\n" + "=" * 60)
print("NOTEBOOK 2 COMPLETE")
print("=" * 60)
best_row = results_df.loc[results_df["Macro_F1"].idxmax()]
print(f"Best model (Macro F1): {best_row['Model']}  F1={best_row['Macro_F1']:.4f}")
best_collapse = results_df.loc[results_df["Collapse_Recall"].idxmax()]
print(f"Best Collapse Recall : {best_collapse['Model']}  Recall={best_collapse['Collapse_Recall']:.4f}")
print(f"\nFigures → {os.path.abspath(FIGURES_DIR)}")
print(f"Models  → {os.path.abspath(MODEL_DIR)}")
