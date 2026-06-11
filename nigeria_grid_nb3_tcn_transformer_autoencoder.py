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
# # Nigeria Grid Instability Prediction — Notebook 3
# ## TCN · Transformer Encoder · Autoencoder Anomaly Detection
#
# **Purpose:** Explore deep-learning approaches that explicitly model the
# *temporal* structure of power-system telemetry:
#
# | Model | Key idea |
# |---|---|
# | TCN | Dilated causal convolutions with residual connections |
# | Transformer | Self-attention over the sequence window |
# | Autoencoder | Unsupervised anomaly scoring; trained on Stable only |
#
# **Grid context:** Nigerian 330 kV transmission network, nominal 50 Hz / 1.0 pu.
# Labels — 0: Stable, 1: Unstable / Warning, 2: Collapse.
#
# **Requirements:**
# - `tensorflow >= 2.8` for TCN, Transformer, and Autoencoder sections.
# - `scikit-learn`, `pandas`, `numpy`, `matplotlib` for all other sections.
# - If TensorFlow is absent the notebook degrades gracefully.

# %% [markdown]
# ## 0. Setup

# %%
import os
import sys
import warnings
import pickle
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    ConfusionMatrixDisplay, f1_score
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ── TensorFlow ─────────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, Input
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    )
    from tensorflow.keras.utils import to_categorical

    TF_VERSION = tf.__version__
    print(f"[OK] TensorFlow {TF_VERSION} detected.")

    # Limit GPU memory growth if a GPU is present — prevents OOM errors
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        print(f"[OK] {len(gpus)} GPU(s) configured with memory growth.")
    else:
        print("[INFO] No GPU detected — running on CPU.")

    HAS_TF = True

except ImportError:
    HAS_TF = False
    print("[WARN] TensorFlow not installed.")
    print("       To install: pip install tensorflow")
    print("       Sections that require TF will be skipped gracefully.\n")

# ── Paths ─────────────────────────────────────────────────────────────────────
# Change DATA_DIR to the folder containing the CSV files if needed.
DATA_DIR    = "."
MODEL_DIR   = os.path.join(DATA_DIR, "models")
FIGURES_DIR = os.path.join(DATA_DIR, "nb3_figures")

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Global constants ──────────────────────────────────────────────────────────
RANDOM_STATE  = 42
LSTM_WINDOW   = 20    # number of time-steps per input sequence
T_HORIZON     = 5     # not used for sliding-window labels, kept for reference

AUGMENTED_CSV = os.path.join(DATA_DIR, "nigeria_grid_stability_augmented.csv")
SEED_CSV      = os.path.join(DATA_DIR, "nigeria_grid_stability_dataset.csv")

LABEL_MAP    = {0: "Stable", 1: "Unstable", 2: "Collapse"}
CLASS_NAMES  = ["Stable", "Unstable", "Collapse"]
N_CLASSES    = 3

np.random.seed(RANDOM_STATE)
if HAS_TF:
    tf.random.set_seed(RANDOM_STATE)

print("Notebook 3 — TCN / Transformer / Autoencoder")
print(f"  DATA_DIR    = {os.path.abspath(DATA_DIR)}")
print(f"  MODEL_DIR   = {os.path.abspath(MODEL_DIR)}")
print(f"  FIGURES_DIR = {os.path.abspath(FIGURES_DIR)}")
print(f"  LSTM_WINDOW = {LSTM_WINDOW}")

# %% [markdown]
# ## 1. Data Loading

# %%
df = pd.read_csv(AUGMENTED_CSV)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

EXPECTED_COLS = [
    "Record_ID", "Scenario", "Source", "Sim_Time_s",
    "Bus_ID", "Bus_Name", "Frequency_Hz", "Voltage_pu", "Voltage_kV",
    "Angle_deg", "Active_Power_MW", "Reactive_Power_MVAr",
    "ROCOF_Hz_per_s", "Ambient_Temp_C", "Grid_State", "Stability_Label"
]
missing = [c for c in EXPECTED_COLS if c not in df.columns]
if missing:
    print(f"[WARN] Missing columns: {missing}")
else:
    print("[OK] All expected columns present.")

print("\nLabel distribution:")
for lbl, cnt in df["Stability_Label"].value_counts().sort_index().items():
    print(f"  {lbl} ({LABEL_MAP.get(lbl,'?'):>10s}): {cnt:>6,}  ({cnt/len(df)*100:.1f}%)")

# %% [markdown]
# ## 2. Feature Engineering

# %%
def engineer_features(data):
    """
    Add physics-motivated derived features to the raw grid telemetry.
    Frequency/voltage deviations from Nigerian nominal (50 Hz, 1.0 pu),
    power-flow quantities, and ROCOF (Rate-of-Change-of-Frequency) features.
    Returns a new DataFrame — the original is not modified.
    """
    d   = data.copy()
    eps = 1e-6

    d["Freq_Dev"]     = (d["Frequency_Hz"] - 50.0).abs()
    d["V_Dev"]        = (d["Voltage_pu"]   - 1.0).abs()
    d["Freq_sq"]      = d["Freq_Dev"] ** 2
    d["V_sq"]         = d["V_Dev"]   ** 2
    d["Apparent_S"]   = (d["Active_Power_MW"]**2 + d["Reactive_Power_MVAr"]**2) ** 0.5
    d["PQ_Ratio"]     = d["Active_Power_MW"] / (d["Reactive_Power_MVAr"].abs() + eps)
    d["Power_Factor"] = d["Active_Power_MW"] / (d["Apparent_S"] + eps)
    d["FxV"]          = d["Frequency_Hz"] * d["Voltage_pu"]
    d["ROCOF_abs"]    = d["ROCOF_Hz_per_s"].abs()
    d["ROCOF_sq"]     = d["ROCOF_Hz_per_s"] ** 2

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
N_FEATURES = len(ENG_FEATURES)
print(f"Feature matrix: {df[ENG_FEATURES].shape}  ({N_FEATURES} features)")

# %% [markdown]
# ## 3. Preprocessing & Sequence Building
#
# ### Why respect event boundaries?
# The dataset contains multiple simulation scenarios.  Each scenario restarts
# `Sim_Time_s` from zero.  If we create sequences that straddle two scenarios,
# the network learns spurious temporal transitions (end-of-one-scenario →
# start-of-next).  `build_sequences` detects time resets and refuses to bridge
# them.
#
# ### Sequence shape
# Each sample is a 3-D tensor `(LSTM_WINDOW, N_FEATURES)` representing a sliding
# window of 20 consecutive timesteps.  The label is the class at the final step.

# %%
def build_sequences(X_scaled, y_labels, sim_times, window=LSTM_WINDOW):
    """
    Build overlapping sliding-window sequences while respecting event boundaries.

    Parameters
    ----------
    X_scaled  : ndarray (n_samples, n_features) — already scaled
    y_labels  : ndarray (n_samples,)
    sim_times : ndarray (n_samples,) — Sim_Time_s column
    window    : int — sequence length

    Returns
    -------
    X_seq : ndarray (n_seq, window, n_features)
    y_seq : ndarray (n_seq,)  — label at last timestep of each window
    """
    X_list, y_list = [], []
    n = len(X_scaled)

    for i in range(window, n):
        # Detect event boundary: if Sim_Time_s decreased or jumped back to ~0
        # between any two consecutive rows in the window, skip this window.
        t_window = sim_times[i - window : i + 1]
        # A reset is any point where time decreases (new scenario started)
        if np.any(np.diff(t_window) < 0):
            continue
        X_list.append(X_scaled[i - window : i])
        y_list.append(y_labels[i])

    if len(X_list) == 0:
        raise ValueError("build_sequences: no valid sequences found. "
                         "Check window size vs data length.")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


# ── Standard scaling ───────────────────────────────────────────────────────────
X_raw  = df[ENG_FEATURES].values
y_all  = df["Stability_Label"].values
t_all  = df["Sim_Time_s"].values

# 80/20 tabular split first so scaler is fit on training rows only
idx_all   = np.arange(len(X_raw))
idx_train, idx_test = train_test_split(
    idx_all, test_size=0.2, stratify=y_all, random_state=RANDOM_STATE
)
# Sort to preserve temporal order within each split
idx_train = np.sort(idx_train)
idx_test  = np.sort(idx_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_raw[idx_train])
X_test_scaled  = scaler.transform(X_raw[idx_test])

y_train_raw = y_all[idx_train]
y_test_raw  = y_all[idx_test]
t_train     = t_all[idx_train]
t_test      = t_all[idx_test]

print(f"Train rows: {len(idx_train):,}  |  Test rows: {len(idx_test):,}")

# Save scaler
scaler_path = os.path.join(MODEL_DIR, "nb3_scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"Scaler saved → {scaler_path}")

# ── Build sequences ────────────────────────────────────────────────────────────
print(f"\nBuilding sequences (window={LSTM_WINDOW}) ...")
X_seq_train, y_seq_train = build_sequences(X_train_scaled, y_train_raw, t_train)
X_seq_test,  y_seq_test  = build_sequences(X_test_scaled,  y_test_raw,  t_test)
print(f"Train sequences : {X_seq_train.shape}")
print(f"Test  sequences : {X_seq_test.shape}")

# ── Class weights for imbalanced training ─────────────────────────────────────
class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_seq_train),
    y=y_seq_train,
)
class_weight_dict = dict(enumerate(class_weights_arr))
print(f"\nClass weights: {class_weight_dict}")

# One-hot labels for Keras
if HAS_TF:
    y_train_cat = to_categorical(y_seq_train, num_classes=N_CLASSES)
    y_test_cat  = to_categorical(y_seq_test,  num_classes=N_CLASSES)

# Also keep tabular (non-sequence) versions for autoencoder
X_train_tab = X_train_scaled
X_test_tab  = X_test_scaled
y_train_tab = y_train_raw
y_test_tab  = y_test_raw

# %% [markdown]
# ## 4. Temporal Convolutional Network (TCN)
#
# ### TCN vs LSTM
# | Property | TCN | LSTM |
# |---|---|---|
# | Parallelism | Full (convolutions) | Sequential (hidden state) |
# | Vanishing gradient | No (residual connections) | Mitigated (gates) |
# | Memory | Bounded by receptive field | Theoretically unlimited |
# | Training speed | ~3–5× faster on CPU | Slower |
# | Causal property | Enforced by padding | Implicit |
#
# ### Receptive field
# With `kernel_size=k` and dilations `[1, 2, 4, 8, 16]`:
#
#     receptive_field = 1 + (k - 1) * sum(dilations) = 1 + 2 * (1+2+4+8+16) = 63
#
# so each output timestep "sees" 63 past input steps — well beyond our 20-step
# window, meaning the network has full context over the window.
#
# ### Residual connections
# If channel counts differ between the skip path and the main path, a 1×1
# convolution projects them to the same dimension before addition.

# %%
if not HAS_TF:
    print("[SKIP] TensorFlow not available — skipping TCN section.")
else:
    def _tcn_residual_block(x, n_filters, kernel_size, dilation, dropout=0.2):
        """
        One TCN residual block:
          causal Conv1D → BatchNorm → ReLU → Dropout ×2 → skip add.
        Causal padding ensures no future information leaks into predictions.
        """
        skip = x  # residual (skip) connection

        # First dilated causal conv
        x = layers.Conv1D(
            filters=n_filters, kernel_size=kernel_size,
            dilation_rate=dilation, padding="causal", activation=None,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(dropout)(x)

        # Second dilated causal conv (same dilation)
        x = layers.Conv1D(
            filters=n_filters, kernel_size=kernel_size,
            dilation_rate=dilation, padding="causal", activation=None,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(dropout)(x)

        # Project skip connection if channel count differs
        if skip.shape[-1] != n_filters:
            skip = layers.Conv1D(n_filters, kernel_size=1, padding="same")(skip)

        return layers.Add()([x, skip])


    def build_tcn(input_shape, n_classes,
                  n_filters=64, kernel_size=3,
                  dilations=None):
        """
        Build a Temporal Convolutional Network.

        Parameters
        ----------
        input_shape : tuple (window, n_features)
        n_classes   : int
        n_filters   : int  — number of convolutional filters per block
        kernel_size : int  — 3 gives effective receptive field growth per block
        dilations   : list — exponential doubling maximises receptive field
                             with the fewest parameters
        """
        if dilations is None:
            dilations = [1, 2, 4, 8, 16]

        inp = Input(shape=input_shape, name="tcn_input")
        x   = inp

        for d in dilations:
            x = _tcn_residual_block(x, n_filters, kernel_size, d)

        # Aggregate sequence dimension — GlobalAveragePooling1D averages over
        # all timesteps, which is more robust than taking just the last step.
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        out = layers.Dense(n_classes, activation="softmax", name="tcn_output")(x)

        return Model(inp, out, name="TCN")


    # Compute and print receptive field
    TCN_DILATIONS   = [1, 2, 4, 8, 16]
    TCN_KERNEL_SIZE = 3
    receptive_field = 1 + (TCN_KERNEL_SIZE - 1) * sum(TCN_DILATIONS)
    print(f"TCN receptive field = {receptive_field} timesteps")
    print(f"Window size         = {LSTM_WINDOW} timesteps")
    print(f"Full context: {'YES' if receptive_field >= LSTM_WINDOW else 'NO'}\n")

    tcn_model = build_tcn(
        input_shape=(LSTM_WINDOW, N_FEATURES),
        n_classes=N_CLASSES,
        n_filters=64,
        kernel_size=TCN_KERNEL_SIZE,
        dilations=TCN_DILATIONS,
    )
    tcn_model.summary()

    # Adam(3e-3) is aggressive but TCN with BatchNorm converges quickly;
    # ReduceLROnPlateau will back it off if needed.
    tcn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    tcn_callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True,
                      verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                          min_lr=1e-6, verbose=1),
    ]

    print("\nTraining TCN ...")
    tcn_history = tcn_model.fit(
        X_seq_train, y_train_cat,
        epochs=60,
        batch_size=256,       # large batch → stable gradient estimates for TCN
        validation_split=0.1,
        class_weight=class_weight_dict,
        callbacks=tcn_callbacks,
        verbose=1,
    )

    # ── TCN training history ──────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(tcn_history.history["loss"],     label="Train", color="#2196F3")
    ax1.plot(tcn_history.history["val_loss"], label="Val",   color="#FF5722")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("TCN — Training Loss"); ax1.legend()

    ax2.plot(tcn_history.history["accuracy"],     label="Train", color="#2196F3")
    ax2.plot(tcn_history.history["val_accuracy"], label="Val",   color="#FF5722")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("TCN — Training Accuracy"); ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "tcn_training_history.png"), dpi=150)
    plt.close(fig)
    print("Saved: tcn_training_history.png")

    # ── TCN evaluation ────────────────────────────────────────────────────────
    y_prob_tcn  = tcn_model.predict(X_seq_test, verbose=0)
    y_pred_tcn  = np.argmax(y_prob_tcn, axis=1)

    print("\nTCN Classification Report:")
    print(classification_report(y_seq_test, y_pred_tcn, target_names=CLASS_NAMES))

    cm_tcn = confusion_matrix(y_seq_test, y_pred_tcn)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm_tcn, display_labels=CLASS_NAMES).plot(
        ax=ax, colorbar=False)
    ax.set_title("TCN — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "tcn_confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print("Saved: tcn_confusion_matrix.png")

    # ROC — Collapse class
    y_test_bin   = (y_seq_test == 2).astype(int)
    fpr_tcn, tpr_tcn, _ = roc_curve(y_test_bin, y_prob_tcn[:, 2])
    auc_tcn = auc(fpr_tcn, tpr_tcn)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_tcn, tpr_tcn, color="#F44336", lw=2,
            label=f"Collapse AUC = {auc_tcn:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("TCN — Collapse ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "tcn_roc.png"), dpi=150)
    plt.close(fig)
    print("Saved: tcn_roc.png")
    print(f"TCN Collapse ROC-AUC = {auc_tcn:.4f}")
    print(f"TCN receptive field  = {receptive_field} timesteps")

# %% [markdown]
# ## 5. Transformer Encoder
#
# ### Why self-attention for grid telemetry?
# The Transformer's multi-head self-attention mechanism computes a weighted
# mixture of all timesteps in the window for every output position.  This allows
# the model to **jump directly** from a current timestep to any earlier timestep
# that caused it (e.g., a fault inception 15 steps ago).  LSTMs can do this but
# must propagate information through all intermediate hidden states.
#
# ### Architecture
# 1. **Input projection:** Dense(64) maps raw features to a uniform embedding
#    dimension `d_model`.
# 2. **Positional encoding:** Sine/cosine signals inject position information
#    (attention has no inherent notion of order).
# 3. **TransformerBlock ×2:** Multi-head attention → LayerNorm → FFN → LayerNorm,
#    each with residual connections.
# 4. **GlobalAveragePooling1D → Dense(32) → Dense(3, softmax).**

# %%
if not HAS_TF:
    print("[SKIP] TensorFlow not available — skipping Transformer section.")
else:
    class PositionalEncoding(layers.Layer):
        """
        Sinusoidal positional encoding (Vaswani et al. 2017).
        Adds a unique sine/cosine signal to each position so the model knows
        where each timestep is in the sequence.
        """
        def __init__(self, max_len=1000, **kwargs):
            super().__init__(**kwargs)
            self.max_len = max_len

        def call(self, x):
            seq_len   = tf.shape(x)[1]
            d_model   = tf.shape(x)[2]

            # Build PE matrix on the fly using numpy for simplicity
            # (called once per unique seq_len — no performance issue)
            d = x.shape[-1]  # static dimension
            pe = np.zeros((self.max_len, d), dtype=np.float32)
            positions = np.arange(self.max_len)[:, None]
            div_term  = np.exp(
                np.arange(0, d, 2) * (-math.log(10000.0) / d)
            )
            pe[:, 0::2] = np.sin(positions * div_term)
            if d % 2 == 0:
                pe[:, 1::2] = np.cos(positions * div_term)
            else:
                pe[:, 1::2] = np.cos(positions * div_term[: d // 2])

            pe_tensor = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
            return x + pe_tensor[:, :seq_len, :]

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"max_len": self.max_len})
            return cfg


    class TransformerBlock(layers.Layer):
        """
        Single Transformer encoder block:
          MultiHeadAttention → Add & LayerNorm → FFN → Add & LayerNorm.

        Parameters
        ----------
        d_model   : embedding dimension
        num_heads : number of attention heads
        key_dim   : per-head key/query dimension
        ffn_dim   : hidden units in the feed-forward network (typically 2–4× d_model)
        dropout   : dropout rate applied after attention and FFN
        """
        def __init__(self, d_model, num_heads, key_dim, ffn_dim,
                     dropout=0.1, **kwargs):
            super().__init__(**kwargs)
            self.att   = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=key_dim,
                dropout=dropout,
            )
            self.ffn   = keras.Sequential([
                layers.Dense(ffn_dim, activation="relu"),
                layers.Dense(d_model),
            ])
            self.norm1 = layers.LayerNormalization(epsilon=1e-6)
            self.norm2 = layers.LayerNormalization(epsilon=1e-6)
            self.drop1 = layers.Dropout(dropout)
            self.drop2 = layers.Dropout(dropout)

            # Store config for serialisation
            self._d_model   = d_model
            self._num_heads = num_heads
            self._key_dim   = key_dim
            self._ffn_dim   = ffn_dim
            self._dropout   = dropout

        def call(self, x, training=False, return_attention_scores=False):
            if return_attention_scores:
                attn_out, attn_weights = self.att(
                    x, x, return_attention_scores=True, training=training
                )
            else:
                attn_out    = self.att(x, x, training=training)
                attn_weights = None

            x = self.norm1(x + self.drop1(attn_out, training=training))
            x = self.norm2(x + self.drop2(self.ffn(x, training=training),
                                          training=training))
            if return_attention_scores:
                return x, attn_weights
            return x

        def get_config(self):
            cfg = super().get_config()
            cfg.update({
                "d_model":   self._d_model,
                "num_heads": self._num_heads,
                "key_dim":   self._key_dim,
                "ffn_dim":   self._ffn_dim,
                "dropout":   self._dropout,
            })
            return cfg


    def build_transformer(input_shape, n_classes,
                          d_model=64, num_heads=4, key_dim=16,
                          n_blocks=2, dropout=0.1):
        """
        Build a Transformer Encoder classifier.

        Parameters
        ----------
        input_shape : (window, n_features)
        d_model     : embedding dimension — 64 balances capacity vs speed
        num_heads   : 4 heads × key_dim=16 = 64-dim attention space
        key_dim     : per-head dimension
        n_blocks    : 2 stacked Transformer blocks provide enough depth
        dropout     : regularisation
        """
        inp = Input(shape=input_shape, name="transformer_input")

        # Project raw features into d_model-dimensional embedding space
        x = layers.Dense(d_model, name="input_projection")(inp)
        x = PositionalEncoding(max_len=200, name="pos_enc")(x)

        for i in range(n_blocks):
            x = TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                key_dim=key_dim,
                ffn_dim=d_model * 2,   # 2× d_model for FFN hidden layer
                dropout=dropout,
                name=f"transformer_block_{i}",
            )(x)

        # Aggregate sequence → fixed-length representation
        x = layers.GlobalAveragePooling1D(name="gap")(x)
        x = layers.Dropout(dropout, name="final_dropout")(x)
        x = layers.Dense(32, activation="relu", name="dense_32")(x)
        out = layers.Dense(n_classes, activation="softmax",
                           name="class_output")(x)

        return Model(inp, out, name="TransformerEncoder")


    transformer_model = build_transformer(
        input_shape=(LSTM_WINDOW, N_FEATURES),
        n_classes=N_CLASSES,
    )
    transformer_model.summary()

    transformer_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    tf_callbacks = [
        EarlyStopping(monitor="val_loss", patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                          min_lr=1e-6, verbose=1),
    ]

    print("\nTraining Transformer Encoder ...")
    tf_history = transformer_model.fit(
        X_seq_train, y_train_cat,
        epochs=60,
        batch_size=256,
        validation_split=0.1,
        class_weight=class_weight_dict,
        callbacks=tf_callbacks,
        verbose=1,
    )

    # ── Transformer training history ──────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(tf_history.history["loss"],     label="Train", color="#9C27B0")
    ax1.plot(tf_history.history["val_loss"], label="Val",   color="#FF9800")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Transformer — Training Loss"); ax1.legend()

    ax2.plot(tf_history.history["accuracy"],     label="Train", color="#9C27B0")
    ax2.plot(tf_history.history["val_accuracy"], label="Val",   color="#FF9800")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Transformer — Training Accuracy"); ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "transformer_training_history.png"),
                dpi=150)
    plt.close(fig)
    print("Saved: transformer_training_history.png")

    # ── Transformer evaluation ────────────────────────────────────────────────
    y_prob_tf  = transformer_model.predict(X_seq_test, verbose=0)
    y_pred_tf  = np.argmax(y_prob_tf, axis=1)

    print("\nTransformer Classification Report:")
    print(classification_report(y_seq_test, y_pred_tf, target_names=CLASS_NAMES))

    cm_tf = confusion_matrix(y_seq_test, y_pred_tf)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm_tf, display_labels=CLASS_NAMES).plot(
        ax=ax, colorbar=False)
    ax.set_title("Transformer — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "transformer_confusion_matrix.png"),
                dpi=150)
    plt.close(fig)
    print("Saved: transformer_confusion_matrix.png")

    fpr_tf, tpr_tf, _ = roc_curve((y_seq_test == 2).astype(int),
                                    y_prob_tf[:, 2])
    auc_tf = auc(fpr_tf, tpr_tf)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_tf, tpr_tf, color="#9C27B0", lw=2,
            label=f"Collapse AUC = {auc_tf:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Transformer — Collapse ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "transformer_roc.png"), dpi=150)
    plt.close(fig)
    print("Saved: transformer_roc.png")
    print(f"Transformer Collapse ROC-AUC = {auc_tf:.4f}")

    # ── Attention weight visualisation (single fault sample) ─────────────────
    # Build a sub-model that exposes attention weights from the first block.
    # This reveals which timesteps in the window the model attends to most.
    try:
        # Find a Collapse sample in the test set
        collapse_indices = np.where(y_seq_test == 2)[0]
        if len(collapse_indices) > 0:
            sample_idx = collapse_indices[0]
            sample_seq = X_seq_test[sample_idx : sample_idx + 1]  # (1, W, F)

            # Get the first TransformerBlock layer
            tb_layer = transformer_model.get_layer("transformer_block_0")

            # Forward pass through input projection and positional encoding
            proj_layer   = transformer_model.get_layer("input_projection")
            pos_layer    = transformer_model.get_layer("pos_enc")
            inp_tensor   = transformer_model.input
            proj_out     = proj_layer(inp_tensor)
            pos_out      = pos_layer(proj_out)

            # Build sub-model up to TransformerBlock input
            pre_block_model = Model(inputs=inp_tensor, outputs=pos_out)
            pre_block_out   = pre_block_model(sample_seq, training=False)

            # Call TransformerBlock with return_attention_scores=True
            _, attn_weights = tb_layer(
                pre_block_out, training=False, return_attention_scores=True
            )
            # attn_weights shape: (batch, heads, seq, seq)
            # Average over heads to get (seq, seq)
            avg_attn = attn_weights.numpy()[0].mean(axis=0)  # (W, W)

            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(avg_attn, cmap="Blues", aspect="auto",
                           vmin=0, vmax=avg_attn.max())
            ax.set_xlabel("Key timestep (attended to)")
            ax.set_ylabel("Query timestep")
            ax.set_title("Transformer Attention — Collapse Sample\n"
                         "(avg over 4 heads, Block 0)")
            plt.colorbar(im, ax=ax, label="Attention weight")
            fig.tight_layout()
            fig.savefig(os.path.join(FIGURES_DIR,
                                     "transformer_attention_collapse.png"),
                        dpi=150)
            plt.close(fig)
            print("Saved: transformer_attention_collapse.png")
        else:
            print("[INFO] No Collapse samples in test set — skipping attention viz.")

    except Exception as e:
        print(f"[WARN] Attention visualisation failed: {e}")

# %% [markdown]
# ## 6. Autoencoder for Anomaly Detection (Unsupervised Pre-Screening)
#
# ### Rationale
# A conventional classifier requires labelled Collapse examples to learn from.
# In early deployment (few historical collapses) this is a bottleneck.
#
# The autoencoder approach needs **only normal (Stable) data**:
# 1. Train the autoencoder to reconstruct Stable readings with low MSE.
# 2. At inference, a high reconstruction error signals the reading is
#    **unlike normal** — potentially an incipient fault or collapse.
#
# ### Architecture
# Dense autoencoder on the **flattened** feature vector (17 features):
# - **Encoder:** 17 → 64 → 32 → 16  (bottleneck)
# - **Decoder:** 16 → 32 → 64 → 17  (linear output for continuous reconstruction)
#
# MSE loss is natural for regression-style reconstruction.

# %%
if not HAS_TF:
    print("[SKIP] TensorFlow not available — skipping Autoencoder section.")
    AE_AVAILABLE = False
else:
    AE_AVAILABLE = True

    def build_autoencoder(n_features):
        """
        Build a dense autoencoder.  Bottleneck dimension 16 forces the encoder
        to learn a compact representation of 'normal' grid behaviour.
        """
        inp = Input(shape=(n_features,), name="ae_input")

        # Encoder
        x = layers.Dense(64, activation="relu", name="enc_1")(inp)
        x = layers.Dense(32, activation="relu", name="enc_2")(x)
        bottleneck = layers.Dense(16, activation="relu",
                                  name="bottleneck")(x)

        # Decoder
        x = layers.Dense(32, activation="relu", name="dec_1")(bottleneck)
        x = layers.Dense(64, activation="relu", name="dec_2")(x)
        out = layers.Dense(n_features, activation="linear",
                           name="ae_output")(x)

        autoencoder = Model(inp, out, name="Autoencoder")
        encoder     = Model(inp, bottleneck, name="Encoder")
        return autoencoder, encoder


    ae_model, enc_model = build_autoencoder(N_FEATURES)
    ae_model.summary()

    ae_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )

    # Train on STABLE samples only — this is the key design choice
    stable_mask_train = y_train_tab == 0
    X_stable_train    = X_train_tab[stable_mask_train]
    print(f"\nAutoencoder training set: {X_stable_train.shape[0]:,} Stable samples")

    ae_callbacks = [
        EarlyStopping(monitor="val_loss", patience=8,
                      restore_best_weights=True, verbose=1),
    ]

    print("Training Autoencoder on Stable data only ...")
    ae_history = ae_model.fit(
        X_stable_train, X_stable_train,    # input == target for reconstruction
        epochs=50,
        batch_size=128,
        validation_split=0.1,
        callbacks=ae_callbacks,
        verbose=1,
    )

    # ── Compute reconstruction errors on full test set ─────────────────────────
    X_test_recon  = ae_model.predict(X_test_tab, verbose=0)
    recon_error   = np.mean((X_test_tab - X_test_recon) ** 2, axis=1)  # MSE per sample

    # ── Reconstruction error histogram by class ────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for lbl, col, style in [(0, "#4CAF50", "solid"),
                             (1, "#FF9800", "dashed"),
                             (2, "#F44336", "dotted")]:
        mask = y_test_tab == lbl
        if mask.sum() == 0:
            continue
        ax.hist(recon_error[mask], bins=50, alpha=0.5, color=col,
                label=LABEL_MAP[lbl], edgecolor="none", density=True,
                linestyle=style)
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Density")
    ax.set_title("Autoencoder Reconstruction Error by Grid State\n"
                 "(trained on Stable only — higher error = more anomalous)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "ae_recon_error_histogram.png"),
                dpi=150)
    plt.close(fig)
    print("Saved: ae_recon_error_histogram.png")

    # ── ROC curve using reconstruction error as anomaly score ─────────────────
    y_binary_collapse = (y_test_tab == 2).astype(int)
    fpr_ae, tpr_ae, thresholds_ae = roc_curve(y_binary_collapse, recon_error)
    auc_ae = auc(fpr_ae, tpr_ae)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_ae, tpr_ae, color="#009688", lw=2,
            label=f"Autoencoder (AUC={auc_ae:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Autoencoder — Collapse Anomaly ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "ae_roc.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: ae_roc.png  (AUC = {auc_ae:.4f})")

    # ── Optimal threshold via Youden's J = max(TPR − FPR) ─────────────────────
    youden_j      = tpr_ae - fpr_ae
    best_idx      = int(np.argmax(youden_j))
    optimal_thresh = float(thresholds_ae[best_idx])
    print(f"\nOptimal threshold (Youden's J): {optimal_thresh:.6f}")
    print(f"  At this threshold: TPR={tpr_ae[best_idx]:.3f}, "
          f"FPR={fpr_ae[best_idx]:.3f}")

    # ── Reconstruction error time-series: fault → collapse scenario ───────────
    # Find a continuous Collapse sequence in test set
    collapse_test_idx = np.where(y_test_tab == 2)[0]

    if len(collapse_test_idx) > 10:
        # Take 60 points centred around a collapse run
        mid = collapse_test_idx[len(collapse_test_idx) // 2]
        start = max(0, mid - 30)
        end   = min(len(y_test_tab), mid + 30)

        ts_recon_err = recon_error[start:end]
        ts_labels    = y_test_tab[start:end]

        fig, ax = plt.subplots(figsize=(10, 4))
        x_ts = np.arange(len(ts_recon_err))
        ax.plot(x_ts, ts_recon_err, color="#009688", lw=2, label="Recon. Error")
        ax.axhline(optimal_thresh, color="#E91E63", linestyle="--",
                   label=f"Threshold ({optimal_thresh:.4f})")
        # Shade by class
        for i, (pos, lbl) in enumerate(zip(x_ts, ts_labels)):
            c = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}[lbl]
            ax.axvspan(pos - 0.5, pos + 0.5, alpha=0.1, color=c, linewidth=0)
        from matplotlib.patches import Patch
        legend_els = [
            ax.get_lines()[0], ax.get_lines()[1],
            Patch(color="#4CAF50", alpha=0.3, label="Stable"),
            Patch(color="#FF9800", alpha=0.3, label="Unstable"),
            Patch(color="#F44336", alpha=0.3, label="Collapse"),
        ]
        ax.legend(handles=legend_els)
        ax.set_xlabel("Sample index"); ax.set_ylabel("Reconstruction Error")
        ax.set_title("Rising Reconstruction Error During Fault → Collapse")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "ae_recon_timeseries.png"),
                    dpi=150)
        plt.close(fig)
        print("Saved: ae_recon_timeseries.png")
    else:
        print("[INFO] Insufficient collapse samples for time-series plot — skipping.")

    print(f"\nAutoencoder Collapse Detection AUC = {auc_ae:.4f}")

# %% [markdown]
# ## 7. Hybrid Pipeline: Autoencoder Score + Random Forest
#
# ### Motivation
# The reconstruction error is an **unsupervised anomaly signal** — it does not
# use any class labels during training.  By adding it as an extra feature to the
# tabular classifier, we combine two sources of information:
# 1. Raw + engineered features (what the measurements look like)
# 2. Reconstruction error (how abnormal those measurements are)
#
# This is especially valuable for detecting early-stage faults that are not yet
# represented in the labelled training distribution.

# %%
if not AE_AVAILABLE if HAS_TF else True:
    print("[SKIP] Autoencoder not available — skipping hybrid pipeline.")
else:
    print("\n=== Hybrid Pipeline: AE Score + Random Forest ===")

    # Compute reconstruction errors for train set too
    X_train_recon    = ae_model.predict(X_train_tab, verbose=0)
    recon_err_train  = np.mean((X_train_tab - X_train_recon) ** 2, axis=1)
    recon_err_test   = recon_error   # already computed in section 6

    # Augment feature matrices
    X_train_hybrid = np.hstack([X_train_tab, recon_err_train[:, None]])
    X_test_hybrid  = np.hstack([X_test_tab,  recon_err_test[:, None]])

    # Baseline RF (no AE score)
    rf_base = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_base.fit(X_train_tab, y_train_tab)
    y_pred_rf_base = rf_base.predict(X_test_tab)
    rep_base = classification_report(
        y_test_tab, y_pred_rf_base, output_dict=True, zero_division=0
    )

    # Hybrid RF (with AE score)
    rf_hybrid = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_hybrid.fit(X_train_hybrid, y_train_tab)
    y_pred_rf_hybrid = rf_hybrid.predict(X_test_hybrid)
    rep_hybrid = classification_report(
        y_test_tab, y_pred_rf_hybrid, output_dict=True, zero_division=0
    )

    print("\nRF baseline (no AE score):")
    print(f"  Macro F1        : {rep_base['macro avg']['f1-score']:.4f}")
    print(f"  Collapse Recall : {rep_base.get('2', {}).get('recall', float('nan')):.4f}")

    print("\nRF hybrid (with AE reconstruction error):")
    print(f"  Macro F1        : {rep_hybrid['macro avg']['f1-score']:.4f}")
    print(f"  Collapse Recall : {rep_hybrid.get('2', {}).get('recall', float('nan')):.4f}")

    improvement = (rep_hybrid['macro avg']['f1-score']
                   - rep_base['macro avg']['f1-score'])
    print(f"\nMacro-F1 improvement from AE augmentation: {improvement:+.4f}")

    # Feature importance: what rank does reconstruction_error get?
    feat_names_hybrid = ENG_FEATURES + ["AE_Recon_Error"]
    importances = rf_hybrid.feature_importances_
    ae_rank = int(np.argsort(importances)[::-1].tolist().index(
        feat_names_hybrid.index("AE_Recon_Error")
    )) + 1
    print(f"AE Recon Error feature importance rank: {ae_rank} / {len(feat_names_hybrid)}")

    # Plot feature importances including AE score
    top_n   = 12
    top_idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = ["#E91E63" if feat_names_hybrid[i] == "AE_Recon_Error"
                  else "#3F51B5" for i in top_idx]
    ax.barh(
        [feat_names_hybrid[i] for i in top_idx][::-1],
        importances[top_idx][::-1],
        color=bar_colors[::-1], alpha=0.85,
    )
    ax.set_xlabel("Feature Importance")
    ax.set_title("Hybrid RF — Top Feature Importances\n"
                 "(pink = AE reconstruction error)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "hybrid_feature_importance.png"),
                dpi=150)
    plt.close(fig)
    print("Saved: hybrid_feature_importance.png")

# %% [markdown]
# ## 8. Attention Visualisation — Transformer Deep-Dive
#
# We extract attention matrices for 5 representative sequences:
# - 1 Stable
# - 2 Unstable
# - 2 Collapse
#
# Each panel shows which query timesteps attend most to which key timesteps.
# High attention weight on early timesteps in a Collapse window indicates the
# model has learnt to look back for the fault inception.

# %%
if not HAS_TF:
    print("[SKIP] TensorFlow not available — skipping attention deep-dive.")
else:
    try:
        # Collect sample indices: 1 stable, 2 unstable, 2 collapse
        sample_specs = [(0, 1, "Stable"), (1, 2, "Unstable"), (2, 2, "Collapse")]
        samples_to_plot = []

        for lbl, n_samples, lbl_name in sample_specs:
            lbl_indices = np.where(y_seq_test == lbl)[0]
            if len(lbl_indices) < n_samples:
                n_samples = len(lbl_indices)
            chosen = lbl_indices[:n_samples]
            for idx in chosen:
                samples_to_plot.append((idx, lbl, lbl_name))

        n_panels = len(samples_to_plot)

        if n_panels == 0:
            print("[INFO] No samples available for attention deep-dive.")
        else:
            # Build sub-model to output attention from block 0
            inp_tensor   = transformer_model.input
            proj_layer   = transformer_model.get_layer("input_projection")
            pos_layer    = transformer_model.get_layer("pos_enc")
            tb0_layer    = transformer_model.get_layer("transformer_block_0")

            proj_out_sym = proj_layer(inp_tensor)
            pos_out_sym  = pos_layer(proj_out_sym)
            pre_model    = Model(inputs=inp_tensor, outputs=pos_out_sym)

            panel_colors = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}
            cols_per_row = min(n_panels, 5)
            n_rows       = math.ceil(n_panels / cols_per_row)

            fig, axes = plt.subplots(n_rows, cols_per_row,
                                     figsize=(cols_per_row * 4, n_rows * 4))
            axes = np.array(axes).ravel() if n_panels > 1 else [axes]

            for panel_i, (seq_idx, lbl, lbl_name) in enumerate(samples_to_plot):
                seq_input = X_seq_test[seq_idx : seq_idx + 1]
                pre_out   = pre_model(seq_input, training=False)
                _, attn_w = tb0_layer(pre_out, training=False,
                                      return_attention_scores=True)
                avg_attn  = attn_w.numpy()[0].mean(axis=0)  # (W, W)

                ax = axes[panel_i]
                im = ax.imshow(avg_attn, cmap="Reds" if lbl == 2 else "Blues",
                               aspect="auto", vmin=0, vmax=avg_attn.max())
                ax.set_title(f"{lbl_name} (idx={seq_idx})",
                             color=panel_colors[lbl], fontweight="bold")
                ax.set_xlabel("Key timestep")
                ax.set_ylabel("Query timestep")
                plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

            # Hide unused panels
            for j in range(n_panels, len(axes)):
                axes[j].set_visible(False)

            fig.suptitle("Transformer Attention Weights — Across Grid States\n"
                         "(average over 4 heads, Block 0)", fontsize=12)
            fig.tight_layout()
            fig.savefig(os.path.join(FIGURES_DIR,
                                     "transformer_attention_multipanel.png"),
                        dpi=150)
            plt.close(fig)
            print("Saved: transformer_attention_multipanel.png")

    except Exception as e:
        print(f"[WARN] Attention deep-dive failed: {e}")

# %% [markdown]
# ## 9. Model Comparison — Full Cross-Notebook Summary
#
# We consolidate metrics from all three notebooks into a single comparison table
# and visualise them.
#
# ### Deployment recommendation
#
# | Use case | Recommended model | Reason |
# |---|---|---|
# | Real-time online inference (<1 ms/sample) | Random Forest (nb1) | Fast tabular inference, no GPU needed |
# | Best accuracy, GPU available | Transformer or TCN (nb3) | Learns temporal dependencies |
# | Early anomaly pre-screening | Autoencoder (nb3) | Works without labels |
# | Interpretability required | LR or RF with SHAP (nb1) | Coefficients/importances are human-readable |
# | Embedded / edge device | SVM-RBF (nb2, small C) | Compact, deterministic inference |

# %%
# ── Collect nb3 metrics ────────────────────────────────────────────────────────
nb3_results = []

def tabular_eval(y_true, y_pred, y_prob_col2, model_name):
    """Build a metrics dict from evaluation arrays."""
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    try:
        fpr_, tpr_, _ = roc_curve((y_true == 2).astype(int), y_prob_col2)
        collapse_auc  = round(auc(fpr_, tpr_), 4)
    except Exception:
        collapse_auc = float("nan")
    return {
        "Model":           model_name,
        "Accuracy":        round(rep["accuracy"], 4),
        "Macro_Precision": round(rep["macro avg"]["precision"], 4),
        "Macro_Recall":    round(rep["macro avg"]["recall"], 4),
        "Macro_F1":        round(rep["macro avg"]["f1-score"], 4),
        "Collapse_Recall": round(rep.get("2", {}).get("recall", float("nan")), 4),
        "Collapse_AUC":    collapse_auc,
    }


if HAS_TF:
    # TCN
    nb3_results.append(
        tabular_eval(y_seq_test, y_pred_tcn, y_prob_tcn[:, 2], "TCN (nb3)")
    )
    # Transformer
    nb3_results.append(
        tabular_eval(y_seq_test, y_pred_tf, y_prob_tf[:, 2], "Transformer (nb3)")
    )
    # Autoencoder — reported as anomaly-detection AUC, not classification metrics
    nb3_results.append({
        "Model":           "Autoencoder AE (nb3)",
        "Accuracy":        float("nan"),
        "Macro_Precision": float("nan"),
        "Macro_Recall":    float("nan"),
        "Macro_F1":        float("nan"),
        "Collapse_Recall": float("nan"),
        "Collapse_AUC":    round(auc_ae, 4),
    })
    # Hybrid RF
    if AE_AVAILABLE:
        nb3_results.append(
            tabular_eval(y_test_tab, y_pred_rf_hybrid,
                         rf_hybrid.predict_proba(X_test_hybrid)[:, 2],
                         "Hybrid RF (nb3)")
        )

# ── Load nb1 and nb2 results if available ─────────────────────────────────────
all_nb_results = nb3_results.copy()

for nb_csv, nb_tag in [
    (os.path.join(MODEL_DIR, "nb2_results.csv"), "nb2"),
    (os.path.join(MODEL_DIR, "nb1_results.csv"), "nb1"),
]:
    if os.path.exists(nb_csv):
        prev_df = pd.read_csv(nb_csv)
        all_nb_results.extend(prev_df.to_dict(orient="records"))
        print(f"Loaded {nb_tag} results from {nb_csv}")
    else:
        print(f"[INFO] {nb_tag} results not found at {nb_csv} — skipping.")

full_comparison = pd.DataFrame(all_nb_results)
full_comparison = full_comparison.drop_duplicates(subset=["Model"]).reset_index(drop=True)

print("\n" + "=" * 70)
print("FULL CROSS-NOTEBOOK MODEL COMPARISON")
print("=" * 70)
print(full_comparison.to_string(index=False))

# ── Grouped bar chart ─────────────────────────────────────────────────────────
# Plot only models where both Macro_F1 and Collapse_Recall are numeric
plot_df = full_comparison.dropna(subset=["Macro_F1", "Collapse_Recall"]).copy()

if len(plot_df) > 0:
    model_names   = plot_df["Model"].tolist()
    macro_f1s     = plot_df["Macro_F1"].tolist()
    collapse_recs = plot_df["Collapse_Recall"].tolist()
    x = np.arange(len(model_names))
    w = 0.4

    fig, ax = plt.subplots(figsize=(max(10, len(model_names) * 1.5), 5))
    b1 = ax.bar(x - w / 2, macro_f1s,     w, label="Macro F1",
                color="#3F51B5", alpha=0.8)
    b2 = ax.bar(x + w / 2, collapse_recs, w, label="Collapse Recall",
                color="#F44336", alpha=0.8)

    for bar in b1.patches + b2.patches:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title("All-Notebook Model Comparison — Macro F1 & Collapse Recall")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "full_comparison_bar.png"), dpi=150)
    plt.close(fig)
    print("Saved: full_comparison_bar.png")

# ── Metric × model heatmap ────────────────────────────────────────────────────
METRIC_COLS = ["Accuracy", "Macro_Precision", "Macro_Recall",
               "Macro_F1", "Collapse_Recall", "Collapse_AUC"]

# Only rows that have at least half the metrics filled in
heatmap_df = full_comparison.set_index("Model")[METRIC_COLS].astype(float)
heatmap_df = heatmap_df.dropna(thresh=3)

if len(heatmap_df) > 0:
    fig, ax = plt.subplots(
        figsize=(len(METRIC_COLS) * 1.4, max(4, len(heatmap_df) * 0.7))
    )
    data_arr = heatmap_df.values

    im = ax.imshow(data_arr, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(METRIC_COLS)))
    ax.set_xticklabels(METRIC_COLS, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(heatmap_df)))
    ax.set_yticklabels(heatmap_df.index.tolist(), fontsize=9)
    ax.set_title("Model × Metric Heatmap (all notebooks)", fontsize=11)

    for i in range(len(heatmap_df)):
        for j in range(len(METRIC_COLS)):
            val = data_arr[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color="black" if 0.3 < val < 0.8 else "white")

    plt.colorbar(im, ax=ax, label="Score")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "metric_heatmap.png"), dpi=150)
    plt.close(fig)
    print("Saved: metric_heatmap.png")

# %% [markdown]
# ## 10. Save Models and Results

# %%
print("\n=== Saving Models ===")

# ── Keras models ───────────────────────────────────────────────────────────────
if HAS_TF:
    tcn_path = os.path.join(MODEL_DIR, "nb3_tcn")
    transformer_path = os.path.join(MODEL_DIR, "nb3_transformer")
    ae_path  = os.path.join(MODEL_DIR, "nb3_autoencoder")

    try:
        tcn_model.save(tcn_path)
        size_mb = sum(
            os.path.getsize(os.path.join(dirpath, f))
            for dirpath, _, files in os.walk(tcn_path)
            for f in files
        ) / (1024 * 1024)
        print(f"Saved TCN           → {tcn_path}  ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"[WARN] TCN save failed: {e}")

    try:
        transformer_model.save(transformer_path)
        size_mb = sum(
            os.path.getsize(os.path.join(dirpath, f))
            for dirpath, _, files in os.walk(transformer_path)
            for f in files
        ) / (1024 * 1024)
        print(f"Saved Transformer   → {transformer_path}  ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"[WARN] Transformer save failed: {e}")

    try:
        ae_model.save(ae_path)
        size_mb = sum(
            os.path.getsize(os.path.join(dirpath, f))
            for dirpath, _, files in os.walk(ae_path)
            for f in files
        ) / (1024 * 1024)
        print(f"Saved Autoencoder   → {ae_path}  ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"[WARN] Autoencoder save failed: {e}")

    # Save hybrid RF
    if AE_AVAILABLE:
        hybrid_rf_path = os.path.join(MODEL_DIR, "nb3_hybrid_rf.pkl")
        with open(hybrid_rf_path, "wb") as f:
            pickle.dump(rf_hybrid, f)
        size_mb = os.path.getsize(hybrid_rf_path) / (1024 * 1024)
        print(f"Saved Hybrid RF     → {hybrid_rf_path}  ({size_mb:.1f} MB)")

# ── Results CSV ───────────────────────────────────────────────────────────────
nb3_results_path = os.path.join(MODEL_DIR, "nb3_results.csv")
pd.DataFrame(nb3_results).to_csv(nb3_results_path, index=False)
print(f"\nNb3 results saved → {nb3_results_path}")

# Full comparison
full_comp_path = os.path.join(MODEL_DIR, "all_notebooks_comparison.csv")
full_comparison.to_csv(full_comp_path, index=False)
print(f"Full comparison   → {full_comp_path}")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("NOTEBOOK 3 COMPLETE")
print("=" * 60)

if nb3_results:
    nb3_df = pd.DataFrame(nb3_results).dropna(subset=["Macro_F1"])
    if len(nb3_df) > 0:
        best_row = nb3_df.loc[nb3_df["Macro_F1"].idxmax()]
        print(f"Best model (Macro F1): {best_row['Model']}  F1={best_row['Macro_F1']:.4f}")

    nb3_df_cr = pd.DataFrame(nb3_results).dropna(subset=["Collapse_Recall"])
    if len(nb3_df_cr) > 0:
        best_cr = nb3_df_cr.loc[nb3_df_cr["Collapse_Recall"].idxmax()]
        print(f"Best Collapse Recall : {best_cr['Model']}  Recall={best_cr['Collapse_Recall']:.4f}")

if HAS_TF and AE_AVAILABLE:
    print(f"Autoencoder Collapse AUC : {auc_ae:.4f}")

print(f"\nFigures → {os.path.abspath(FIGURES_DIR)}")
print(f"Models  → {os.path.abspath(MODEL_DIR)}")
print("\nDeployment guidance:")
print("  Real-time (<1ms/sample)  → Random Forest (models/random_forest.pkl)")
print("  Best accuracy, GPU       → Transformer   (models/nb3_transformer/)")
print("  Unsupervised screening   → Autoencoder   (models/nb3_autoencoder/)")
print("  Interpretability / audit → LR / RF with SHAP  (nb1)")
print("  Edge / embedded device   → SVM-RBF       (models/nb2_svm.pkl)")
