# ============================================================
#  Nigeria Power Grid Instability Prediction — Streamlit App
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score
)
from sklearn.neural_network import MLPClassifier

# ── Optional dependencies ──────────────────────────────────────────────────────
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from imblearn.over_sampling import BorderlineSMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


# ── Constants ──────────────────────────────────────────────────────────────────
CSV_AUG  = "https://raw.githubusercontent.com/Okwybobby/AI-Driven-Prediction-of-Power-Grid-Instability/refs/heads/main/nigeria_grid_stability_main.csv"
CSV_SEED = "https://raw.githubusercontent.com/Okwybobby/AI-Driven-Prediction-of-Power-Grid-Instability/refs/heads/main/nigeria_grid_stability_dataset.csv"
REQUIRED = [
    "Frequency_Hz","Voltage_pu","Angle_deg","Active_Power_MW",
    "Reactive_Power_MVAr","ROCOF_Hz_per_s","Ambient_Temp_C","Stability_Label"
]
ENG_FEATURES = [
    "Frequency_Hz","Voltage_pu","Angle_deg","Active_Power_MW",
    "Reactive_Power_MVAr","ROCOF_Hz_per_s","Ambient_Temp_C",
    "Freq_Dev","V_Dev","Freq_sq","V_sq","Apparent_S",
    "PQ_Ratio","Power_Factor","FxV","ROCOF_abs","ROCOF_sq"
]
LABEL_NAMES  = ["Stable","Unstable","Collapse"]
CLASS_COLORS = ["#059669","#D97706","#DC2626"]
PALETTE      = {"Stable":"#059669","Unstable":"#D97706","Collapse":"#DC2626"}

PAGE_TITLES = [
    "Setup & Configuration",
    "Data Loading & Validation",
    "Exploratory Data Analysis",
    "Feature Engineering",
    "Preprocessing",
    "Logistic Regression",
    "Random Forest",
    "XGBoost / GBM",
    "MLP Neural Network",
    "Model Comparison",
    "Seed Dataset Validation",
    "Alert System",
    "Cross-Validation",
    "Summary & Export",
]
N_PAGES = len(PAGE_TITLES)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nigeria Grid Instability AI",
    page_icon="⚡",
    layout="wide"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Compact layout: strip Streamlit's default excess whitespace ── */
.block-container,
[data-testid="block-container"],
[data-testid="stMainBlockContainer"] > div {
    padding-top: 4rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}
[data-testid="stAppViewBlockContainer"],
[data-testid="stMainBlockContainer"] {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}
/* Tighten vertical gaps between Streamlit elements */
[data-testid="stVerticalBlock"] {
    gap: 0.35rem !important;
}
/* Shrink default element container margins */
.element-container {
    margin-bottom: 0.1rem !important;
}
/* Compact the horizontal rule used by st.markdown("---") */
hr {
    margin: 0.4rem 0 !important;
    border-color: #E2E8F0 !important;
}
/* Remove excess padding on tab content */
[data-testid="stTabContent"] {
    padding-top: 0.6rem !important;
    padding-bottom: 0 !important;
}
/* Reduce sidebar padding */
[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebarContent"],
section[data-testid="stSidebar"] > div {
    padding-top: 0.0125rem !important;
    padding-bottom: 0.5rem !important;
}
/* Pull sidebar content up as far as possible */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .element-container:first-child,
[data-testid="stSidebar"] .element-container:first-child h1,
[data-testid="stSidebar"] .element-container:first-child h2,
[data-testid="stSidebar"] .element-container:first-child p {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child > div:first-child {
    transform: translateY(-4rem);
}
/* Tighten sidebar button gaps */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.1rem !important;
}
/* Reduce st.metric default padding */
[data-testid="stMetricValue"] { line-height: 1.2 !important; }
[data-testid="metric-container"] { padding: 0.5rem 0.75rem !important; }

/* ── Light theme base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #F0F4F8;
    color: #0F172A;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 3px solid #2563EB;
    box-shadow: 2px 0 12px rgba(37,99,235,0.08);
}
[data-testid="stSidebar"] * {
    color: #0F172A !important;
}

/* ── Main content cards ── */
[data-testid="stMainBlockContainer"] {
    background-color: #F0F4F8;
}

/* ── Page headers ── */
.main-header {
    font-size: 1.65rem;
    font-weight: 800;
    color: #1E3A5F;
    letter-spacing: 0.3px;
    margin: 2px 0 15px 0;
    line-height: 1.2;
}
.sub-header {
    font-size: 0.92rem;
    color: #64748B;
    margin: 0 0 10px 0;
    line-height: 1.4;
}

/* ── Metric cards ── */
.metric-card {
    background: #FFFFFF;
    border: none;
    border-left: 4px solid #2563EB;
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
}
.metric-card .metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #2563EB;
    line-height: 1.2;
}
.metric-card .metric-label {
    font-size: 0.8rem;
    color: #64748B;
    margin-top: 2px;
}

/* ── Sidebar badge colors ── */
.badge-complete   { color: #059669; font-weight: 700; }
.badge-current    { color: #2563EB; font-weight: 700; }
.badge-pending    { color: #94A3B8; }

/* ── Step buttons in sidebar ── */
.step-btn button {
    background: transparent !important;
    border: none !important;
    text-align: left !important;
    width: 100% !important;
    padding: 4px 8px !important;
    font-size: 0.82rem !important;
}

/* ── Alert rows ── */
.alert-critical { background-color: rgba(220,38,38,0.10); }
.alert-warning  { background-color: rgba(217,119,6,0.10); }

/* ── Streamlit overrides ── */
div[data-testid="stHorizontalBlock"] { gap: 12px; }
div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ── Expander & containers ── */
[data-testid="stExpander"] {
    background: #FFFFFF;
    border-radius: 10px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}

/* ── Buttons: force styles on all devices including mobile ── */
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-secondary"] > button,
button[kind="secondary"],
div.stButton > button {
    -webkit-appearance: none !important;
    appearance: none !important;
    background-color: #FFFFFF !important;
    color: #1E3A5F !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    -webkit-tap-highlight-color: transparent !important;
}
div.stButton > button:hover,
div.stButton > button:focus,
div.stButton > button:active {
    background-color: #F1F5F9 !important;
    color: #1E3A5F !important;
    border-color: #2563EB !important;
}
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primary"] > button,
button[kind="primary"],
div.stButton > button[kind="primary"] {
    -webkit-appearance: none !important;
    appearance: none !important;
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    -webkit-tap-highlight-color: transparent !important;
}
button[kind="primary"]:hover,
button[kind="primary"]:active,
button[kind="primary"]:focus {
    background-color: #1D4ED8 !important;
    color: #FFFFFF !important;
}

/* ── Info / success / warning boxes ── */
[data-testid="stAlert"] {
    border-radius: 8px;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #2563EB !important;
    border-bottom-color: #2563EB !important;
    font-weight: 700;
}
[data-testid="stTabs"] {
    margin-bottom: 0 !important;
}

/* ── Reduce markdown heading margins ── */
h1, h2, h3, h4, h5 {
    margin-top: 0.4rem !important;
    margin-bottom: 0.15rem !important;
    line-height: 1.25 !important;
}
/* ── Tighten paragraph spacing ── */
p { margin-bottom: 0.25rem !important; }

/* ── Reduce stMarkdown / stText container gaps ── */
[data-testid="stMarkdownContainer"] > p:last-child { margin-bottom: 0 !important; }

/* ── Remove top margin from first child inside columns ── */
[data-testid="column"] > div > div > div > .element-container:first-child {
    margin-top: 0 !important;
}

/* ── st.success / st.warning / st.info height ── */
[data-testid="stAlert"] {
    padding: 0.5rem 0.8rem !important;
    border-radius: 8px;
}

/* ── Plotly chart container: no extra margin ── */
[data-testid="stPlotlyChart"] {
    margin-bottom: 0 !important;
}

/* ── Dataframe container: no extra margin ── */
[data-testid="stDataFrame"] {
    margin-bottom: 0 !important;
}

/* ── st.columns gap ── */
[data-testid="stHorizontalBlock"] {
    gap: 0.75rem !important;
    align-items: start !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────
_defaults = dict(
    page=0,
    completed=set(),
    config=dict(LSTM_WINDOW=20, T_HORIZON=5, RANDOM_STATE=42),
    df=None, df_seed=None,
    df_eng=None, df_seed_eng=None,
    X_train_sc=None, X_test_sc=None,
    y_train=None, y_test=None,
    X_train_sm=None, y_train_sm=None,
    scaler=None,
    class_weights=None,
    lr_model=None, rf_model=None,
    boost_model=None, boost_name=None, boost_importance=None,
    lstm_model=None, lstm_history=None,
    y_pred_lr=None, y_pred_rf=None, y_pred_boost=None,
    y_proba_lr=None, y_proba_rf=None, y_proba_boost=None,
    y_pred_lstm=None, y_proba_lstm=None,
    y_seq_te=None,
    results=dict(),
    cv_results=None,
    alerts_df=None,
)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ────────────────────────────────────────────────────────────────────
def mark_complete(idx: int):
    st.session_state.completed.add(idx)

def go_to(idx: int):
    st.session_state.page = idx
    st.rerun()

def sep():
    """Lightweight section separator — much less vertical space than st.markdown('---')."""
    st.markdown(
        '<hr style="border:none;border-top:1px solid #E2E8F0;margin:6px 0;">',
        unsafe_allow_html=True
    )

def metric_card(label, value, col):
    col.markdown(
        f'<div class="metric-card"><div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True
    )

def light_fig(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        font_color="#0F172A",
        font=dict(family="Inter, Segoe UI, sans-serif"),
        title_font=dict(color="#1E3A5F", size=14, family="Inter, Segoe UI, sans-serif"),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0", borderwidth=1),
        xaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1", tickfont=dict(color="#475569")),
        yaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1", tickfont=dict(color="#475569")),
    )
    return fig

# Keep alias for any internal callers
dark_fig = light_fig

def confusion_heatmap(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm, text_auto=True,
        x=LABEL_NAMES, y=LABEL_NAMES,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual"),
        title=title
    )
    dark_fig(fig)
    fig.update_layout(width=400, height=320)
    return fig

def roc_curves_fig(models_dict, y_test):
    """models_dict = {name: y_proba}  y_test must be 1-D integer labels"""
    classes = sorted(np.unique(y_test))
    y_bin = label_binarize(y_test, classes=classes)
    fig = make_subplots(
        rows=1, cols=len(models_dict),
        subplot_titles=list(models_dict.keys())
    )
    for col_i, (name, proba) in enumerate(models_dict.items(), 1):
        for ci, cn in enumerate(LABEL_NAMES):
            fpr, tpr, _ = roc_curve(y_bin[:, ci], proba[:, ci])
            auc_v = roc_auc_score(y_bin[:, ci], proba[:, ci])
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr,
                           name=f"{cn} (AUC={auc_v:.2f})",
                           line=dict(color=CLASS_COLORS[ci])),
                row=1, col=col_i
            )
        fig.add_trace(
            go.Scatter(x=[0,1], y=[0,1], mode="lines",
                       line=dict(dash="dash", color="#555"),
                       showlegend=False),
            row=1, col=col_i
        )
    dark_fig(fig)
    fig.update_layout(height=340)
    return fig

# ── engineer_features ──────────────────────────────────────────────────────────
def engineer_features(data):
    d = data.copy(); eps = 1e-6
    d["Freq_Dev"]     = (d["Frequency_Hz"] - 50.0).abs()
    d["V_Dev"]        = (d["Voltage_pu"] - 1.0).abs()
    d["Freq_sq"]      = d["Freq_Dev"] ** 2
    d["V_sq"]         = d["V_Dev"] ** 2
    d["Apparent_S"]   = np.sqrt(d["Active_Power_MW"]**2 + d["Reactive_Power_MVAr"]**2)
    d["PQ_Ratio"]     = d["Active_Power_MW"] / (d["Reactive_Power_MVAr"].abs() + eps)
    d["Power_Factor"] = d["Active_Power_MW"] / (d["Apparent_S"] + eps)
    d["FxV"]          = d["Frequency_Hz"] * d["Voltage_pu"]
    d["ROCOF_abs"]    = d["ROCOF_Hz_per_s"].abs()
    d["ROCOF_sq"]     = d["ROCOF_Hz_per_s"] ** 2
    d.replace([np.inf, -np.inf], 0.0, inplace=True)
    d.fillna(0.0, inplace=True)
    return d

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading datasets from GitHub…")
def load_datasets():
    df      = pd.read_csv(CSV_AUG)
    df_seed = pd.read_csv(CSV_SEED)
    return df, df_seed

def build_sequences(X, y, window):
    xs, ys = [], []
    for i in range(len(X) - window):
        xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(xs), np.array(ys)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚡AI-Driven Prediction of Power Grid Instability in Nigeria")
        sep()
        cur  = st.session_state.page
        done = st.session_state.completed
        prog = len(done)
        st.progress(prog / N_PAGES, text=f"{prog}/{N_PAGES} complete")
        st.markdown("### Pipeline Steps")
        for i, title in enumerate(PAGE_TITLES):
            if i in done:
                icon = "✅"
                cls  = "badge-complete"
            elif i == cur:
                icon = "▶️"
                cls  = "badge-current"
            else:
                icon = "○"
                cls  = "badge-pending"
            label = f"{icon} {i}. {title}"
            if st.button(label, key=f"nav_{i}", use_container_width=True):
                go_to(i)

# ── NAV FOOTER ─────────────────────────────────────────────────────────────────
def render_nav():
    cur = st.session_state.page
    sep()
    c1, c2, c3 = st.columns([1, 3, 1])
    with c1:
        if cur > 0:
            if st.button("◀ Previous", use_container_width=True):
                st.session_state.page -= 1
                st.rerun()
        else:
            st.button("◀ Previous", disabled=True, use_container_width=True)
    with c2:
        dots = ""
        for i in range(N_PAGES):
            if i == cur:
                dots += "🔵"
            elif i in st.session_state.completed:
                dots += "🟢"
            else:
                dots += "⚪"
        st.markdown(
            f'<div style="text-align:center;font-size:0.6rem;letter-spacing:2px">{dots}</div>',
            unsafe_allow_html=True
        )
    with c3:
        if cur < N_PAGES - 1:
            if st.button("Next ▶", use_container_width=True):
                st.session_state.page += 1
                st.rerun()
        else:
            st.button("Next ▶", disabled=True, use_container_width=True)

# ==============================================================
#  PAGE RENDERERS
# ==============================================================

# ── Page 0 ────────────────────────────────────────────────────
def page_0():
    st.markdown('<div class="main-header">⚡ Setup & Configuration</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Configure pipeline hyper-parameters and verify dependencies.</div>', unsafe_allow_html=True)

    cfg = st.session_state.config

    col1, spacer, col2 = st.columns([1, 0.1, 1])
    with col1:
        lstm_win = st.slider("LSTM Window (time-steps)", 5, 50, cfg.get("LSTM_WINDOW", 20), key="s_lstm_win")
        t_hor    = st.slider("Prediction Horizon T", 1, 20, cfg.get("T_HORIZON", 5), key="s_t_hor")
        rand_st  = st.slider("Random State", 0, 9999, cfg.get("RANDOM_STATE", 42), key="s_rand")
    with col2:
        st.markdown("### Dependency Status")
        deps = {
            "NumPy":            True,
            "Pandas":           True,
            "scikit-learn":     True,
            "Plotly":           True,
            "XGBoost":          HAS_XGB,
            "imbalanced-learn": HAS_IMBLEARN,
            "MLP (sklearn)":    True,
        }
        for dep, ok in deps.items():
            icon = "✅" if ok else "⚠️ (optional)"
            st.markdown(f"- **{dep}** {icon}")

    sep()
    st.markdown("### Target Labels")
    st.markdown("""
| Label | Name | Frequency | Voltage |
|---|---|---|---|
| 0 | **Stable** | 49.5 – 50.5 Hz | ≥ 0.97 pu |
| 1 | **Unstable** | Outside nominal | 0.90 – 0.97 pu |
| 2 | **Collapse** | < 48.5 or > 51.5 Hz | < 0.80 pu |
""")

    st.markdown('<br/>', unsafe_allow_html=True)

    if st.button("💾 Save Configuration", type="primary"):
        st.session_state.config = dict(
            LSTM_WINDOW=lstm_win,
            T_HORIZON=t_hor,
            RANDOM_STATE=rand_st
        )
        mark_complete(0)
        st.success(f"Configuration saved — Window={lstm_win}, Horizon={t_hor}, Seed={rand_st}")

# ── Page 1 ────────────────────────────────────────────────────
def page_1():
    st.markdown('<div class="main-header">📂 Data Loading & Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Load both CSVs from GitHub and run basic sanity checks.</div>', unsafe_allow_html=True)

    if st.button("🔄 Load Datasets", type="primary"):
        df, df_seed = load_datasets()
        # Validate
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            st.error(f"Missing columns in main dataset: {missing}")
            return
        df["Stability_Label"] = df["Stability_Label"].astype(int)
        df_seed["Stability_Label"] = df_seed["Stability_Label"].astype(int)
        st.session_state.df      = df
        st.session_state.df_seed = df_seed
        mark_complete(1)
        st.success("Datasets loaded successfully!")

    df      = st.session_state.df
    df_seed = st.session_state.df_seed
    if df is None:
        st.info("Click **Load Datasets** to begin.")
        return

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    metric_card("Main Rows",   f"{df.shape[0]:,}",  c1)
    metric_card("Main Cols",   f"{df.shape[1]}",     c2)
    metric_card("Seed Rows",   f"{df_seed.shape[0]:,}", c3)
    metric_card("Null Cells",  f"{df.isnull().sum().sum()}", c4)
    sep()

    # Tabs
    t_main, t_seed, t_stats, t_dist = st.tabs(["Main Dataset","Seed Dataset","Feature Stats","Label Distribution"])
    with t_main:
        st.dataframe(df.head(50), use_container_width=True)
    with t_seed:
        st.dataframe(df_seed.head(50), use_container_width=True)
    with t_stats:
        st.dataframe(df[REQUIRED[:-1]].describe().T.round(4), use_container_width=True)
    with t_dist:
        vc = df["Stability_Label"].value_counts().sort_index()
        vc_df = pd.DataFrame({
            "Label": [LABEL_NAMES[i] for i in vc.index],
            "Count": vc.values,
            "Pct":   (vc.values / vc.values.sum() * 100).round(2)
        })
        fig = px.bar(
            vc_df, x="Label", y="Count", color="Label",
            color_discrete_map=PALETTE, title="Label Distribution",
            text="Count"
        )
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(vc_df, use_container_width=True)
        null_total = df.isnull().sum().sum()
        if null_total == 0:
            st.success("✅ No null values detected in main dataset.")
        else:
            st.warning(f"⚠️ {null_total} null values found — will be handled in preprocessing.")

# ── Page 2 ────────────────────────────────────────────────────
def page_2():
    st.markdown('<div class="main-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    df = st.session_state.df
    if df is None:
        st.warning("⚠️ Please load data first (Page 1).")
        return

    label_col = "Stability_Label"
    name_map   = {0:"Stable", 1:"Unstable", 2:"Collapse"}
    df = df.copy()
    df["Label"] = df[label_col].map(name_map)

    tabs = st.tabs(["Class Distribution","ROCOF Analysis","Time-Series","Correlation Heatmap","Pairplot"])

    # ── Tab 1
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        with c1:
            vc = df["Label"].value_counts()
            fig = px.bar(x=vc.index, y=vc.values, color=vc.index,
                         color_discrete_map=PALETTE,
                         labels={"x":"Class","y":"Count"}, title="Class Counts")
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(df, x="Frequency_Hz", color="Label",
                               color_discrete_map=PALETTE,
                               barmode="overlay", opacity=0.7,
                               title="Frequency by Class")
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            fig = px.histogram(df, x="Voltage_pu", color="Label",
                               color_discrete_map=PALETTE,
                               barmode="overlay", opacity=0.7,
                               title="Voltage by Class")
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2
    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="ROCOF_Hz_per_s", color="Label",
                               color_discrete_map=PALETTE,
                               barmode="overlay", opacity=0.7,
                               title="ROCOF Distribution by Class")
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(df, x="ROCOF_Hz_per_s", y="Label",
                         color="Label", color_discrete_map=PALETTE,
                         orientation="h", title="ROCOF Box-Plot by Class")
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3
    with tabs[2]:
        scenario = st.selectbox("Scenario", ["Normal (Stable)", "Fault (Collapse)", "Both"])
        if scenario == "Normal (Stable)":
            sub = df[df["Label"] == "Stable"].head(200).reset_index(drop=True)
            frames = {"Normal": sub}
        elif scenario == "Fault (Collapse)":
            sub = df[df["Label"] == "Collapse"].head(200).reset_index(drop=True)
            frames = {"Collapse": sub}
        else:
            s1 = df[df["Label"] == "Stable"].head(100).reset_index(drop=True)
            s2 = df[df["Label"] == "Collapse"].head(100).reset_index(drop=True)
            frames = {"Stable": s1, "Collapse": s2}

        for sc_name, sc_df in frames.items():
            st.markdown(f"#### Scenario: {sc_name}")
            col1, col2, col3 = st.columns(3)
            for feat, ref, col in [
                ("Frequency_Hz", 50.0, col1),
                ("Voltage_pu",   1.0,  col2),
                ("ROCOF_Hz_per_s", 0.0, col3),
            ]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=sc_df[feat], mode="lines",
                    name=feat, line=dict(color="#2563EB")
                ))
                fig.add_hline(y=ref, line_dash="dash",
                              line_color="#e74c3c", annotation_text=f"Ref {ref}")
                dark_fig(fig)
                fig.update_layout(title=feat, height=280)
                col.plotly_chart(fig, use_container_width=True)

    # ── Tab 4
    with tabs[3]:
        num_cols = [c for c in REQUIRED[:-1] if pd.api.types.is_numeric_dtype(df[c])]
        corr = df[num_cols].corr().round(3)
        fig = px.imshow(
            corr, text_auto=True, color_continuous_scale="RdBu_r",
            title="Feature Correlation Heatmap"
        )
        dark_fig(fig)
        fig.update_layout(height=460)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 5
    with tabs[4]:
        pair_cols = ["Frequency_Hz","Voltage_pu","ROCOF_Hz_per_s","Active_Power_MW"]
        fig = px.scatter_matrix(
            df, dimensions=pair_cols, color="Label",
            color_discrete_map=PALETTE,
            title="Scatter Matrix (Pairplot)"
        )
        fig.update_traces(diagonal_visible=False, showupperhalf=False)
        dark_fig(fig)
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

    mark_complete(2)

# ── Page 3 ────────────────────────────────────────────────────
def page_3():
    st.markdown('<div class="main-header">⚙️ Feature Engineering</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Derive 10 physics-informed features from the 7 raw signals.</div>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first (Page 1).")
        return

    # Feature table
    feat_info = [
        ("Freq_Dev",     "|f – 50|",                       "Absolute deviation from nominal frequency"),
        ("V_Dev",        "|V – 1.0|",                      "Absolute deviation from nominal voltage"),
        ("Freq_sq",      "Freq_Dev²",                      "Quadratic frequency stress indicator"),
        ("V_sq",         "V_Dev²",                         "Quadratic voltage stress indicator"),
        ("Apparent_S",   "√(P² + Q²)",                     "Apparent power magnitude (MVA)"),
        ("PQ_Ratio",     "P / (|Q| + ε)",                  "Active-to-reactive power ratio"),
        ("Power_Factor", "P / (S + ε)",                    "Power factor (efficiency metric)"),
        ("FxV",          "Frequency × Voltage",            "Coupled frequency-voltage interaction"),
        ("ROCOF_abs",    "|ROCOF|",                        "Absolute rate-of-change of frequency"),
        ("ROCOF_sq",     "ROCOF²",                         "Squared ROCOF (emphasises extremes)"),
    ]
    feat_df = pd.DataFrame(feat_info, columns=["Feature","Formula","Rationale"])
    st.dataframe(feat_df, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    metric_card("Base Features",       "7",  c1)
    metric_card("Engineered Features", "10", c2)
    metric_card("Total Features",      "17", c3)
    sep()

    if st.button("🔧 Apply Feature Engineering", type="primary"):
        df      = st.session_state.df
        df_seed = st.session_state.df_seed.copy()
        # Seed dataset lacks ROCOF_Hz_per_s — fill with 0 before engineering
        if "ROCOF_Hz_per_s" not in df_seed.columns:
            df_seed["ROCOF_Hz_per_s"] = 0.0
        st.session_state.df_eng      = engineer_features(df)
        st.session_state.df_seed_eng = engineer_features(df_seed)
        mark_complete(3)
        st.success("Feature engineering applied to both datasets!")

    if st.session_state.df_eng is not None:
        eng = st.session_state.df_eng
        st.dataframe(eng[ENG_FEATURES + ["Stability_Label"]].head(30), use_container_width=True)

        sel = st.selectbox("Inspect feature distribution", ENG_FEATURES)
        name_map = {0:"Stable", 1:"Unstable", 2:"Collapse"}
        tmp = eng.copy()
        tmp["Label"] = tmp["Stability_Label"].map(name_map)
        fig = px.histogram(
            tmp, x=sel, color="Label",
            color_discrete_map=PALETTE,
            barmode="overlay", opacity=0.75,
            title=f"{sel} by Class"
        )
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

# ── Page 4 ────────────────────────────────────────────────────
def page_4():
    st.markdown('<div class="main-header">🔬 Preprocessing</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Train/test split, standardisation, optional SMOTE oversampling.</div>', unsafe_allow_html=True)

    if st.session_state.df_eng is None:
        st.warning("⚠️ Please run Feature Engineering first (Page 3).")
        return

    test_pct = st.slider("Test set size (%)", 10, 40, 20, key="s_test_pct")
    use_smote = st.checkbox("Apply BorderlineSMOTE", value=False)

    if use_smote and not HAS_IMBLEARN:
        st.warning("imbalanced-learn not installed. SMOTE will be skipped; class_weight='balanced' used.")

    if st.button("⚙️ Preprocess Data", type="primary"):
        eng  = st.session_state.df_eng
        rs   = st.session_state.config.get("RANDOM_STATE", 42)
        X    = eng[ENG_FEATURES].values
        y    = eng["Stability_Label"].values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_pct/100, random_state=rs, stratify=y
        )
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc  = scaler.transform(X_te)

        cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        class_weights = dict(enumerate(cw))

        X_sm, y_sm = X_tr_sc, y_tr
        if use_smote and HAS_IMBLEARN:
            try:
                smote = BorderlineSMOTE(random_state=rs)
                X_sm, y_sm = smote.fit_resample(X_tr_sc, y_tr)
            except Exception as e:
                st.warning(f"SMOTE failed: {e}")

        st.session_state.X_train_sc  = X_tr_sc
        st.session_state.X_test_sc   = X_te_sc
        st.session_state.y_train     = y_tr
        st.session_state.y_test      = y_te
        st.session_state.X_train_sm  = X_sm
        st.session_state.y_train_sm  = y_sm
        st.session_state.scaler      = scaler
        st.session_state.class_weights = class_weights
        mark_complete(4)
        st.success("Preprocessing complete!")

    if st.session_state.X_train_sc is not None:
        X_tr = st.session_state.X_train_sc
        X_te = st.session_state.X_test_sc
        y_tr = st.session_state.y_train
        y_te = st.session_state.y_test
        X_sm = st.session_state.X_train_sm
        y_sm = st.session_state.y_train_sm
        cw   = st.session_state.class_weights

        c1, c2, c3 = st.columns(3)
        metric_card("Train Samples", f"{X_tr.shape[0]:,}", c1)
        metric_card("Test Samples",  f"{X_te.shape[0]:,}", c2)
        metric_card("Post-SMOTE",    f"{X_sm.shape[0]:,}", c3)
        sep()

        # Class weights bar chart
        fig = px.bar(
            x=[LABEL_NAMES[k] for k in sorted(cw.keys())],
            y=[cw[k] for k in sorted(cw.keys())],
            color=[LABEL_NAMES[k] for k in sorted(cw.keys())],
            color_discrete_map=PALETTE,
            labels={"x":"Class","y":"Weight"},
            title="Computed Class Weights"
        )
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

        # Post-SMOTE distribution
        sm_vc = pd.Series(y_sm).value_counts().sort_index()
        fig2 = px.bar(
            x=[LABEL_NAMES[i] for i in sm_vc.index],
            y=sm_vc.values,
            color=[LABEL_NAMES[i] for i in sm_vc.index],
            color_discrete_map=PALETTE,
            title="Post-SMOTE Class Distribution", text=sm_vc.values
        )
        dark_fig(fig2)
        st.plotly_chart(fig2, use_container_width=True)

# ── Page 5 ────────────────────────────────────────────────────
def page_5():
    st.markdown('<div class="main-header">📈 Logistic Regression — Baseline</div>', unsafe_allow_html=True)
    st.info("Logistic Regression serves as a fast linear baseline to benchmark all subsequent models.")

    if st.session_state.X_train_sm is None:
        st.warning("⚠️ Please run Preprocessing first (Page 4).")
        return

    if st.button("🚀 Train Logistic Regression", type="primary"):
        rs  = st.session_state.config.get("RANDOM_STATE", 42)
        cw  = st.session_state.class_weights
        X_tr = st.session_state.X_train_sm
        y_tr = st.session_state.y_train_sm
        X_te = st.session_state.X_test_sc
        y_te = st.session_state.y_test

        model = LogisticRegression(
            max_iter=2000, random_state=rs,
            class_weight=cw, solver="lbfgs"
        )
        model.fit(X_tr, y_tr)
        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)

        st.session_state.lr_model   = model
        st.session_state.y_pred_lr  = y_pred
        st.session_state.y_proba_lr = y_proba

        y_bin = label_binarize(y_te, classes=[0,1,2])
        auc   = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
        f1m   = f1_score(y_te, y_pred, average="macro")
        # Collapse recall
        rpt   = classification_report(y_te, y_pred, target_names=LABEL_NAMES, output_dict=True)
        rec_c = rpt.get("Collapse", {}).get("recall", 0.0)

        st.session_state.results["Logistic Regression"] = dict(
            AUC_macro=auc, F1_macro=f1m, F1_Collapse=rpt.get("Collapse",{}).get("f1-score",0),
            Recall_Collapse=rec_c, model=model
        )
        mark_complete(5)
        st.success("Logistic Regression trained!")

    if st.session_state.lr_model is None:
        return

    y_te   = st.session_state.y_test
    y_pred = st.session_state.y_pred_lr
    y_prob = st.session_state.y_proba_lr
    rpt    = classification_report(y_te, y_pred, target_names=LABEL_NAMES, output_dict=True)

    y_bin = label_binarize(y_te, classes=[0,1,2])
    auc   = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    f1m   = f1_score(y_te, y_pred, average="macro")
    rec_c = rpt.get("Collapse",{}).get("recall",0)

    c1, c2, c3 = st.columns(3)
    metric_card("Macro AUC-ROC",    f"{auc:.3f}", c1)
    metric_card("Macro F1",         f"{f1m:.3f}", c2)
    metric_card("Collapse Recall",  f"{rec_c:.3f}", c3)
    sep()

    # Classification report table
    rpt_df = pd.DataFrame(rpt).T.round(3)
    st.dataframe(rpt_df, use_container_width=True)

    c1, c2 = st.columns([1,2])
    with c1:
        st.plotly_chart(confusion_heatmap(y_te, y_pred, "LR Confusion Matrix"), use_container_width=True)
    with c2:
        fig = roc_curves_fig({"Logistic Regression": y_prob}, y_te)
        fig.update_layout(title="ROC Curves — Logistic Regression")
        st.plotly_chart(fig, use_container_width=True)

# ── Page 6 ────────────────────────────────────────────────────
def page_6():
    st.markdown('<div class="main-header">🌲 Random Forest</div>', unsafe_allow_html=True)

    if st.session_state.X_train_sm is None:
        st.warning("⚠️ Please run Preprocessing first (Page 4).")
        return

    n_est  = st.slider("n_estimators", 50, 500, 300, step=50, key="rf_n")
    max_d  = st.selectbox("max_depth", [None, 5, 10, 15, 20], key="rf_d",
                          format_func=lambda x: "None (full)" if x is None else str(x))

    if st.button("🚀 Train Random Forest", type="primary"):
        rs  = st.session_state.config.get("RANDOM_STATE", 42)
        cw  = st.session_state.class_weights
        X_tr = st.session_state.X_train_sm
        y_tr = st.session_state.y_train_sm
        X_te = st.session_state.X_test_sc
        y_te = st.session_state.y_test

        model = RandomForestClassifier(
            n_estimators=n_est, max_depth=max_d,
            class_weight=cw, random_state=rs, n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)

        st.session_state.rf_model   = model
        st.session_state.y_pred_rf  = y_pred
        st.session_state.y_proba_rf = y_proba

        y_bin = label_binarize(y_te, classes=[0,1,2])
        auc   = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
        f1m   = f1_score(y_te, y_pred, average="macro")
        rpt   = classification_report(y_te, y_pred, target_names=LABEL_NAMES, output_dict=True)
        rec_c = rpt.get("Collapse",{}).get("recall",0)

        st.session_state.results["Random Forest"] = dict(
            AUC_macro=auc, F1_macro=f1m,
            F1_Collapse=rpt.get("Collapse",{}).get("f1-score",0),
            Recall_Collapse=rec_c, model=model
        )
        mark_complete(6)
        st.success("Random Forest trained!")

    if st.session_state.rf_model is None:
        return

    rf     = st.session_state.rf_model
    y_te   = st.session_state.y_test
    y_pred = st.session_state.y_pred_rf
    y_prob = st.session_state.y_proba_rf
    rpt    = classification_report(y_te, y_pred, target_names=LABEL_NAMES, output_dict=True)

    y_bin = label_binarize(y_te, classes=[0,1,2])
    auc   = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    f1m   = f1_score(y_te, y_pred, average="macro")
    rec_c = rpt.get("Collapse",{}).get("recall",0)

    c1, c2, c3 = st.columns(3)
    metric_card("Macro AUC-ROC",   f"{auc:.3f}", c1)
    metric_card("Macro F1",        f"{f1m:.3f}", c2)
    metric_card("Collapse Recall", f"{rec_c:.3f}", c3)
    sep()

    rpt_df = pd.DataFrame(rpt).T.round(3)
    st.dataframe(rpt_df, use_container_width=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(confusion_heatmap(y_te, y_pred, "RF Confusion Matrix"), use_container_width=True)
    with c2:
        # Feature importances
        fi = pd.Series(rf.feature_importances_, index=ENG_FEATURES).sort_values()
        q75 = fi.quantile(0.75)
        colors = ["#1D4ED8" if v >= q75 else "#93C5FD" for v in fi.values]
        fig = go.Figure(go.Bar(
            x=fi.values, y=fi.index, orientation="h",
            marker_color=colors
        ))
        dark_fig(fig)
        fig.update_layout(title="Feature Importances (top quartile in dark blue)", height=400)
        st.plotly_chart(fig, use_container_width=True)

# ── Page 7 ────────────────────────────────────────────────────
def page_7():
    st.markdown('<div class="main-header">🚀 XGBoost / GBM</div>', unsafe_allow_html=True)
    if HAS_XGB:
        st.info("XGBoost is available and will be used.")
    else:
        st.warning("XGBoost not installed — falling back to GradientBoostingClassifier.")

    if st.session_state.X_train_sm is None:
        st.warning("⚠️ Please run Preprocessing first (Page 4).")
        return

    n_est  = st.slider("n_estimators", 50, 300, 100, step=50, key="xgb_n")
    max_d  = st.slider("max_depth", 2, 8, 4, key="xgb_d")
    lr_val = st.slider("learning_rate", 0.01, 0.50, 0.10, step=0.01, key="xgb_lr")

    if st.button("🚀 Train XGBoost/GBM", type="primary"):
        rs   = st.session_state.config.get("RANDOM_STATE", 42)
        cw   = st.session_state.class_weights
        X_tr = st.session_state.X_train_sm
        y_tr = st.session_state.y_train_sm
        X_te = st.session_state.X_test_sc
        y_te = st.session_state.y_test

        if HAS_XGB:
            sw = np.array([cw[yi] for yi in y_tr])
            model = xgb.XGBClassifier(
                n_estimators=n_est, max_depth=max_d,
                learning_rate=lr_val,
                eval_metric="mlogloss", random_state=rs, n_jobs=-1
            )
            model.fit(X_tr, y_tr, sample_weight=sw)
            fi = dict(zip(ENG_FEATURES, model.feature_importances_))
            bname = "XGBoost"
        else:
            model = GradientBoostingClassifier(
                n_estimators=n_est, max_depth=max_d,
                learning_rate=lr_val, random_state=rs
            )
            model.fit(X_tr, y_tr)
            fi = dict(zip(ENG_FEATURES, model.feature_importances_))
            bname = "GBM"

        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)

        st.session_state.boost_model      = model
        st.session_state.boost_name       = bname
        st.session_state.boost_importance = fi
        st.session_state.y_pred_boost     = y_pred
        st.session_state.y_proba_boost    = y_proba

        y_bin = label_binarize(y_te, classes=[0,1,2])
        auc   = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
        f1m   = f1_score(y_te, y_pred, average="macro")
        rpt   = classification_report(y_te, y_pred, target_names=LABEL_NAMES, output_dict=True)
        rec_c = rpt.get("Collapse",{}).get("recall",0)

        st.session_state.results[bname] = dict(
            AUC_macro=auc, F1_macro=f1m,
            F1_Collapse=rpt.get("Collapse",{}).get("f1-score",0),
            Recall_Collapse=rec_c, model=model
        )
        mark_complete(7)
        st.success(f"{bname} trained!")

    if st.session_state.boost_model is None:
        return

    bname  = st.session_state.boost_name
    y_te   = st.session_state.y_test
    y_pred = st.session_state.y_pred_boost
    y_prob = st.session_state.y_proba_boost
    fi     = st.session_state.boost_importance
    rpt    = classification_report(y_te, y_pred, target_names=LABEL_NAMES, output_dict=True)

    y_bin = label_binarize(y_te, classes=[0,1,2])
    auc   = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    f1m   = f1_score(y_te, y_pred, average="macro")
    rec_c = rpt.get("Collapse",{}).get("recall",0)

    c1, c2, c3 = st.columns(3)
    metric_card("Macro AUC-ROC",   f"{auc:.3f}", c1)
    metric_card("Macro F1",        f"{f1m:.3f}", c2)
    metric_card("Collapse Recall", f"{rec_c:.3f}", c3)
    sep()

    rpt_df = pd.DataFrame(rpt).T.round(3)
    st.dataframe(rpt_df, use_container_width=True)

    c1, c2 = st.columns([1,2])
    with c1:
        st.plotly_chart(confusion_heatmap(y_te, y_pred, f"{bname} Confusion Matrix"), use_container_width=True)
    with c2:
        fi_s = pd.Series(fi).sort_values()
        fig  = px.bar(x=fi_s.values, y=fi_s.index, orientation="h",
                      title=f"{bname} Feature Importances",
                      color=fi_s.values, color_continuous_scale="Blues")
        dark_fig(fig)
        fig.update_layout(height=390)
        st.plotly_chart(fig, use_container_width=True)

    # ROC curves for all trained models
    st.markdown("### ROC Curves — All Trained Models")
    roc_models = {}
    if st.session_state.y_proba_lr    is not None: roc_models["Logistic Regression"] = st.session_state.y_proba_lr
    if st.session_state.y_proba_rf    is not None: roc_models["Random Forest"]        = st.session_state.y_proba_rf
    if st.session_state.y_proba_boost is not None: roc_models[bname]                  = st.session_state.y_proba_boost
    if roc_models:
        fig = roc_curves_fig(roc_models, y_te)
        st.plotly_chart(fig, use_container_width=True)

# ── Page 8 ────────────────────────────────────────────────────
def page_8():
    st.markdown('<div class="main-header">🧠 MLP Neural Network</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Layer Perceptron — a fully-connected feedforward neural network. No GPU or TensorFlow required.</div>', unsafe_allow_html=True)

    if st.session_state.X_train_sm is None:
        st.warning("⚠️ Please run Preprocessing first (Page 4).")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        arch_choice = st.selectbox("Hidden Layers",
            ["(64, 32)", "(128, 64, 32)", "(256, 128, 64)", "(128, 64, 32, 16)"],
            index=1, key="mlp_arch")
    with c2:
        activation = st.selectbox("Activation Function", ["relu", "tanh"], key="mlp_act")
    with c3:
        lr_init = st.select_slider("Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001, key="mlp_lr")

    hidden_layers = tuple(int(x.strip()) for x in arch_choice.strip("()").split(","))

    # Architecture table
    arch_rows = [{"Layer": "Input", "Size": f"{len(ENG_FEATURES)} features", "Activation": "—"}]
    for i, n in enumerate(hidden_layers):
        arch_rows.append({"Layer": f"Hidden {i+1}", "Size": f"{n} neurons", "Activation": activation})
    arch_rows.append({"Layer": "Output", "Size": "3 classes", "Activation": "softmax"})
    st.dataframe(pd.DataFrame(arch_rows), use_container_width=True, hide_index=True)

    if st.button("🧠 Train MLP Neural Network", type="primary"):
        rs   = st.session_state.config.get("RANDOM_STATE", 42)
        X_tr = st.session_state.X_train_sm
        y_tr = st.session_state.y_train_sm
        X_te = st.session_state.X_test_sc
        y_te = st.session_state.y_test

        with st.spinner("Training MLP Neural Network (early stopping enabled)…"):
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation=activation,
                solver="adam",
                learning_rate_init=lr_init,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=rs
            )
            model.fit(X_tr, y_tr)

        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)

        st.session_state.lstm_model   = model
        st.session_state.lstm_history = {"loss": model.loss_curve_}
        st.session_state.y_pred_lstm  = y_pred
        st.session_state.y_proba_lstm = y_proba
        st.session_state.y_seq_te     = y_te

        y_bin = label_binarize(y_te, classes=[0,1,2])
        auc   = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
        f1m   = f1_score(y_te, y_pred, average="macro")
        rpt   = classification_report(y_te, y_pred, target_names=LABEL_NAMES, output_dict=True)
        rec_c = rpt.get("Collapse",{}).get("recall",0)
        st.session_state.results["MLP Neural Net"] = dict(
            AUC_macro=auc, F1_macro=f1m,
            F1_Collapse=rpt.get("Collapse",{}).get("f1-score",0),
            Recall_Collapse=rec_c, model=model
        )
        mark_complete(8)
        st.success(f"MLP trained — converged in {len(model.loss_curve_)} iterations!")

    if st.session_state.lstm_history is None:
        return

    hist   = st.session_state.lstm_history
    y_seq  = st.session_state.y_seq_te
    y_pred = st.session_state.y_pred_lstm
    y_prob = st.session_state.y_proba_lstm

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=hist["loss"], mode="lines",
        name="Training Loss", line=dict(color="#2563EB", width=2)
    ))
    dark_fig(fig)
    fig.update_layout(title="MLP Training Loss per Iteration",
                      xaxis_title="Iteration", yaxis_title="Loss", height=300)
    st.plotly_chart(fig, use_container_width=True)

    rpt   = classification_report(y_seq, y_pred, target_names=LABEL_NAMES, output_dict=True)
    y_bin = label_binarize(y_seq, classes=[0,1,2])
    auc   = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    f1m   = f1_score(y_seq, y_pred, average="macro")
    rec_c = rpt.get("Collapse",{}).get("recall",0)

    c1, c2, c3 = st.columns(3)
    metric_card("Macro AUC-ROC",   f"{auc:.3f}", c1)
    metric_card("Macro F1",        f"{f1m:.3f}", c2)
    metric_card("Collapse Recall", f"{rec_c:.3f}", c3)
    sep()
    rpt_df = pd.DataFrame(rpt).T.round(3)
    st.dataframe(rpt_df, use_container_width=True)
    sep()
    st.plotly_chart(confusion_heatmap(y_seq, y_pred, "MLP Confusion Matrix"), use_container_width=True)

# ── Page 9 ────────────────────────────────────────────────────
def page_9():
    st.markdown('<div class="main-header">🏆 Model Comparison</div>', unsafe_allow_html=True)

    res = st.session_state.results
    if not res:
        st.warning("⚠️ Train at least one model first.")
        return

    rows = []
    for name, v in res.items():
        rows.append({
            "Model":           name,
            "AUC (macro)":     round(v.get("AUC_macro",0),4),
            "F1 (macro)":      round(v.get("F1_macro",0),4),
            "F1 (Collapse)":   round(v.get("F1_Collapse",0),4),
            "Recall(Collapse)":round(v.get("Recall_Collapse",0),4),
        })
    comp_df = pd.DataFrame(rows).set_index("Model")

    # Styled dataframe
    def style_df(df):
        styled = df.style
        for col in df.columns:
            styled = styled.highlight_max(subset=[col], color="#D1FAE5")
            styled = styled.highlight_min(subset=[col], color="#FEE2E2")
        return styled

    st.dataframe(style_df(comp_df), use_container_width=True)
    sep()

    # Grouped bar chart
    metrics = ["F1 (macro)","AUC (macro)","F1 (Collapse)","Recall(Collapse)"]
    fig = go.Figure()
    colors_bar = ["#2563EB","#059669","#7C3AED","#DC2626"]
    for mi, met in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=met, x=comp_df.index.tolist(),
            y=comp_df[met].tolist(),
            marker_color=colors_bar[mi]
        ))
    fig.update_layout(barmode="group", title="Model Comparison — Key Metrics")
    dark_fig(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Collapse Recall only
    st.markdown("### Collapse Recall (Critical Safety Metric)")
    rec_vals  = comp_df["Recall(Collapse)"].tolist()
    rec_names = comp_df.index.tolist()
    bar_colors = ["#059669" if v >= 0.85 else "#DC2626" for v in rec_vals]
    fig2 = go.Figure(go.Bar(
        x=rec_vals, y=rec_names, orientation="h",
        marker_color=bar_colors, text=[f"{v:.3f}" for v in rec_vals],
        textposition="auto"
    ))
    dark_fig(fig2)
    fig2.add_vline(x=0.85, line_dash="dash", line_color="#2563EB",
                   annotation_text="Target 0.85")
    fig2.update_layout(title="Collapse Recall by Model", height=300)
    st.plotly_chart(fig2, use_container_width=True)

    # Best model
    best_name = comp_df["Recall(Collapse)"].idxmax()
    st.markdown(
        f'<div style="background:#EFF6FF;border:2px solid #2563EB;border-radius:10px;'
        f'padding:18px;text-align:center;font-size:1.3rem;color:#1E3A5F;">'
        f'★ Recommended Production Model: <strong>{best_name}</strong></div>',
        unsafe_allow_html=True
    )
    mark_complete(9)

# ── Page 10 ────────────────────────────────────────────────────
def page_10():
    st.markdown('<div class="main-header">🌱 Seed Dataset Validation</div>', unsafe_allow_html=True)
    st.info("The 123-row seed dataset is an independent holdout used to verify model generalisation on unseen data distinct from the augmented training pool.")

    if st.session_state.df_seed_eng is None:
        st.warning("⚠️ Please run Feature Engineering first (Page 3).")
        return
    if st.session_state.scaler is None:
        st.warning("⚠️ Please run Preprocessing first (Page 4).")
        return

    seed_eng = st.session_state.df_seed_eng
    scaler   = st.session_state.scaler
    X_seed   = scaler.transform(seed_eng[ENG_FEATURES].values)
    y_seed   = seed_eng["Stability_Label"].values

    metric_card("Seed Records", f"{len(y_seed)}", st.columns(3)[0])
    sep()

    model_map = {}
    if st.session_state.rf_model    is not None: model_map["Random Forest"] = st.session_state.rf_model
    if st.session_state.boost_model is not None: model_map[st.session_state.boost_name] = st.session_state.boost_model

    if not model_map:
        st.warning("⚠️ Train RF and/or XGBoost/GBM first.")
        return

    for mname, model in model_map.items():
        st.markdown(f"### {mname}")
        y_pred = model.predict(X_seed)
        rpt    = classification_report(y_seed, y_pred, target_names=LABEL_NAMES, output_dict=True)
        c1, c2 = st.columns([2,1])
        with c1:
            st.dataframe(pd.DataFrame(rpt).T.round(3), use_container_width=True)
        with c2:
            st.plotly_chart(confusion_heatmap(y_seed, y_pred, f"{mname} — Seed"), use_container_width=True)
        sep()

    mark_complete(10)

# ── Page 11 ────────────────────────────────────────────────────
def page_11():
    st.markdown('<div class="main-header">🚨 Alert System</div>', unsafe_allow_html=True)
    st.markdown("Simulate a real-time alert replay using row-by-row inference on a demo scenario.")

    if st.session_state.scaler is None or st.session_state.rf_model is None:
        st.warning("⚠️ Train at least Random Forest first (Pages 4–6).")
        return

    p_warn  = st.slider("P(Unstable) WARNING threshold", 0.10, 0.60, 0.30, step=0.05, key="al_warn")
    p_crit  = st.slider("P(Collapse) CRITICAL threshold", 0.05, 0.50, 0.15, step=0.05, key="al_crit")

    model_choices = {}
    if st.session_state.lr_model    is not None: model_choices["Logistic Regression"] = (st.session_state.lr_model,    st.session_state.y_proba_lr)
    if st.session_state.rf_model    is not None: model_choices["Random Forest"]        = (st.session_state.rf_model,    None)
    if st.session_state.boost_model is not None: model_choices[st.session_state.boost_name] = (st.session_state.boost_model, None)

    sel_model = st.selectbox("Alert model", list(model_choices.keys()), key="al_model")

    scenarios = {
        "Line Fault → Collapse": dict(
            freq=np.concatenate([np.linspace(50.0, 48.0, 60), np.linspace(48.0, 46.5, 40)]),
            volt=np.concatenate([np.linspace(1.00, 0.85, 60), np.linspace(0.85, 0.72, 40)]),
            roc =np.concatenate([np.linspace(0.0, -0.8, 60), np.linspace(-0.8, -1.5, 40)]),
        ),
        "Voltage Instability": dict(
            freq=np.linspace(50.0, 49.2, 100),
            volt=np.concatenate([np.linspace(1.0, 0.88, 70), np.linspace(0.88, 0.75, 30)]),
            roc =np.random.uniform(-0.3, 0.1, 100),
        ),
        "Generation Loss": dict(
            freq=np.concatenate([np.linspace(50.0, 48.5, 50), np.linspace(48.5, 47.0, 50)]),
            volt=np.linspace(1.0, 0.90, 100),
            roc =np.concatenate([np.linspace(0.0, -1.0, 50), np.linspace(-1.0, -1.8, 50)]),
        ),
    }
    sel_sc = st.selectbox("Demo scenario", list(scenarios.keys()), key="al_sc")

    if st.button("▶ Run Alert Replay", type="primary"):
        sc      = scenarios[sel_sc]
        n       = len(sc["freq"])
        model_o = model_choices[sel_model][0]
        scaler  = st.session_state.scaler

        rows = []
        for i in range(n):
            f   = sc["freq"][i] + np.random.normal(0, 0.05)
            v   = sc["volt"][i] + np.random.normal(0, 0.005)
            roc = sc["roc"][i]  + np.random.normal(0, 0.02)
            row = {
                "Frequency_Hz": f, "Voltage_pu": v, "Angle_deg": np.random.uniform(-30,30),
                "Active_Power_MW": np.random.uniform(800,1200),
                "Reactive_Power_MVAr": np.random.uniform(100,400),
                "ROCOF_Hz_per_s": roc,
                "Ambient_Temp_C": np.random.uniform(28,42),
                "Stability_Label": 0
            }
            rows.append(row)

        replay_df = pd.DataFrame(rows)
        replay_df = engineer_features(replay_df)
        X_rep     = scaler.transform(replay_df[ENG_FEATURES].values)
        proba     = model_o.predict_proba(X_rep)

        alerts = []
        for i in range(n):
            p_u = proba[i, 1]
            p_c = proba[i, 2]
            if p_c >= p_crit:
                level = "CRITICAL"
            elif p_u >= p_warn:
                level = "WARNING"
            else:
                level = "NORMAL"
            alerts.append({
                "t": i, "Freq": sc["freq"][i], "Volt": sc["volt"][i],
                "P_Unstable": p_u, "P_Collapse": p_c, "Alert": level
            })

        alerts_df = pd.DataFrame(alerts)
        st.session_state.alerts_df = alerts_df
        mark_complete(11)
        st.success("Alert replay complete!")

    alerts_df = st.session_state.alerts_df
    if alerts_df is None:
        return

    n_warn = (alerts_df["Alert"] == "WARNING").sum()
    n_crit = (alerts_df["Alert"] == "CRITICAL").sum()
    crit_t = alerts_df[alerts_df["Alert"]=="CRITICAL"]["t"].min()

    c1, c2, c3, c4 = st.columns(4)
    metric_card("Total Readings",   f"{len(alerts_df)}",   c1)
    metric_card("WARNINGs",         f"{n_warn}",            c2)
    metric_card("CRITICALs",        f"{n_crit}",            c3)
    metric_card("1st CRITICAL @t",  f"{crit_t if n_crit>0 else 'N/A'}", c4)
    sep()

    # 3-subplot chart
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=["Frequency (Hz)","Voltage (pu)","Alert Probabilities"],
                        row_heights=[0.33,0.33,0.34])
    fig.add_trace(go.Scatter(x=alerts_df["t"], y=alerts_df["Freq"], name="Frequency",
                             line=dict(color="#2563EB", width=2)), row=1, col=1)
    fig.add_hline(y=49.5, row=1, col=1, line_dash="dot", line_color="#D97706")
    fig.add_hline(y=48.5, row=1, col=1, line_dash="dash", line_color="#DC2626")

    fig.add_trace(go.Scatter(x=alerts_df["t"], y=alerts_df["Volt"], name="Voltage",
                             line=dict(color="#9b59b6")), row=2, col=1)
    fig.add_hline(y=0.97, row=2, col=1, line_dash="dot", line_color="#D97706")
    fig.add_hline(y=0.80, row=2, col=1, line_dash="dash", line_color="#DC2626")

    fig.add_trace(go.Scatter(x=alerts_df["t"], y=alerts_df["P_Collapse"],
                             name="P(Collapse)", line=dict(color="#e74c3c")), row=3, col=1)
    fig.add_trace(go.Scatter(x=alerts_df["t"], y=alerts_df["P_Unstable"],
                             name="P(Unstable)", line=dict(color="#D97706", width=1.5)), row=3, col=1)
    fig.add_hline(y=p_crit, row=3, col=1, line_dash="dash", line_color="#DC2626",
                  annotation_text="CRITICAL thr")
    fig.add_hline(y=p_warn, row=3, col=1, line_dash="dash", line_color="#D97706",
                  annotation_text="WARNING thr")

    # Background shading
    for _, row_a in alerts_df.iterrows():
        t = row_a["t"]
        color = "rgba(231,76,60,0.15)" if row_a["Alert"]=="CRITICAL" \
                else ("rgba(243,156,18,0.15)" if row_a["Alert"]=="WARNING" else None)
        if color:
            for rn in [1,2,3]:
                fig.add_vrect(x0=t-0.5, x1=t+0.5, fillcolor=color, opacity=1.0,
                              layer="below", line_width=0, row=rn, col=1)

    dark_fig(fig)
    fig.update_layout(height=650, title=f"Alert Replay — {sel_sc}")
    st.plotly_chart(fig, use_container_width=True)

    # Alert log table
    st.markdown("### Alert Log")
    def row_color(alert):
        if alert == "CRITICAL": return "background-color: rgba(231,76,60,0.35)"
        if alert == "WARNING":  return "background-color: rgba(243,156,18,0.25)"
        return ""

    show_df = alerts_df.copy()
    show_df = show_df.round(4)
    styled  = show_df.style.map(
        lambda v: "background-color: rgba(231,76,60,0.35)" if v=="CRITICAL"
                  else ("background-color: rgba(243,156,18,0.25)" if v=="WARNING" else ""),
        subset=["Alert"]
    )
    st.dataframe(styled, use_container_width=True)

# ── Page 12 ────────────────────────────────────────────────────
def page_12():
    st.markdown('<div class="main-header">🔁 Cross-Validation</div>', unsafe_allow_html=True)

    if st.session_state.df_eng is None or st.session_state.rf_model is None:
        st.warning("⚠️ Please train at least Random Forest first (Pages 3–6).")
        return

    n_folds = st.slider("Number of folds", 3, 10, 5, key="cv_folds")

    if st.button("▶ Run Cross-Validation", type="primary"):
        eng  = st.session_state.df_eng
        rs   = st.session_state.config.get("RANDOM_STATE", 42)
        cw   = st.session_state.class_weights
        X    = eng[ENG_FEATURES].values
        y    = eng["Stability_Label"].values

        model = RandomForestClassifier(
            n_estimators=200, class_weight=cw, random_state=rs, n_jobs=-1
        )
        skf   = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rs)
        cv    = cross_validate(
            model, X, y, cv=skf,
            scoring={"f1_macro":"f1_macro","accuracy":"accuracy"},
            return_train_score=True
        )
        cv_df = pd.DataFrame({
            "Fold":     list(range(1, n_folds+1)),
            "Train F1": cv["train_f1_macro"].round(4),
            "Val F1":   cv["test_f1_macro"].round(4),
            "Val Acc":  cv["test_accuracy"].round(4),
        })
        st.session_state.cv_results = cv_df
        mark_complete(12)
        st.success("Cross-validation complete!")

    cv_df = st.session_state.cv_results
    if cv_df is None:
        return

    mean_f1 = cv_df["Val F1"].mean()
    std_f1  = cv_df["Val F1"].std()

    st.dataframe(cv_df, use_container_width=True)
    metric_card("Val F1 Mean ± Std", f"{mean_f1:.3f} ± {std_f1:.3f}", st.columns(3)[0])
    sep()

    # Line chart with confidence band
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cv_df["Fold"], y=cv_df["Train F1"],
        name="Train F1", mode="lines+markers", line=dict(color="#94A3B8", dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=cv_df["Fold"], y=cv_df["Val F1"],
        name="Val F1", mode="lines+markers", line=dict(color="#2563EB", width=2),
        error_y=dict(type="constant", value=std_f1, visible=True, color="#2563EB")
    ))
    dark_fig(fig)
    fig.update_layout(
        title="Train vs Validation F1 per Fold",
        xaxis_title="Fold", yaxis_title="F1 Score",
        height=380
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Page 13 ────────────────────────────────────────────────────
def page_13():
    st.markdown('<div class="main-header">📋 Summary & Export</div>', unsafe_allow_html=True)
    mark_complete(13)

    done = st.session_state.completed
    # Completion grid
    st.markdown("### Pipeline Completion Status")
    cols = st.columns(7)
    for i, title in enumerate(PAGE_TITLES):
        col = cols[i % 7]
        dot   = "🟢" if i in done else "⚪"
        short = title[:16] + "…" if len(title) > 16 else title
        border_color = "#059669" if i in done else "#E2E8F0"
        col.markdown(
            f'<div style="background:#FFFFFF;border-radius:8px;padding:10px;'
            f'text-align:center;border:2px solid {border_color};font-size:0.78rem;'
            f'box-shadow:0 1px 4px rgba(0,0,0,0.06);">'
            f'{dot}<br><b>{i}</b><br>{short}</div>',
            unsafe_allow_html=True
        )
    sep()

    # Results table
    res = st.session_state.results
    if res:
        st.markdown("### Trained Model Results")
        rows = []
        for name, v in res.items():
            rows.append({
                "Model":           name,
                "AUC (macro)":     round(v.get("AUC_macro",0),4),
                "F1 (macro)":      round(v.get("F1_macro",0),4),
                "F1 (Collapse)":   round(v.get("F1_Collapse",0),4),
                "Recall(Collapse)":round(v.get("Recall_Collapse",0),4),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        sep()

    # Config table
    st.markdown("### Configuration Used")
    st.dataframe(
        pd.DataFrame([st.session_state.config]).T.rename(columns={0:"Value"}),
        use_container_width=True
    )
    sep()

    # Export
    st.markdown("### Save Trained Models")
    save_dir = st.text_input("Save directory", value=str(Path.home() / "grid_models"), key="save_dir")
    if st.button("💾 Save All Trained Models", type="primary"):
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved = []
        model_map = {
            "lr_model":    st.session_state.lr_model,
            "rf_model":    st.session_state.rf_model,
            "boost_model": st.session_state.boost_model,
        }
        for fname, model in model_map.items():
            if model is not None:
                p = out / f"{fname}.pkl"
                with open(p, "wb") as f:
                    pickle.dump(model, f)
                saved.append(str(p))
        if st.session_state.scaler is not None:
            p = out / "scaler.pkl"
            with open(p, "wb") as f:
                pickle.dump(st.session_state.scaler, f)
            saved.append(str(p))
        if saved:
            st.success(f"Saved: {', '.join(saved)}")
        else:
            st.warning("No models to save yet.")

    # Production code snippet
    st.markdown("### Production Loading Code")
    st.code("""
import pickle, numpy as np

# Load artefacts
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

ENG_FEATURES = [
    "Frequency_Hz","Voltage_pu","Angle_deg","Active_Power_MW",
    "Reactive_Power_MVAr","ROCOF_Hz_per_s","Ambient_Temp_C",
    "Freq_Dev","V_Dev","Freq_sq","V_sq","Apparent_S",
    "PQ_Ratio","Power_Factor","FxV","ROCOF_abs","ROCOF_sq"
]
LABEL_NAMES = ["Stable","Unstable","Collapse"]

def predict_live(row_dict: dict) -> str:
    import pandas as pd
    df = pd.DataFrame([row_dict])
    eps = 1e-6
    df["Freq_Dev"]     = (df["Frequency_Hz"] - 50.0).abs()
    df["V_Dev"]        = (df["Voltage_pu"]   - 1.0 ).abs()
    df["Freq_sq"]      = df["Freq_Dev"] ** 2
    df["V_sq"]         = df["V_Dev"]   ** 2
    df["Apparent_S"]   = (df["Active_Power_MW"]**2 + df["Reactive_Power_MVAr"]**2)**0.5
    df["PQ_Ratio"]     = df["Active_Power_MW"] / (df["Reactive_Power_MVAr"].abs() + eps)
    df["Power_Factor"] = df["Active_Power_MW"] / (df["Apparent_S"] + eps)
    df["FxV"]          = df["Frequency_Hz"] * df["Voltage_pu"]
    df["ROCOF_abs"]    = df["ROCOF_Hz_per_s"].abs()
    df["ROCOF_sq"]     = df["ROCOF_Hz_per_s"] ** 2
    X = scaler.transform(df[ENG_FEATURES].values)
    return LABEL_NAMES[model.predict(X)[0]]
""", language="python")

# ── MAIN DISPATCH ──────────────────────────────────────────────
def main():
    render_sidebar()

    cur = st.session_state.page
    PAGE_FNS = [
        page_0, page_1, page_2, page_3, page_4, page_5, page_6,
        page_7, page_8, page_9, page_10, page_11, page_12, page_13
    ]
    PAGE_FNS[cur]()
    render_nav()

if __name__ == "__main__":
    main()
