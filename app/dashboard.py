"""AeroSweep — Streamlit Dashboard for Fuzzy + GA Optimization.

Launch with:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
import copy
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yaml

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aerosweep.fuzzy_ga.fuzzy_logic import (
    DEFAULT_CONFIG,
    evaluate_fuzzy,
    fuzzify,
    membership_degree,
)
from aerosweep.fuzzy_ga.ga_optimizer import optimize_fuzzy, run_fuzzy_with_config

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AeroSweep - Fuzzy + GA Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f23 0%, #1a1a3e 100%);
}
[data-testid="stSidebar"] * {
    color: #e0e0ff !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e1e3f 0%, #2a2a5f 100%);
    border: 1px solid rgba(100, 100, 255, 0.2);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}
[data-testid="stMetric"] label {
    color: #8888cc !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background-color: rgba(15, 15, 35, 0.5);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4a4aff 0%, #7a4aff 100%) !important;
}

/* Header gradient text */
.gradient-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.sub-header {
    color: #8888bb;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

/* Risk level badges */
.risk-low      { background: #1b5e20; color: #a5d6a7; padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
.risk-medium   { background: #e65100; color: #ffcc80; padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
.risk-high     { background: #b71c1c; color: #ef9a9a; padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
.risk-critical { background: #4a148c; color: #ce93d8; padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }

/* Divider */
.styled-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #667eea, transparent);
    border: none;
    margin: 1.5rem 0;
}

/* Card container */
.info-card {
    background: linear-gradient(135deg, #1a1a3f 0%, #252560 100%);
    border: 1px solid rgba(100, 100, 255, 0.15);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ─────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "df_raw": None,
        "df_fuzzy_default": None,
        "df_fuzzy_ga": None,
        "ga_result": None,
        "config_default": None,
        "config_ga": None,
        "csv_filename": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Helper functions ─────────────────────────────────────────────────────────

def _load_config(path: Path | None = None) -> Dict[str, Any]:
    """Load fuzzy config from YAML or use defaults."""
    if path and path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return copy.deepcopy(DEFAULT_CONFIG)


def _get_fuzzy_config(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only the fuzzy-relevant keys from a full YAML config."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for key in ("variables", "category_weights", "risk_output"):
        if key in raw_config:
            cfg[key] = raw_config[key]
    return cfg


RISK_COLORS: Dict[str, str] = {
    "low": "#4caf50",
    "medium": "#ff9800",
    "high": "#f44336",
    "critical": "#9c27b0",
}

RISK_ORDER = ["low", "medium", "high", "critical"]


def _risk_distribution_chart(df: pd.DataFrame, col: str = "risk_level", title: str = "") -> go.Figure:
    """Bar chart of risk level distribution."""
    if col not in df.columns:
        return go.Figure()
    counts = df[col].value_counts().reindex(RISK_ORDER, fill_value=0)
    colors = [RISK_COLORS.get(r, "#999") for r in counts.index]
    fig = go.Figure(go.Bar(
        x=counts.index,
        y=counts.values,
        marker_color=colors,
        text=counts.values,
        textposition="outside",
        textfont=dict(size=14, color="white"),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#ccc")),
        xaxis_title="Risk Level",
        yaxis_title="Count",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
        margin=dict(t=50, b=40, l=40, r=20),
        height=350,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig


def _plot_membership_functions(
    config: Dict[str, Any],
    var_name: str,
    title: str = "",
    overlay_config: Dict[str, Any] | None = None,
) -> go.Figure:
    """Plot membership functions for a given variable."""
    terms = config["variables"][var_name]

    # Determine range
    all_pts = [p for pts in terms.values() for p in pts]
    x_min, x_max = min(all_pts), max(all_pts)
    x = np.linspace(x_min, x_max, 500)

    fig = go.Figure()
    colors_main = ["#667eea", "#43e97b", "#f7971e", "#fc5c7d", "#a18cd1"]
    colors_overlay = [
        "rgba(102,126,234,0.5)", "rgba(67,233,123,0.5)", "rgba(247,151,30,0.5)",
        "rgba(252,92,125,0.5)", "rgba(161,140,209,0.5)",
    ]

    for i, (term_name, points) in enumerate(terms.items()):
        y = [membership_degree(xi, points) for xi in x]
        color = colors_main[i % len(colors_main)]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=f"{term_name}",
            line=dict(color=color, width=3),
        ))

    if overlay_config is not None:
        overlay_terms = overlay_config["variables"][var_name]
        for i, (term_name, points) in enumerate(overlay_terms.items()):
            y = [membership_degree(xi, points) for xi in x]
            color = colors_overlay[i % len(colors_overlay)]
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines", name=f"{term_name} (GA)",
                line=dict(color=color, width=2, dash="dash"),
            ))

    fig.update_layout(
        title=dict(text=title or var_name, font=dict(size=15, color="#ccc")),
        xaxis_title=var_name,
        yaxis_title="Membership Degree",
        yaxis_range=[-0.05, 1.1],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
        legend=dict(font=dict(size=11)),
        margin=dict(t=50, b=40, l=40, r=20),
        height=320,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)")
    return fig


def _fitness_curve_chart(history: List, title: str = "Fitness Curve") -> go.Figure:
    """Line chart of best & average fitness over generations."""
    gens = list(range(1, len(history) + 1))
    best = [h[0] for h in history]
    avg = [h[1] for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gens, y=best, mode="lines+markers", name="Best Fitness",
        line=dict(color="#667eea", width=3),
        marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=gens, y=avg, mode="lines", name="Avg Fitness",
        line=dict(color="#fc5c7d", width=2, dash="dot"),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#ccc")),
        xaxis_title="Generation",
        yaxis_title="Fitness (lower = better)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
        margin=dict(t=50, b=40, l=40, r=20),
        height=400,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)")
    return fig


def _config_comparison_table(default_cfg: Dict[str, Any], ga_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Create a comparison table of default vs GA-optimised parameters."""
    rows = []
    for var_name in ("area_density_pct", "jumlah_instance", "confidence_score"):
        for term_name in default_cfg["variables"][var_name]:
            default_pts = default_cfg["variables"][var_name][term_name]
            ga_pts = ga_cfg["variables"][var_name][term_name]
            rows.append({
                "Variable": var_name,
                "Term": term_name,
                "Default": str([round(p, 4) for p in default_pts]),
                "GA Optimized": str([round(p, 4) for p in ga_pts]),
            })
    # Risk centers
    for cname in ("low", "medium", "high", "critical"):
        rows.append({
            "Variable": "risk_center",
            "Term": cname,
            "Default": str(round(default_cfg["risk_output"]["centers"][cname], 2)),
            "GA Optimized": str(round(ga_cfg["risk_output"]["centers"][cname], 2)),
        })
    return pd.DataFrame(rows)


# ── Sidebar ──────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## AeroSweep")
        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Data Source")

        upload_mode = st.radio(
            "Sumber data", ["Upload CSV", "File default"],
            label_visibility="collapsed",
        )

        if upload_mode == "Upload CSV":
            uploaded = st.file_uploader("Upload ANN Feature CSV", type=["csv"])
            if uploaded is not None:
                df = pd.read_csv(uploaded)
                st.session_state.df_raw = df
                st.session_state.csv_filename = uploaded.name
                # Reset results when new data is loaded
                st.session_state.df_fuzzy_default = None
                st.session_state.df_fuzzy_ga = None
                st.session_state.ga_result = None
        else:
            default_csv = ROOT_DIR / "outputs" / "output_fitur_ann_ke_fuzzy.csv"
            if default_csv.exists():
                if st.button(" Load Default CSV"):
                    df = pd.read_csv(default_csv)
                    st.session_state.df_raw = df
                    st.session_state.csv_filename = default_csv.name
                    st.session_state.df_fuzzy_default = None
                    st.session_state.df_fuzzy_ga = None
                    st.session_state.ga_result = None
            else:
                st.warning(f"File tidak ditemukan:\n`{default_csv}`\n\nJalankan:\n```\npython scripts/generate_sample_data.py\n```")

        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        if st.session_state.df_raw is not None:
            df = st.session_state.df_raw
            st.markdown("###  Data Summary")
            st.markdown(f"**File:** `{st.session_state.csv_filename}`")
            st.metric("Total Rows", len(df))
            if "has_detection" in df.columns:
                st.metric("Active Grids", int((df["has_detection"] == 1).sum()))
            if "risk_gt" in df.columns:
                st.success(" Ground truth tersedia")
            else:
                st.info("ℹ️ Tanpa ground truth")


_render_sidebar()


# ── Main content ─────────────────────────────────────────────────────────────

st.markdown('<p class="gradient-header">AeroSweep Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Fuzzy Logic Risk Assessment + Genetic Algorithm Optimization</p>', unsafe_allow_html=True)

if st.session_state.df_raw is None:
    st.markdown("""
    <div class="info-card">
        <h3 style="color: #667eea; margin-top: 0;"> Selamat Datang!</h3>
        <p style="color: #aaa; font-size: 1.1rem;">
            Upload file CSV hasil ANN atau load file default dari sidebar untuk memulai analisis.
        </p>
        <p style="color: #888; font-size: 0.9rem;">
            Jika belum punya data, jalankan: <code>python scripts/generate_sample_data.py</code>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_dashboard, tab_fuzzy, tab_ga, tab_compare, tab_tester = st.tabs([
    " Dashboard",
    " Fuzzy Inference",
    " GA Optimization",
    " Comparison",
    " Grid Tester",
])

df_raw = st.session_state.df_raw

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    st.markdown("###  Dataset Overview")

    df_active = df_raw[df_raw["has_detection"] == 1] if "has_detection" in df_raw.columns else df_raw

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Grids", len(df_raw))
    with c2:
        st.metric("Active Grids", len(df_active))
    with c3:
        if "area_density_pct" in df_active.columns:
            st.metric("Avg Density", f"{df_active['area_density_pct'].mean():.2f}%")
    with c4:
        if "confidence_score" in df_active.columns:
            st.metric("Avg Confidence", f"{df_active['confidence_score'].mean():.3f}")

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Distribusi Kategori Material")
        if "kategori_dominan" in df_active.columns:
            cat_counts = df_active["kategori_dominan"].value_counts().head(15)
            fig_cat = go.Figure(go.Bar(
                x=cat_counts.values,
                y=cat_counts.index,
                orientation="h",
                marker=dict(
                    color=cat_counts.values,
                    colorscale="Viridis",
                ),
                text=cat_counts.values,
                textposition="outside",
            ))
            fig_cat.update_layout(
                height=400,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"),
                margin=dict(t=20, b=20, l=150, r=40),
                yaxis=dict(autorange="reversed"),
            )
            fig_cat.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(fig_cat, use_container_width=True)

    with col_right:
        st.markdown("#### Distribusi Fitur Numerik")
        if "area_density_pct" in df_active.columns:
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=df_active["area_density_pct"], name="Area Density %",
                marker_color="#667eea", opacity=0.7, nbinsx=30,
            ))
            fig_dist.update_layout(
                height=400,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"),
                margin=dict(t=20, b=40, l=40, r=20),
                barmode="overlay",
            )
            fig_dist.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
            fig_dist.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(fig_dist, use_container_width=True)

    # Ground truth distribution if available
    if "risk_gt" in df_active.columns:
        st.markdown("#### Ground Truth Risk Distribution")
        fig_gt = _risk_distribution_chart(df_active, "risk_gt", "Ground Truth Labels")
        st.plotly_chart(fig_gt, use_container_width=True)

    st.markdown("####  Data Preview")
    st.dataframe(df_raw.head(20), use_container_width=True, height=400)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: FUZZY INFERENCE (default params)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_fuzzy:
    st.markdown("###  Fuzzy Inference (Default Parameters)")
    st.markdown("Menjalankan inferensi fuzzy menggunakan parameter membership function **default** (tanpa optimasi GA).")

    config_path = ROOT_DIR / "configs" / "fuzzy.yaml"
    full_config = _load_config(config_path)
    fuzzy_config = _get_fuzzy_config(full_config)
    st.session_state.config_default = fuzzy_config

    # Membership function visualisation
    st.markdown("#### Membership Functions")
    mf_col1, mf_col2, mf_col3 = st.columns(3)
    with mf_col1:
        fig = _plot_membership_functions(fuzzy_config, "area_density_pct", "Area Density (%)")
        st.plotly_chart(fig, use_container_width=True)
    with mf_col2:
        fig = _plot_membership_functions(fuzzy_config, "jumlah_instance", "Jumlah Instance")
        st.plotly_chart(fig, use_container_width=True)
    with mf_col3:
        fig = _plot_membership_functions(fuzzy_config, "confidence_score", "Confidence Score")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    if st.button("▶️ Run Fuzzy Inference (Default)", type="primary", key="btn_fuzzy_default"):
        with st.spinner("Running fuzzy inference..."):
            df_result = run_fuzzy_with_config(df_raw, fuzzy_config)
            st.session_state.df_fuzzy_default = df_result
        st.success(f" Selesai! {len(df_result)} rows diproses.")

    if st.session_state.df_fuzzy_default is not None:
        df_fuzz = st.session_state.df_fuzzy_default

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Avg Fuzzy Score", f"{df_fuzz['fuzzy_score'].mean():.2f}")
        with c2:
            st.metric("Min Score", f"{df_fuzz['fuzzy_score'].min():.2f}")
        with c3:
            st.metric("Max Score", f"{df_fuzz['fuzzy_score'].max():.2f}")
        with c4:
            st.metric("Std Dev", f"{df_fuzz['fuzzy_score'].std():.2f}")

        fig_risk = _risk_distribution_chart(df_fuzz, "risk_level", "Risk Level Distribution (Default)")
        st.plotly_chart(fig_risk, use_container_width=True)

        st.markdown("####  Hasil Fuzzy Inference")
        display_cols = [c for c in [
            "grid_id", "area_density_pct", "jumlah_instance",
            "confidence_score", "kategori_dominan",
            "fuzzy_score", "risk_level", "category_weight",
        ] if c in df_fuzz.columns]
        st.dataframe(df_fuzz[display_cols], use_container_width=True, height=400)

        # Download
        csv_out = df_fuzz.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV Hasil Fuzzy", csv_out, "fuzzy_results_default.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: GA OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ga:
    st.markdown("###  Genetic Algorithm Optimization")
    st.markdown("Optimasi parameter membership function fuzzy menggunakan Genetic Algorithm.")

    st.markdown("#### ⚙️ GA Hyperparameters")
    ga_col1, ga_col2, ga_col3 = st.columns(3)

    with ga_col1:
        pop_size = st.slider("Population Size", 10, 200, 50, 10, key="ga_pop")
        generations = st.slider("Generations", 5, 200, 40, 5, key="ga_gens")
        seed_val = st.number_input("Random Seed", 0, 9999, 42, key="ga_seed")

    with ga_col2:
        mutation_rate = st.slider("Mutation Rate", 0.01, 0.50, 0.10, 0.01, key="ga_mut")
        crossover_rate = st.slider("Crossover Rate", 0.50, 1.00, 0.80, 0.05, key="ga_cx")
        tournament_size = st.slider("Tournament Size", 2, 10, 3, 1, key="ga_tourn")

    with ga_col3:
        elitism_count = st.slider("Elitism Count", 1, 10, 2, 1, key="ga_elite")
        blx_alpha = st.slider("BLX-α", 0.0, 1.0, 0.5, 0.1, key="ga_alpha")
        mutation_sigma = st.slider("Mutation σ", 0.01, 0.50, 0.10, 0.01, key="ga_sigma")

    has_gt = "risk_gt" in df_raw.columns
    if has_gt:
        fitness_metric = st.selectbox("Fitness Metric", ["mae", "mse", "distribution"], key="ga_metric")
    else:
        st.info("ℹ️ Kolom `risk_gt` tidak ditemukan. Menggunakan fitness **distribution** (unsupervised).")
        fitness_metric = "distribution"

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    if st.button(" Run GA Optimization", type="primary", key="btn_run_ga"):
        ga_params = {
            "population_size": pop_size,
            "generations": generations,
            "mutation_rate": mutation_rate,
            "crossover_rate": crossover_rate,
            "tournament_size": tournament_size,
            "elitism_count": elitism_count,
            "blx_alpha": blx_alpha,
            "mutation_sigma": mutation_sigma,
            "fitness_metric": fitness_metric,
            "random_seed": seed_val,
        }

        progress_bar = st.progress(0, text="Initializing GA...")
        status_text = st.empty()
        fitness_placeholder = st.empty()

        gen_data: List = []

        def _ui_progress(gen: int, total: int, best: float, avg: float) -> None:
            pct = gen / total
            progress_bar.progress(pct, text=f"Generation {gen}/{total}")
            status_text.markdown(
                f"**Gen {gen}/{total}** — Best fitness: `{best:.4f}` | Avg: `{avg:.4f}`"
            )
            gen_data.append((gen, best, avg))

        config_path_ga = ROOT_DIR / "configs" / "fuzzy.yaml"

        # Save data temporarily for GA to read
        temp_csv = ROOT_DIR / "outputs" / "_temp_ga_input.csv"
        temp_csv.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(temp_csv, index=False, encoding="utf-8")

        try:
            result = optimize_fuzzy(
                data_path=str(temp_csv),
                config_path=str(config_path_ga),
                ga_params=ga_params,
                progress_callback=_ui_progress,
            )
            st.session_state.ga_result = result
            st.session_state.config_ga = result["best_config"]

            # Run fuzzy with GA-optimised config
            df_ga_result = run_fuzzy_with_config(df_raw, result["best_config"])
            st.session_state.df_fuzzy_ga = df_ga_result

            progress_bar.progress(1.0, text=" Complete!")
            st.success(
                f" Optimization selesai dalam **{result['elapsed_seconds']:.1f}s** — "
                f"Best fitness: **{result['best_fitness']:.4f}**"
            )
        except Exception as e:
            st.error(f" Error: {e}")
        finally:
            if temp_csv.exists():
                temp_csv.unlink()

    # Show results if available
    if st.session_state.ga_result is not None:
        result = st.session_state.ga_result

        st.markdown("####  Fitness Curve")
        fig_fitness = _fitness_curve_chart(result["fitness_history"])
        st.plotly_chart(fig_fitness, use_container_width=True)

        st.markdown("####  Optimised Membership Functions")
        ga_cfg = result["best_config"]
        default_cfg = result["default_config"]

        mf_c1, mf_c2, mf_c3 = st.columns(3)
        with mf_c1:
            fig = _plot_membership_functions(
                default_cfg, "area_density_pct",
                "Area Density — Default vs GA", ga_cfg
            )
            st.plotly_chart(fig, use_container_width=True)
        with mf_c2:
            fig = _plot_membership_functions(
                default_cfg, "jumlah_instance",
                "Jumlah Instance — Default vs GA", ga_cfg
            )
            st.plotly_chart(fig, use_container_width=True)
        with mf_c3:
            fig = _plot_membership_functions(
                default_cfg, "confidence_score",
                "Confidence — Default vs GA", ga_cfg
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("####  Parameter Comparison")
        comp_df = _config_comparison_table(default_cfg, ga_cfg)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        if st.session_state.df_fuzzy_ga is not None:
            st.markdown("#### Risk Distribution (GA Optimized)")
            fig_ga_risk = _risk_distribution_chart(
                st.session_state.df_fuzzy_ga, "risk_level", "Risk Level — GA Optimized"
            )
            st.plotly_chart(fig_ga_risk, use_container_width=True)

            st.markdown("####  Hasil Fuzzy + GA")
            df_ga = st.session_state.df_fuzzy_ga
            display_cols = [c for c in [
                "grid_id", "area_density_pct", "jumlah_instance",
                "confidence_score", "kategori_dominan",
                "fuzzy_score", "risk_level",
            ] if c in df_ga.columns]
            st.dataframe(df_ga[display_cols], use_container_width=True, height=400)

            csv_ga_out = df_ga.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV Hasil GA", csv_ga_out, "fuzzy_results_ga.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("###  Perbandingan Default vs GA Optimized")

    if st.session_state.df_fuzzy_default is None or st.session_state.df_fuzzy_ga is None:
        st.info("ℹ️ Jalankan **Fuzzy Inference** (tab Fuzzy) dan **GA Optimization** (tab GA) terlebih dahulu untuk melihat perbandingan.")
    else:
        df_def = st.session_state.df_fuzzy_default
        df_ga = st.session_state.df_fuzzy_ga

        st.markdown("#### Metrics Comparison")
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("##### Default Fuzzy")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Avg Score", f"{df_def['fuzzy_score'].mean():.2f}")
            with m2:
                st.metric("Std Dev", f"{df_def['fuzzy_score'].std():.2f}")
            with m3:
                if "risk_gt" in df_def.columns:
                    from aerosweep.fuzzy_ga.ga_utils import encode_risk_label
                    gt_vals = df_def["risk_gt"].apply(encode_risk_label)
                    mae_def = (df_def["fuzzy_score"] - gt_vals).abs().mean()
                    st.metric("MAE", f"{mae_def:.2f}")
                else:
                    st.metric("Levels", df_def["risk_level"].nunique())

        with mc2:
            st.markdown("##### GA Optimized")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Avg Score", f"{df_ga['fuzzy_score'].mean():.2f}")
            with m2:
                st.metric("Std Dev", f"{df_ga['fuzzy_score'].std():.2f}")
            with m3:
                if "risk_gt" in df_ga.columns:
                    gt_vals = df_ga["risk_gt"].apply(encode_risk_label)
                    mae_ga = (df_ga["fuzzy_score"] - gt_vals).abs().mean()
                    st.metric("MAE", f"{mae_ga:.2f}")
                else:
                    st.metric("Levels", df_ga["risk_level"].nunique())

        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        # Side-by-side risk distribution
        st.markdown("#### Risk Distribution Comparison")
        rc1, rc2 = st.columns(2)
        with rc1:
            fig1 = _risk_distribution_chart(df_def, "risk_level", "Default")
            st.plotly_chart(fig1, use_container_width=True)
        with rc2:
            fig2 = _risk_distribution_chart(df_ga, "risk_level", "GA Optimized")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        # Score comparison scatter
        st.markdown("#### Fuzzy Score Comparison (per Grid)")
        if len(df_def) == len(df_ga):
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=df_def["fuzzy_score"],
                y=df_ga["fuzzy_score"],
                mode="markers",
                marker=dict(
                    color=[RISK_COLORS.get(r, "#999") for r in df_ga["risk_level"]],
                    size=8,
                    opacity=0.7,
                    line=dict(width=1, color="white"),
                ),
                text=[f"Grid: {gid}" for gid in df_def["grid_id"]] if "grid_id" in df_def.columns else None,
                hovertemplate="Default: %{x:.2f}<br>GA: %{y:.2f}<extra>%{text}</extra>",
            ))
            # Diagonal reference line
            max_score = max(df_def["fuzzy_score"].max(), df_ga["fuzzy_score"].max())
            fig_scatter.add_trace(go.Scatter(
                x=[0, max_score], y=[0, max_score],
                mode="lines", name="y=x",
                line=dict(color="rgba(255,255,255,0.3)", dash="dash"),
                showlegend=False,
            ))
            fig_scatter.update_layout(
                xaxis_title="Default Fuzzy Score",
                yaxis_title="GA Optimized Score",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"),
                height=450,
                margin=dict(t=20, b=40, l=40, r=20),
            )
            fig_scatter.update_xaxes(gridcolor="rgba(255,255,255,0.07)")
            fig_scatter.update_yaxes(gridcolor="rgba(255,255,255,0.07)")
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Delta table
        st.markdown("#### Delta (GA − Default)")
        if len(df_def) == len(df_ga) and "grid_id" in df_def.columns:
            delta_df = pd.DataFrame({
                "grid_id": df_def["grid_id"].values,
                "default_score": df_def["fuzzy_score"].values,
                "ga_score": df_ga["fuzzy_score"].values,
                "delta": df_ga["fuzzy_score"].values - df_def["fuzzy_score"].values,
                "default_risk": df_def["risk_level"].values,
                "ga_risk": df_ga["risk_level"].values,
                "risk_changed": df_def["risk_level"].values != df_ga["risk_level"].values,
            })
            changed = delta_df[delta_df["risk_changed"]].copy()
            st.metric("Grids with Risk Change", len(changed))
            if len(changed) > 0:
                st.dataframe(changed, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: SINGLE GRID TESTER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_tester:
    st.markdown("###  Single Grid Tester")
    st.markdown("Masukkan nilai secara manual dan lihat proses fuzzy step-by-step.")

    tc1, tc2 = st.columns([1, 2])

    with tc1:
        st.markdown("#### Input Values")
        test_density = st.slider("Area Density (%)", 0.0, 100.0, 25.0, 0.5, key="test_dens")
        test_count = st.slider("Jumlah Instance", 0, 100, 10, 1, key="test_count")
        test_conf = st.slider("Confidence Score", 0.0, 1.0, 0.70, 0.01, key="test_conf")

        categories = list(DEFAULT_CONFIG.get("category_weights", {"none": 0}).keys())
        config_path_tester = ROOT_DIR / "configs" / "fuzzy.yaml"
        if config_path_tester.exists():
            with open(config_path_tester, "r", encoding="utf-8") as f:
                tester_yaml = yaml.safe_load(f) or {}
            if "category_weights" in tester_yaml:
                categories = list(tester_yaml["category_weights"].keys())
        test_category = st.selectbox("Kategori Dominan", categories, key="test_cat")

        use_ga_config = False
        if st.session_state.config_ga is not None:
            use_ga_config = st.checkbox("Gunakan parameter GA-optimized", key="test_use_ga")

    with tc2:
        test_input = {
            "area_density_pct": test_density,
            "jumlah_instance": test_count,
            "confidence_score": test_conf,
            "kategori_dominan": test_category,
        }

        test_config = st.session_state.config_ga if use_ga_config else _get_fuzzy_config(_load_config(config_path_tester))

        # Run fuzzy evaluation
        result = evaluate_fuzzy(test_input, config=test_config)
        memberships = fuzzify(test_input, config=test_config)

        st.markdown("####  Results")

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.metric("Fuzzy Score", f"{result['fuzzy_score']:.2f}")
        with rc2:
            risk = result["risk_level"]
            st.metric("Risk Level", risk.upper())
        with rc3:
            st.metric("Category Weight", f"{result['category_weight']:.2f}")

        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        # Step 1: Fuzzification
        st.markdown("#### Step 1: Fuzzification")
        fuzz_data = []
        for var_key, var_label in [("density", "Area Density"), ("count", "Jumlah Instance"), ("confidence", "Confidence")]:
            for term, degree in memberships[var_key].items():
                fuzz_data.append({"Variable": var_label, "Term": term, "Degree": round(degree, 4)})
        st.dataframe(pd.DataFrame(fuzz_data), use_container_width=True, hide_index=True)

        # Step 2: Rule Evaluation
        st.markdown("#### Step 2: Rule Strengths")
        rule_cols = st.columns(4)
        for i, level in enumerate(["low", "medium", "high", "critical"]):
            with rule_cols[i]:
                val = result[f"rule_{level}"]
                color = RISK_COLORS[level]
                st.markdown(
                    f'<div style="text-align:center; padding:12px; border-radius:10px; '
                    f'border:2px solid {color}; margin:4px;">'
                    f'<div style="color:{color}; font-size:0.85rem; text-transform:uppercase; font-weight:600;">{level}</div>'
                    f'<div style="color:white; font-size:1.5rem; font-weight:700;">{val:.4f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Step 3: Defuzzification
        st.markdown("#### Step 3: Defuzzification → Score")
        centers = test_config.get("risk_output", {}).get("centers", DEFAULT_CONFIG["risk_output"]["centers"])
        st.markdown(f"""
        - **Metode**: Weighted Average (center of sums)
        - **Centers**: Low={centers['low']}, Medium={centers['medium']}, High={centers['high']}, Critical={centers['critical']}
        - **Raw Score** → Category Adjustment → **Final: {result['fuzzy_score']:.2f}**
        """)

        # Membership function visualisation with input markers
        st.markdown("#### Membership Functions (with Input Position)")
        mf_t1, mf_t2, mf_t3 = st.columns(3)

        for col, (var_name, input_val, title) in zip(
            [mf_t1, mf_t2, mf_t3],
            [
                ("area_density_pct", test_density, "Area Density"),
                ("jumlah_instance", float(test_count), "Jumlah Instance"),
                ("confidence_score", test_conf, "Confidence"),
            ]
        ):
            with col:
                fig = _plot_membership_functions(test_config, var_name, title)
                fig.add_vline(
                    x=input_val,
                    line=dict(color="white", width=2, dash="dash"),
                    annotation_text=f"Input: {input_val}",
                    annotation_font_color="white",
                )
                st.plotly_chart(fig, use_container_width=True)


# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center; color:#555; font-size:0.8rem;">'
    'AeroSweep - Fuzzy + GA Dashboard - Soft Computing 2025'
    '</p>',
    unsafe_allow_html=True,
)
