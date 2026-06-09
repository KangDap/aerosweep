"""AeroSweep Streamlit Dashboard for Hybrid GA-Fuzzy."""

import sys
from pathlib import Path

# Add the 'src' directory to Python path so 'aerosweep' module can be found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from aerosweep.fuzzy_ga.fuzzy_logic import evaluate_fuzzy, DEFAULT_CONFIG
from aerosweep.fuzzy_ga.ga_optimizer import optimize_fuzzy, evaluate_metrics


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AeroSweep Hybrid GA-Fuzzy",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1E2127;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E333D;
        border-bottom: 2px solid #4D96FF;
    }
    .metric-card {
        background: #1E2127;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        text-align: center;
        border-left: 4px solid #4D96FF;
    }
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        color: #4D96FF;
    }
    .metric-label {
        font-size: 1em;
        color: #A0AABF;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


def run_fuzzy_on_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Run fuzzy logic on all rows with detection and return dataframe with results."""
    df_active = df[df["has_detection"] == 1].copy()
    if len(df_active) == 0:
        return pd.DataFrame()
        
    scores = []
    levels = []
    
    for _, row in df_active.iterrows():
        inputs = {
            "area_density_pct": row.get("area_density_pct", 0.0),
            "jumlah_instance": row.get("jumlah_instance", 0),
            "confidence_score": row.get("confidence_score", 0.0),
            "kategori_dominan": row.get("kategori_dominan", "none")
        }
        res = evaluate_fuzzy(inputs, config)
        scores.append(res["fuzzy_score"])
        levels.append(res["risk_level"])
        
    df_active["fuzzy_score"] = scores
    df_active["risk_level"] = levels
    return df_active


def plot_risk_distribution(df_res: pd.DataFrame, title: str):
    """Plot the distribution of risk levels."""
    if df_res.empty:
        st.warning("No data to plot.")
        return
        
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    
    order = ["low", "medium", "high", "critical"]
    palette = {"low": "#4CAF50", "medium": "#FFC107", "high": "#FF9800", "critical": "#F44336"}
    
    sns.countplot(data=df_res, x="risk_level", order=order, palette=palette, ax=ax)
    
    ax.set_title(title, color='white', pad=20)
    ax.set_xlabel("Risk Level", color='white')
    ax.set_ylabel("Count", color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
        
    st.pyplot(fig)


def main() -> None:
    st.title("🚁 AeroSweep Hybrid GA-Fuzzy Optimizer")
    st.markdown("Optimize Fuzzy Logic parameters using Genetic Algorithm to improve risk assessment accuracy based on expert ground truth.")

    data_path = "outputs/output_fitur_ann_ke_fuzzy.csv"
    df = load_data(data_path)

    if df.empty:
        st.warning(f"Could not load data from `{data_path}`. Ensure the file exists.")
        return

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Overview", 
        "🧠 Default Fuzzy", 
        "🧬 GA Optimization", 
        "⚖️ Comparison"
    ])

    with tab1:
        st.subheader("Tabular Features (from ANN pipeline)")
        st.dataframe(df.head(100), use_container_width=True)
        st.info(f"Total grids: {len(df)} | Grids with detections: {len(df[df['has_detection'] == 1])}")

    with tab2:
        st.subheader("Default Fuzzy Logic Inference")
        with st.spinner("Evaluating default fuzzy rules..."):
            df_default = run_fuzzy_on_data(df, DEFAULT_CONFIG)
            
        if not df_default.empty:
            metrics_def = evaluate_metrics(DEFAULT_CONFIG, df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">Mean Squared Error (MSE)</div>
                        <div class="metric-value">{metrics_def["MSE"]}</div>
                    </div>
                ''', unsafe_allow_html=True)
            with col2:
                st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">Mean Absolute Error (MAE)</div>
                        <div class="metric-value">{metrics_def["MAE"]}</div>
                    </div>
                ''', unsafe_allow_html=True)
                
            st.markdown("---")
            plot_risk_distribution(df_default, "Risk Distribution (Default)")
            st.dataframe(df_default[["grid_id", "area_density_pct", "jumlah_instance", "kategori_dominan", "fuzzy_score", "risk_level"]].head(20), use_container_width=True)

    with tab3:
        st.subheader("Run Genetic Algorithm Optimization")
        
        col1, col2 = st.columns(2)
        pop_size = col1.slider("Population Size", min_value=5, max_value=50, value=15, step=5)
        generations = col2.slider("Generations", min_value=5, max_value=50, value=10, step=5)
        
        if st.button("🚀 Start GA Optimization", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(gen, fitness):
                progress = gen / generations
                progress_bar.progress(progress)
                status_text.text(f"Generation {gen}/{generations} | Best Fitness: {fitness:.4f}")

            with st.spinner("Evolving parameters..."):
                best_config = optimize_fuzzy(
                    data_path=data_path, 
                    pop_size=pop_size, 
                    generations=generations, 
                    progress_callback=update_progress
                )
                
                st.session_state["best_config"] = best_config
                st.success("Optimization Complete!")
                st.balloons()

        if "best_config" in st.session_state:
            st.markdown("### 🏆 Best Found Configuration")
            st.json(st.session_state["best_config"]["variables"])

    with tab4:
        st.subheader("Comparison: Default vs GA-Optimized")
        
        if "best_config" not in st.session_state:
            st.warning("Please run GA Optimization in the previous tab first to see the comparison.")
        else:
            best_config = st.session_state["best_config"]
            
            with st.spinner("Evaluating Optimized Fuzzy rules..."):
                df_ga = run_fuzzy_on_data(df, best_config)
                metrics_ga = evaluate_metrics(best_config, df)
                metrics_def = evaluate_metrics(DEFAULT_CONFIG, df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Default Fuzzy")
                st.markdown(f"**MSE:** {metrics_def['MSE']} | **MAE:** {metrics_def['MAE']}")
                plot_risk_distribution(df_default, "Risk Distribution (Default)")
                
            with col2:
                st.markdown("#### GA-Optimized Fuzzy")
                mse_diff = metrics_def['MSE'] - metrics_ga['MSE']
                mae_diff = metrics_def['MAE'] - metrics_ga['MAE']
                st.markdown(f"**MSE:** {metrics_ga['MSE']} 🟢 (-{mse_diff:.2f}) | **MAE:** {metrics_ga['MAE']} 🟢 (-{mae_diff:.2f})")
                plot_risk_distribution(df_ga, "Risk Distribution (GA-Optimized)")
                
            st.info("The GA optimization adjusts the membership parameters so that the fuzzy inference aligns more closely with the expert ground truth, often shifting priority classifications to better reflect critical states.")


if __name__ == "__main__":
    main()
