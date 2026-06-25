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
    page_icon="",
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
def load_data(path: str, cache_key: str = "") -> pd.DataFrame:
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
    st.title(" AeroSweep Hybrid GA-Fuzzy Optimizer")
    st.markdown("Optimize Fuzzy Logic parameters using Genetic Algorithm to improve risk assessment accuracy based on expert ground truth.")

    # --- SIDEBAR: INPUT HASIL ANN & PANDUAN ---
    st.sidebar.header(" Input Hasil ANN")
    st.sidebar.markdown("""
    Unggah file CSV hasil ekstraksi fitur dari pipeline Artificial Neural Network (ANN) Anda untuk dianalisis dan dioptimasi menggunakan logika Fuzzy & GA.
    """)
    
    uploaded_file = st.sidebar.file_uploader("Pilih file CSV Hasil ANN", type=["csv"])
    cache_key = ""
    
    custom_data_path = Path("outputs/uploaded_fitur_ann_ke_fuzzy.csv")
    
    if uploaded_file is not None:
        save_dir = Path("outputs")
        save_dir.mkdir(exist_ok=True)
        with open(custom_data_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        data_path = str(custom_data_path)
        cache_key = f"{uploaded_file.name}_{uploaded_file.size}"
        st.sidebar.success(f"File `{uploaded_file.name}` berhasil diunggah dan aktif!")
    elif custom_data_path.exists():
        data_path = str(custom_data_path)
        import os
        cache_key = f"custom_{os.path.getmtime(custom_data_path)}"
        st.sidebar.success("️ Menggunakan data CSV kustom / hasil penambahan dari inferensi ANN.")
        if st.sidebar.button(" Reset ke Data Default"):
            custom_data_path.unlink(missing_ok=True)
            st.rerun()
    else:
        data_path = "outputs/output_fitur_ann_ke_fuzzy.csv"
        st.sidebar.info("Menggunakan data default (`outputs/output_fitur_ann_ke_fuzzy.csv`).")

    st.sidebar.markdown("---")
    st.sidebar.header(" Panduan Penggunaan & Format Input")
    with st.sidebar.expander("Lihat Panduan Format CSV ANN", expanded=False):
        st.markdown("""
        **Format Kolom CSV yang Dibutuhkan:**
        File CSV hasil ANN harus memiliki kolom-kolom berikut agar kompatibel dengan evaluasi Fuzzy & GA:
        
        1. `grid_id`: Identifier unik untuk setiap grid/region (contoh: `grid_0_0`).
        2. `has_detection`: Flag integer (`1` jika ada deteksi anomali/objek, `0` jika tidak ada).
        3. `area_density_pct`: Persentase kepadatan area deteksi (rentang `0.0` - `100.0`).
        4. `jumlah_instance`: Jumlah instans/objek yang terdeteksi (integer, `0` - `100`).
        5. `confidence_score`: Nilai kepercayaan dari prediksi ANN (rentang `0.0` - `1.0`).
        6. `kategori_dominan`: Kategori objek dominan (contoh: `Asbestos`, `Hazardous`, `Vehicles`, `none`).

        **Alur Kerja (Workflow):**
        1. **Tab ANN Inference (CV)**: Unggah citra UAV untuk menjalankan segmentasi instans YOLO dan otomatis mengekstrak fitur ke tabel Fuzzy.
        2. **Input Data**: Atau unggah file CSV Anda pada uploader di atas.
        3. **Tab Data Overview**: Periksa apakah data hasil ANN berhasil dimuat dengan benar.
        4. **Tab Default Fuzzy**: Lihat hasil perhitungan skor risiko (risk level) menggunakan aturan fuzzy bawaan.
        5. **Tab GA Optimization**: Jalankan Algoritma Genetika (GA) untuk mengoptimalkan parameter keanggotaan (membership function) fuzzy agar sesuai dengan *ground truth* pakar.
        6. **Tab Comparison**: Bandingkan performa (MSE & MAE) antara Fuzzy Default vs GA-Optimized.
        """)

    df = load_data(data_path, cache_key=cache_key)

    if df.empty:
        st.warning(f"Could not load data from `{data_path}`. Ensure the file exists.")
        return

    # Tabs
    tab_ann, tab1, tab2, tab3, tab4 = st.tabs([
        "️ ANN Inference (CV)",
        " Data Overview", 
        " Default Fuzzy", 
        " GA Optimization", 
        "️ Comparison"
    ])

    with tab_ann:
        st.subheader("️ Segmen ANN: YOLO Instance Segmentation & Ekstraksi Fitur")
        st.markdown("""
        Jalankan model Artificial Neural Network (YOLO Instance Segmentation) secara langsung pada citra UAV/Grid. 
        Sistem akan mendeteksi objek/sampah, menghitung kepadatan area, koordinat centroid, jumlah instans, serta kategori material dominan untuk diteruskan ke sistem Fuzzy.
        """)
        
        col_model, col_img = st.columns(2)
        
        # --- MODEL WEIGHT SELECTION / UPLOAD ---
        with col_model:
            st.markdown("#### 1. Persiapan Model YOLO (`best.pt`)")
            default_model_dir = Path("models/nn_weights/aerosweep_trained/weights")
            default_model_path = default_model_dir / "best.pt"
            
            if default_model_path.exists():
                st.success(f"️ Model terlatih terdeteksi di `{default_model_path}`")
                active_model_path = default_model_path
            else:
                st.info("️ File `best.pt` belum ada di repositori. Unggah file `best.pt` Anda atau gunakan base model pre-trained.")
                uploaded_model = st.file_uploader("Unggah file model (best.pt / .pt)", type=["pt"])
                use_base_model = st.checkbox("Gunakan Base Model YOLO11n-seg (Otomatis unduh jika belum ada)", value=False)
                
                if uploaded_model is not None:
                    default_model_dir.mkdir(parents=True, exist_ok=True)
                    with open(default_model_path, "wb") as f:
                        f.write(uploaded_model.getbuffer())
                    st.success(f"️ Model `{uploaded_model.name}` berhasil diunggah!")
                    active_model_path = default_model_path
                elif use_base_model:
                    with st.spinner("Mengunduh/Memuat base model yolo11n-seg.pt..."):
                        from ultralytics import YOLO
                        _temp_model = YOLO("yolo11n-seg.pt")
                    active_model_path = Path("yolo11n-seg.pt")
                    st.success("️ Base model `yolo11n-seg.pt` siap digunakan!")
                else:
                    active_model_path = None
                    st.warning("️ Silakan unggah model `.pt` atau centang opsi Base Model di atas untuk melanjutkan.")

            st.markdown("---")
            st.markdown("#### Pengaturan Inferensi")
            conf_thresh = st.slider("Confidence Threshold", min_value=0.05, max_value=1.0, value=0.25, step=0.05)
            iou_thresh = st.slider("IoU Threshold", min_value=0.05, max_value=1.0, value=0.45, step=0.05)

        # --- IMAGE UPLOAD & INFERENCE ---
        with col_img:
            st.markdown("#### 2. Unggah Citra UAV / Grid")
            uploaded_image = st.file_uploader("Pilih gambar grid UAV (.jpg / .png)", type=["jpg", "jpeg", "png"])
            
            if uploaded_image is not None:
                grid_dir = Path("outputs/uploaded_grids")
                grid_dir.mkdir(parents=True, exist_ok=True)
                img_path = grid_dir / uploaded_image.name
                with open(img_path, "wb") as f:
                    f.write(uploaded_image.getbuffer())
                
                st.image(uploaded_image, caption=f"Citra Input: {uploaded_image.name}", use_container_width=True)
                
                if active_model_path is not None:
                    if st.button(" Jalankan Inferensi ANN & Ekstraksi Fitur", type="primary"):
                        with st.spinner("Menjalankan model YOLO Instance Segmentation..."):
                            from aerosweep.neuro.inference_cv import FeatureExtractor
                            try:
                                extractor = FeatureExtractor(
                                    model_path=str(active_model_path),
                                    config_path="configs/cv.yaml",
                                    conf_threshold=conf_thresh,
                                    iou_threshold=iou_thresh,
                                    device="cpu",
                                    save_annotated=True,
                                    annotated_dir="outputs/annotated_grids"
                                )
                                row_result = extractor.process_single_grid(img_path)
                                st.session_state["ann_result"] = row_result
                                st.session_state["ann_img_stem"] = img_path.stem
                                st.success("️ Inferensi berhasil!")
                            except Exception as e:
                                st.error(f"Terjadi kesalahan saat inferensi: {e}")
                else:
                    st.error("️ Model belum siap. Selesaikan Persiapan Model di sebelah kiri.")

        # --- DISPLAY RESULTS & INTEGRATE WITH FUZZY ---
        if "ann_result" in st.session_state:
            st.markdown("---")
            st.markdown("###  Hasil Ekstraksi Fitur ANN (Computer Vision)")
            res = st.session_state["ann_result"]
            img_stem = st.session_state["ann_img_stem"]
            
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">Jumlah Instance</div>
                        <div class="metric-value">{res["jumlah_instance"]}</div>
                    </div>
                ''', unsafe_allow_html=True)
            with mc2:
                st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">Area Density (%)</div>
                        <div class="metric-value">{res["area_density_pct"]:.2f}%</div>
                    </div>
                ''', unsafe_allow_html=True)
            with mc3:
                st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">Kategori Dominan</div>
                        <div class="metric-value" style="font-size:1.4em;">{res["kategori_dominan"]}</div>
                    </div>
                ''', unsafe_allow_html=True)
            with mc4:
                st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">Avg Confidence</div>
                        <div class="metric-value">{res["confidence_score"]:.2f}</div>
                    </div>
                ''', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            col_annotated, col_action = st.columns([2, 1])
            with col_annotated:
                annotated_file = Path("outputs/annotated_grids") / f"{img_stem}_annotated.jpg"
                if annotated_file.exists():
                    st.image(str(annotated_file), caption="Hasil Deteksi YOLO (Masks, Boxes, Centroid)", use_container_width=True)
                else:
                    st.info("Citra teranotasi tidak ditemukan atau tidak ada deteksi.")
                    
            with col_action:
                st.markdown("####  Kirim ke Pipeline Fuzzy")
                st.markdown("Tambahkan hasil ekstraksi fitur citra ini langsung ke tabel data Fuzzy & GA untuk dianalisis tingkat risikonya.")
                if st.button(" Masukkan ke Tabel Data Fuzzy", type="primary"):
                    new_row_df = pd.DataFrame([res])
                    new_row_df["has_detection"] = 1 if res["jumlah_instance"] > 0 else 0
                    new_row_df["processed_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    custom_path = Path("outputs/uploaded_fitur_ann_ke_fuzzy.csv")
                    if custom_path.exists():
                        existing_df = pd.read_csv(custom_path)
                        updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
                    else:
                        default_df = pd.read_csv("outputs/output_fitur_ann_ke_fuzzy.csv")
                        updated_df = pd.concat([default_df, new_row_df], ignore_index=True)
                    
                    updated_df.to_csv(custom_path, index=False)
                    st.session_state["last_added_grid"] = res["grid_id"]
                    st.success("️ Berhasil ditambahkan! Silakan buka tab ** Data Overview** atau ** Default Fuzzy** untuk melihat hasilnya.")
                    st.balloons()

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
        
        if st.button(" Start GA Optimization", type="primary"):
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
            st.markdown("###  Best Found Configuration")
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
                st.markdown(f"**MSE:** {metrics_ga['MSE']}  (-{mse_diff:.2f}) | **MAE:** {metrics_ga['MAE']}  (-{mae_diff:.2f})")
                plot_risk_distribution(df_ga, "Risk Distribution (GA-Optimized)")
                
            st.info("The GA optimization adjusts the membership parameters so that the fuzzy inference aligns more closely with the expert ground truth, often shifting priority classifications to better reflect critical states.")


if __name__ == "__main__":
    main()
