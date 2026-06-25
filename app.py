import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Add the workspace root to python path to allow absolute imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import RAW_DATA_DIR, MODELS_DIR, OUTPUTS_DIR, EXPERIMENTS_DIR
from src.inference.predict import predict_on_day

# Page Configuration
st.set_page_config(
    page_title="Flarion AI - Aditya-L1 Solar Flare Hub",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium UI Header
st.markdown("""
<div style="background: linear-gradient(135deg, #1e1b4b, #311042); padding: 30px; border-radius: 20px; text-align: center; margin-bottom: 30px; border: 1px solid #4c1d95; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);">
    <h1 style="color: #f5f3ff; margin: 0; font-family: 'Outfit', 'Inter', sans-serif; font-size: 3rem; font-weight: 800; letter-spacing: -0.05em; text-shadow: 0 4px 6px rgba(0,0,0,0.3);">
        🌞 Flarion
    </h1>
    <p style="color: #c084fc; margin: 10px 0 0 0; font-size: 1.25rem; font-weight: 500; opacity: 0.9;">
        Aditya-L1 Solar Flare Forecasting & Analytics Hub
    </p>
    <div style="display: inline-block; margin-top: 15px; padding: 4px 12px; background: rgba(192, 132, 252, 0.1); border-radius: 20px; border: 1px solid rgba(192, 132, 252, 0.2); font-size: 0.85rem; color: #e9d5ff; font-weight: 600;">
        🛰️ Payload Integration: SoLEXS & HEL1OS
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <h3 style="margin: 0; color: #c084fc;">⚙️ Control Panel</h3>
    <hr style="margin: 10px 0; border-color: rgba(192, 132, 252, 0.2);"/>
</div>
""", unsafe_allow_html=True)

# Data source selection
data_source = st.sidebar.radio(
    "📂 Choose Data Source",
    options=["Discovered Datasets", "Upload Custom FITS Files"],
    help="Discovered Datasets: Use pre-downloaded observation days in data/raw.\nUpload Custom FITS Files: Upload your own SoLEXS and HEL1OS FITS lightcurve files."
)

# Initialize variables
solexs_file = None
helios_file = None

# 1. Discover Date Folders or setup custom upload fields
if data_source == "Discovered Datasets":
    if not os.path.exists(RAW_DATA_DIR):
        st.sidebar.error(f"Raw data folder not found at: {RAW_DATA_DIR}")
        dates = []
    else:
        dates = sorted([d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d)) and d != "custom_upload"])

    if dates:
        selected_date = st.sidebar.selectbox(
            "📅 Select Observation Day",
            dates,
            index=len(dates) - 1 if len(dates) > 0 else 0,
            help="Select a date folder from raw observations to run predictions on."
        )
    else:
        selected_date = None
        st.sidebar.warning("No date subfolders found in data/raw/")
        
    st.sidebar.markdown("---")
    # Detector Options
    detector = st.sidebar.selectbox(
        "📡 HEL1OS Detector Band",
        options=["all", "cdte1", "cdte2", "czt1", "czt2"],
        format_func=lambda x: "Combined All Detectors" if x == "all" else f"Detector: {x.upper()}",
        key="discovered_detector"
    )
else:
    selected_date = "custom_upload"
    st.sidebar.info("📤 Upload satellite FITS files below:")
    solexs_file = st.sidebar.file_uploader(
        "1. SoLEXS Soft X-Ray (.lc.gz)",
        type=["gz", "fits"],
        help="Upload the SoLEXS spectrum lightcurve file (.lc.gz)."
    )
    helios_file = st.sidebar.file_uploader(
        "2. HEL1OS Hard X-Ray (.fits)",
        type=["fits"],
        help="Upload the HEL1OS lightcurve FITS file for a specific detector."
    )
    
    detector = st.sidebar.selectbox(
        "📡 Uploaded HEL1OS Detector Type",
        options=["cdte1", "cdte2", "czt1", "czt2"],
        help="Select the exact detector type corresponding to your uploaded HEL1OS file.",
        key="custom_detector"
    )

# 2. Select Inference Task
task = st.sidebar.radio(
    "🎯 Select Prediction Task",
    options=["detection", "forecast", "classification"],
    format_func=lambda x: {
        "detection": "Detection (Nowcasting - Active Flare?)",
        "forecast": "Forecasting (Flare in next 5-30m?)",
        "classification": "Classification (Intensity class: C/M/X)"
    }[x],
    help="Detection: nowcasting (current flare).\nForecasting: prediction ahead.\nClassification: predicting specific flare magnitude."
)

# 3. Discover Available Models for the Selected Task
available_models = ["ensemble"]
if os.path.exists(MODELS_DIR):
    for f in os.listdir(MODELS_DIR):
        if f.startswith(f"{task}_") and f.endswith("_latest.joblib"):
            # Extract name (e.g. "random_forest" from "detection_random_forest_latest.joblib")
            model_name = f[len(task) + 1:-len("_latest.joblib")]
            available_models.append(model_name)

# Model selection dropdown
model_selection = st.sidebar.selectbox(
    "🤖 Machine Learning Model",
    options=available_models,
    format_func=lambda x: "🔗 Weighted Ensemble (All Models Combined)" if x == "ensemble" else f"🧠 {x.replace('_', ' ').title()}",
    help="Select the model architecture or use the combined ensemble."
)

st.sidebar.markdown("---")

# Predict trigger button
run_prediction = st.sidebar.button("⚡ Run Live Inference", use_container_width=True)

st.sidebar.markdown("""
<div style="margin-top: 30px; padding: 15px; background: rgba(30, 27, 75, 0.4); border-radius: 10px; border: 1px solid rgba(76, 29, 149, 0.4); font-size: 0.85rem; color: #c084fc;">
    <strong>💡 Tip:</strong> Random Forest generally performs best on this dataset, but the Weighted Ensemble incorporates predictions from XGBoost and LightGBM to yield the most stable output!
</div>
""", unsafe_allow_html=True)


# ----------------- MAIN PANEL tabs -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Predictions & Lightcurves", 
    "📈 Leaderboards & Experiments", 
    "🧬 Physics Feature Importance",
    "📂 Data Explorer"
])

# Resolve model path
if model_selection == "ensemble":
    model_path = "ensemble"
else:
    model_path = os.path.join(MODELS_DIR, f"{task}_{model_selection}_latest.joblib")


# Run prediction on click or if session state predictions are available
if run_prediction and selected_date:
    proceed = True
    if data_source == "Upload Custom FITS Files":
        if not solexs_file or not helios_file:
            st.sidebar.error("❌ Please upload both SoLEXS and HEL1OS files.")
            proceed = False
        else:
            with st.spinner("Processing uploaded files..."):
                try:
                    custom_dir = os.path.join(RAW_DATA_DIR, "custom_upload")
                    solexs_dir = os.path.join(custom_dir, "SOLEXS")
                    helios_dir = os.path.join(custom_dir, "HEL1OS")
                    
                    # Ensure dirs exist
                    os.makedirs(solexs_dir, exist_ok=True)
                    os.makedirs(helios_dir, exist_ok=True)
                    
                    # Clean old files
                    for root_d in [solexs_dir, helios_dir]:
                        if os.path.exists(root_d):
                            for f_name in os.listdir(root_d):
                                file_path = os.path.join(root_d, f_name)
                                try:
                                    if os.path.isfile(file_path):
                                        os.unlink(file_path)
                                except Exception:
                                    pass
                                    
                    # Save SoLEXS file
                    solexs_target = os.path.join(solexs_dir, "solexs_data.lc.gz")
                    with open(solexs_target, "wb") as f:
                        f.write(solexs_file.getbuffer())
                        
                    # Save HEL1OS file
                    helios_target = os.path.join(helios_dir, f"lightcurve_{detector}.fits")
                    with open(helios_target, "wb") as f:
                        f.write(helios_file.getbuffer())
                except Exception as e:
                    st.sidebar.error(f"❌ Failed to prepare custom files: {e}")
                    proceed = False

    if proceed:
        display_name = "Uploaded Custom Files" if selected_date == "custom_upload" else selected_date
        with st.spinner(f"Ingesting & aligning payloads for {display_name}... Running {task} models..."):
            try:
                df_output = predict_on_day(
                    date_str=selected_date,
                    model_path=model_path,
                    task=task,
                    detector=detector
                )
                st.session_state["df_output"] = df_output
                st.session_state["last_run_params"] = {
                    "date": display_name,
                    "task": task,
                    "model": model_selection
                }
                st.toast("Inference run complete!", icon="✅")
            except Exception as e:
                st.error(f"Error during inference execution: {str(e)}")
                st.exception(e)


# Load results from state if they exist
df_output = st.session_state.get("df_output", None)
last_params = st.session_state.get("last_run_params", None)

with tab1:
    if df_output is not None and last_params is not None:
        # Check if the parameters match current sidebar selection
        st.markdown(f"### Results for **{last_params['date']}** (Task: **{last_params['task'].upper()}** | Model: **{last_params['model'].upper()}**)")
        
        # 1. Summary Cards
        c1, c2, c3, c4 = st.columns(4)
        
        # Max SoLEXS and HEL1OS
        max_solexs = df_output['solexs_counts'].max()
        max_helios = df_output['helios_counts'].max()
        
        c1.metric("Peak Soft X-Ray (SoLEXS)", f"{max_solexs:.1f} c/s")
        c2.metric("Peak Hard X-Ray (HEL1OS)", f"{max_helios:.1f} c/s")
        
        # Predict Class Counts & Summary
        pred_counts = df_output['predicted_class'].value_counts()
        
        # Label Class Names
        if last_params['task'] == 'classification':
            class_names = {0: "Quiet", 1: "C-Class", 2: "M-Class", 3: "X-Class"}
            max_class_val = df_output['predicted_class'].max()
            max_class_name = class_names.get(max_class_val, "Quiet")
            
            c3.metric("Strongest Flare Predicted", max_class_name)
            
            flare_rows = df_output[df_output['predicted_class'] > 0]
            flare_duration = len(flare_rows) # assuming 1-second cadence
            c4.metric("Total Flare Active Time", f"{flare_duration}s" if flare_duration > 0 else "0s")
        else:
            flares_detected = 1 in df_output['predicted_class'].values
            status_text = "🚨 Flare Detected" if flares_detected else "🟢 Quiet (No Flare)"
            c3.metric("Pipeline Alerts", status_text)
            
            flare_duration = (df_output['predicted_class'] == 1).sum() # count seconds
            c4.metric("Total Active/Risk Time", f"{flare_duration}s")
            
        st.markdown("---")
        
        # 2. Plotting Fluxes
        st.subheader("🛰️ Multi-Payload Solar Lightcurves (Count Rates)")
        
        # Prep chart data
        plot_df = df_output.copy()
        
        # Handle time axis
        time_col = 'TIME'
        if plot_df['TIME'].min() > 1e9: # Unix timestamp
            plot_df['Time (UTC)'] = pd.to_datetime(plot_df['TIME'], unit='s')
            time_col = 'Time (UTC)'
            
        chart_data = plot_df.set_index(time_col)[['solexs_counts', 'helios_counts']]
        st.line_chart(chart_data, y=["solexs_counts", "helios_counts"], color=["#f97316", "#a855f7"], height=350)
        
        # 3. Plotting Predictions / Confidences
        st.subheader("🎯 Prediction Alerts & Model Confidences")
        
        p1, p2 = st.columns([3, 1])
        with p1:
            if 'prediction_probability' in df_output.columns:
                # Plot prediction probability
                prob_chart = plot_df.set_index(time_col)[['prediction_probability', 'predicted_class']]
                st.line_chart(prob_chart, y=["prediction_probability", "predicted_class"], color=["#3b82f6", "#ef4444"], height=250)
            else:
                # Plot predicted class only
                class_chart = plot_df.set_index(time_col)[['predicted_class']]
                st.line_chart(class_chart, color=["#ef4444"], height=200)
                
        with p2:
            st.markdown("""
            <div style="background: rgba(17, 24, 39, 0.4); padding: 15px; border-radius: 10px; border: 1px solid rgba(76, 29, 149, 0.2); height: 100%;">
                <h4 style="margin: 0 0 10px 0; color: #c084fc; font-size: 1rem;">Plot Legend</h4>
                <ul style="padding-left: 15px; margin: 0; font-size: 0.85rem; line-height: 1.6; color: #d1d5db;">
                    <li><strong style="color: #f97316;">Orange:</strong> SoLEXS soft X-rays (1-22 keV). Rises during thermal heating phase of flares.</li>
                    <li><strong style="color: #a855f7;">Purple:</strong> HEL1OS hard X-rays (10-150 keV). Peaks sharply during impulsive particle acceleration.</li>
                    <li><strong style="color: #3b82f6;">Blue Line:</strong> Real-time probability score generated by the classifier (0% to 100%).</li>
                    <li><strong style="color: #ef4444;">Red Line:</strong> Final Binary Alert (0 = Quiet, 1 = Flare Active / High Risk).</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.info("👈 Please select a date and model in the sidebar and click 'Run Live Inference' to display results.")


with tab2:
    st.markdown("### 🏆 ML Leaderboards & Model Performance")
    st.markdown("These tables compare the models on our chronological test dataset to track which architecture operates best.")
    
    comp_file = os.path.join(EXPERIMENTS_DIR, f"model_comparison_{task}.csv")
    if os.path.exists(comp_file):
        try:
            comp_df = pd.read_csv(comp_file)
            
            # Sort by Macro F1
            if 'Macro F1' in comp_df.columns:
                comp_df = comp_df.sort_values(by='Macro F1', ascending=False)
                
            st.dataframe(
                comp_df.style.highlight_max(subset=['Macro F1', 'Accuracy', 'PR-AUC', 'ROC-AUC'], color='#311042'),
                use_container_width=True
            )
            
            # Show best model alert
            if not comp_df.empty:
                best_model = comp_df.iloc[0]['Model Name']
                best_f1 = comp_df.iloc[0]['Macro F1']
                st.success(f"🏆 **Leaderboard Champion:** **{best_model.upper()}** holds the highest Macro F1 score of **{best_f1:.4f}** for the **{task.upper()}** task!")
                
        except Exception as e:
            st.error(f"Could not load leaderboard comparison file: {e}")
    else:
        st.warning(f"No experiment leaderboard comparison file found at {comp_file} for task: {task}. Have you run training for all models?")


with tab3:
    st.markdown("### 🧬 Physics Feature Importance (Explainable AI)")
    st.markdown("This section maps the relative contribution of different astronomical observations, derivatives, and lags used by the model.")
    
    # Load feature importance file for current selection
    fi_file = os.path.join(MODELS_DIR, f"{task}_{model_selection}_feature_importance.csv")
    
    # If ensemble, look for the best model in the ensemble to explain
    if model_selection == "ensemble":
        # Find best model from leaderboard
        comp_file = os.path.join(EXPERIMENTS_DIR, f"model_comparison_{task}.csv")
        if os.path.exists(comp_file):
            try:
                comp_df = pd.read_csv(comp_file)
                if not comp_df.empty:
                    best_model_name = comp_df.iloc[0]['Model Name'].lower().replace(" ", "_")
                    fi_file = os.path.join(MODELS_DIR, f"{task}_{best_model_name}_feature_importance.csv")
                    st.info(f"Showing feature importance rankings for the best individual model in the ensemble: **{best_model_name.upper()}**.")
            except Exception:
                pass
                
    if os.path.exists(fi_file):
        try:
            fi_df = pd.read_csv(fi_file)
            # Reorder by importance
            importance_col = [c for c in fi_df.columns if c not in ['feature', 'Feature']][0]
            fi_df = fi_df.sort_values(by=importance_col, ascending=True).tail(15) # Show top 15
            
            # Plot using matplotlib for clean, themed visualization
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#0f172a') # dark slate
            ax.set_facecolor('#0f172a')
            
            bars = ax.barh(fi_df['feature'], fi_df[importance_col], color='#a855f7', edgecolor='#c084fc', height=0.6)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#475569')
            ax.spines['bottom'].set_color('#475569')
            ax.tick_params(colors='#94a3b8', labelsize=10)
            ax.xaxis.grid(True, linestyle='--', alpha=0.1, color='#94a3b8')
            ax.set_title("Top 15 Predictive Physics Features", color='#f8fafc', fontsize=14, pad=15)
            
            # Adjust layout
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Write a quick explanation of what these features represent
            st.markdown("""
            **💡 How to read this chart:**
            * **Flux derivatives (`*_deriv`):** Capture the *rate of change* in X-ray flux. High rates of change are the earliest indicators of an impulsive solar flare.
            * **Lags (`*_lag_*`):** Let the model compare current X-ray levels against 30-second or 60-second prior values to compute flare acceleration.
            * **Rolling Statistics (`*_roll_*`):** Smooth out instrumentation spikes (noise) and capture longer-term trends.
            """)
        except Exception as e:
            st.error(f"Could not load feature importance chart: {e}")
    else:
        st.warning(f"Feature importance file not found at {fi_file}. Train this model to generate physics weights.")


with tab4:
    st.markdown("### 📂 Data Explorer & Exports")
    if df_output is not None:
        st.markdown(f"Below is the raw prediction outputs for the selected observation day: **{selected_date}**")
        st.dataframe(df_output, use_container_width=True)
        
        # Download prediction CSV
        out_filename = f"predictions_{task}_{selected_date.replace('-', '')}.csv"
        csv_data = df_output.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv_data,
            file_name=out_filename,
            mime='text/csv',
            use_container_width=True
        )
    else:
        st.info("Please run live predictions from the sidebar control panel to load data explorer views.")
