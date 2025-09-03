import streamlit as st
import pandas as pd

from utils.data import read_any_table
from utils.model import train_and_predict
from utils.viz import plot_top_cpt, plot_denials_by_insurance, plot_denials_by_physician

st.set_page_config(page_title="Healthcare Denial Reason Predictor", layout="wide")
st.title("📊 Healthcare Denial Reason Predictor")

st.markdown(
    "Upload your CSV/XLSX **once**. The app will detect headers, "
    "train on rows with a labeled **Denial Reason**, and predict the missing ones."
)

uploaded_file = st.file_uploader("Upload a single CSV/XLSX", type=["csv", "xlsx"])

if uploaded_file is not None:
    # --- Read & preprocess ---
    df_raw, header_row = read_any_table(uploaded_file)
    st.write("#### Detected header row:", header_row if header_row is not None else 0)
    st.write("#### Preview (first 10 rows)")
    st.dataframe(df_raw.head(10), use_container_width=True)

    # --- Train + Predict ---
    df_done, metrics, n_labeled, n_unlabeled = train_and_predict(df_raw)
    st.success(f"Trained on {n_labeled} labeled rows; predicted {n_unlabeled} missing rows.")

    # --- Metrics ---
    st.subheader("📈 Model Evaluation")
    if metrics["accuracy"] is not None:
        report_df = pd.DataFrame(metrics["classification_report"]).transpose().round(3)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Accuracy", f"{metrics['accuracy']*100:.2f}%")
        with col2:
            st.metric("Macro Avg F1-Score", f"{report_df.loc['macro avg','f1-score']:.2f}")
        st.dataframe(report_df, use_container_width=True)
    else:
        st.info("Not enough class diversity to compute evaluation.")

    st.markdown("---")

    # --- Completed Dataset ---
    st.subheader("📑 Completed Dataset (Original + Predicted)")
    show_cols = [
        "cpt_code","insurance_company","physician","payment","balance",
        "denial_reason","predicted_denial_reason","denial_reason_final"
    ]
    st.dataframe(df_done[show_cols], use_container_width=True)

    csv = df_done.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Completed CSV", csv, "completed_dataset.csv", "text/csv")

    st.markdown("---")

    # --- Visualizations ---
    st.subheader("📊 Visualizations")
    chart1 = plot_top_cpt(df_done)
    if chart1: st.plotly_chart(chart1, use_container_width=True)
    chart2 = plot_denials_by_insurance(df_done)
    if chart2: st.plotly_chart(chart2, use_container_width=True)
    chart3 = plot_denials_by_physician(df_done)
    if chart3: st.plotly_chart(chart3, use_container_width=True)

    # --- Insights ---
    st.subheader("📊 Insights & Recommendations")
    most_common_reason = df_done["denial_reason_final"].mode()[0] if not df_done["denial_reason_final"].empty else "N/A"
    top_cpt = df_done["cpt_code"].mode()[0] if not df_done["cpt_code"].empty else "N/A"
    top_payer = df_done["insurance_company"].mode()[0] if not df_done["insurance_company"].empty else "N/A"

    st.markdown(f"""
    **Trends:**  
    - Most frequent denial reason: **{most_common_reason}**  
    - Highest denied CPT Code: **{top_cpt}**  
    - Top payer with denials: **{top_payer}**
    """)
