# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st

st.header("Development and validation of an interpretable model to predict intraoperative massive blood loss among metastatic spinal disease using machine learning techniques")
st.sidebar.title("Parameters Selection Panel")
st.sidebar.markdown("Picking up parameters")
Tumortype = st.sidebar.selectbox("Tumor type", ("Thyroid cancer", "Prostate cancer", "Breast cancer", "Renal cancer", "Lung cancer", "Hepatocellular carcinoma", "Gastrointestinal system cancer", "Urogenital cancer", "Others"))
Smoking = st.sidebar.selectbox("Smoking", ("Never", "Quitting", "Current"))
ECOG = st.sidebar.selectbox("ECOG score", ("1", "2", "3", "4"))
Surgicalprocess = st.sidebar.selectbox("Surgical process", ("Palliative decompression", "Partial resection of vertebrae", "En bloc resection of vertebrae"))
PLT = st.sidebar.slider("Preoperative platelet count (×10^9/L)", 50, 500)


if st.button("Submit"):
    rf_clf = jl.load("Xgbc_clf_final_round.pkl")
    x = pd.DataFrame([[Tumortype,Smoking, ECOG, Surgicalprocess, PLT]],
                     columns=["Tumortype","Smoking", "ECOG", "Surgicalprocess", "PLT"])
    x = x.replace(["Thyroid cancer", "Prostate cancer", "Breast cancer", "Renal cancer", "Lung cancer", "Hepatocellular carcinoma", "Gastrointestinal system cancer", "Urogenital cancer", "Others"], [1, 2, 3, 4, 5, 6, 7, 8, 9])
    x = x.replace(["Never", "Quitting", "Current"], [1, 2, 3])
    x = x.replace(["1", "2", "3", "4"], [1, 2, 3, 4])
    x = x.replace(["Palliative decompression", "Partial resection of vertebrae", "En bloc resection of vertebrae"], [1, 2, 3])

    # Get prediction
    prediction = rf_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.text(f"Probability of severe sleep disturbance: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.332:
        st.text(f"Risk group: low-risk group")
    else:
        st.text(f"Risk group: High-risk group")
    if prediction < 0.332:
        st.markdown(f"Recommendation: Routine preoperative evaluation and management with a multidisciplinary approach.")
    else:
        st.markdown(f"Recommendation: A multidisciplinary approach involving surgeons, anesthesiologists, and hematologists is crucial to ensure optimal outcomes for patients undergoing surgery for spinal metastases, particularly those in the high-risk group. Furthermore, the study recommends improving platelet counts before surgery, quitting smoking, and making appropriate surgical plans to decrease intraoperative bleeding. Intraoperative monitoring can also effectively manage bleeding complications.")
st.subheader('Model information')
st.markdown('The model was developed using the XGBoosting machine algorithm, achieving an area under the curve (AUC) of 0.857 [95%CI: 0.827, 0.877]. The external validation of the model resulted in an AUC of 0.809 [95%CI: 0.778, 0.860]. This online calculator is designed to evaluate the risk of intraoperative massive blood loss, specifically among patients with metastatic spinal tumors undergoing decompressive surgery. It is accessible at no cost and intended solely for research purposes.')