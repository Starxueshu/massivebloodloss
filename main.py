# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st

st.header("Development and validation of a web-based AI prediction model to assess intraoperative massive blood loss among metastatic spinal disease using machine learning techniques")
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
    st.text(f"Predicted risk of intraoperative massive blood loss: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.332:
        st.text(f"Risk group: low-risk group")
    else:
        st.text(f"Risk group: High-risk group")
    if prediction < 0.332:
        st.markdown(f"Regarding low-risk individuals, while those have a lower likelihood of experiencing massive blood loss, it is essential to closely monitor their intraoperative blood loss and coagulation parameters during surgery. This enables prompt detection and intervention in case of unexpected bleeding. Furthermore, in individuals assessed as low-risk, a more conservative approach to blood management may be suitable. This includes avoiding unnecessary blood product transfusions and restricting interventions to specific clinical indications.")
    else:
        st.markdown(f"High-risk individuals should be thoroughly evaluated and optimized before surgery. This includes managing any underlying comorbidities, optimizing blood parameters (such as platelet count), and considering preoperative pharmacological interventions to improve coagulation function. In addition, a comprehensive blood management plan should be implemented for high-risk patients. This may involve preoperative autologous blood donation, intraoperative cell salvage, or targeted administration of blood products based on individual needs. What’s more, spine surgeons should consider employing strategies to minimize blood loss during the surgical procedure, such as meticulous hemostasis, minimizing tissue trauma, and utilizing advanced surgical techniques, including minimally invasive approaches when appropriate.")
st.subheader('Model information')
st.markdown('The AI prediction model was developed using the XGBoosting machine algorithm, achieving an area under the curve (AUC) of 0.857 [95%CI: 0.827, 0.877]. The external validation of the model resulted in an AUC of 0.809 [95%CI: 0.778, 0.860]. This online AI calculator is designed to evaluate the risk of intraoperative massive blood loss specifically among patients with metastatic spinal tumors undergoing decompressive surgery. It is accessible at no cost and intended solely for research purposes.')
