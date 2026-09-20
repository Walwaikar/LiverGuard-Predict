import os
import joblib
import pandas as pd
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION & MODEL LOADING
# ==========================================
st.set_page_config(
    page_title="LiverGuard Predict",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model_and_scaler():
    """
    Loads trained Logistic Regression model and StandardScaler pipeline.
    Uses st.cache_resource to avoid re-loading artifacts on every Streamlit rerun.
    """
    model_path = "model/liver_model.pkl"
    scaler_path = "model/scaler.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error(
            "⚠️ Model or Scaler file not found! "
            "Please run `python train_model.py` first to generate `model/liver_model.pkl` and `model/scaler.pkl`."
        )
        return None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


# ==========================================
# 2. UI INPUT MODULES (SLIDERS & SELECTS)
# ==========================================
def render_patient_demographics():
    """Module for basic patient demographic details."""
    st.subheader("1. Patient Demographics")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", min_value=1, max_value=120, value=45)
    with col2:
        gender = st.selectbox("Gender", options=["Male", "Female"])
        
    return age, gender


def render_bilirubin_metrics():
    """Module for Bilirubin level."""
    st.subheader("2. Bilirubin Metrics")
    
    total_bilirubin = st.slider("Total Bilirubin (mg/dL)", min_value=0.0, max_value=50.0, value=0.8, step=0.1)
        
    return total_bilirubin


def render_enzyme_metrics():
    """Module for liver enzymes."""
    st.subheader("3. Liver Enzymes")
    col1, col2 = st.columns(2)
    
    with col1:
        alt = st.slider("Alamine Aminotransferase - ALT (IU/L)", min_value=0, max_value=2000, value=35)
    with col2:
        ast = st.slider("Aspartate Aminotransferase - AST (IU/L)", min_value=0, max_value=2000, value=40)
        
    return alt, ast


def render_protein_metrics():
    """Module for protein level."""
    st.subheader("4. Protein Level")
    
    total_proteins = st.slider("Total Proteins (g/dL)", min_value=0.0, max_value=15.0, value=6.8, step=0.1)
        
    return total_proteins


def render_medical_advice(disease_prob):
    """
    Renders tailored medical and exercise recommendations based on disease probability.
    """
    st.subheader("💡 Personalised Medical & Lifestyle Recommendations")
    
    if disease_prob >= 70.0:
        st.warning("### Risk Level: High Risk")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🩺 Medical Next Steps:**
            * **Consult a Specialist:** Schedule an immediate evaluation with a Hepatologist or Gastroenterologist.
            * **Further Testing:** Request comprehensive imaging (e.g., Liver Ultrasound, FibroScan) and a full Hepatitis panel.
            * **Medication Review:** Have a physician review all active prescriptions, OTC drugs, and supplements to avoid drug-induced liver injury.
            * **Avoid Hepatotoxins:** Strictly eliminate alcohol consumption and avoid unverified herbal supplements.
            """)
        with col2:
            st.markdown("""
            **🏋️ Exercise & Lifestyle Protocol:**
            * **Light Aerobic Activity:** Engage in low-impact walking or gentle cycling (20–30 minutes daily). Avoid intense overexertion.
            * **Dietary Adjustments:** Focus on an anti-inflammatory diet rich in vegetables, lean proteins, and low in sodium and saturated fats.
            * **Hydration:** Maintain optimal hydration unless fluid restriction has been explicitly advised by a doctor.
            """)
            
    elif 30.0 <= disease_prob < 70.0:
        st.info("### Risk Level: Moderate Risk")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🩺 Medical Next Steps:**
            * **Primary Care Follow-up:** Schedule a routine check-up with your primary care physician within 2–4 weeks.
            * **Repeat LFTs:** Monitor enzyme trends (ALT, AST, Bilirubin) with follow-up blood tests as advised by your doctor.
            * **Limit Alcohol:** Reduce or completely eliminate alcohol intake to minimize liver strain.
            """)
        with col2:
            st.markdown("""
            **🏋️ Exercise & Lifestyle Protocol:**
            * **Moderate Cardio:** Target 150 minutes per week of moderate-intensity cardio (brisk walking, swimming, cycling).
            * **Weight Management:** Focus on gradual, sustainable weight loss if BMI is elevated to reduce hepatic fat accumulation.
            * **Balanced Nutrition:** Adopt a Mediterranean-style diet low in processed sugars and refined carbohydrates.
            """)
            
    else:
        st.success("### Risk Level: Low Risk")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🩺 Medical Next Steps:**
            * **Routine Check-ups:** Continue regular annual health screenings and blood panels with your healthcare provider.
            * **Preventative Care:** Stay up-to-date with vaccinations (e.g., Hepatitis A and B).
            * **Medication Care:** Always follow dosage instructions for over-the-counter painkillers like acetaminophen/paracetamol.
            """)
        with col2:
            st.markdown("""
            **🏋️ Exercise & Lifestyle Protocol:**
            * **Regular Exercise:** Maintain a mix of aerobic conditioning and resistance training (3–5 times per week).
            * **Maintain Healthy Weight:** Continue balanced eating rich in high-fiber foods, whole grains, and healthy fats.
            * **Hydration:** Maintain good daily hydration and moderate alcohol intake.
            """)

    st.caption("⚠️ **Disclaimer:** This tool provides automated predictions based on machine learning algorithms and does not constitute official medical advice or diagnosis. Always consult a qualified medical professional.")


# ==========================================
# 3. MAIN APPLICATION PIPELINE
# ==========================================
def main():
    st.title("🫁 LiverGuard Predict")
    st.markdown("Enter the patient's lab results and clinical metrics below to assess liver disease risk.")
    st.divider()

    model, scaler = load_model_and_scaler()

    # Form encapsulating all modular input components
    with st.form(key="liver_prediction_form"):
        age, gender = render_patient_demographics()
        st.divider()
        
        total_bilirubin = render_bilirubin_metrics()
        st.divider()
        
        alt, ast = render_enzyme_metrics()
        st.divider()
        
        total_proteins = render_protein_metrics()
        st.divider()

        submit_button = st.form_submit_button(label="Predict Liver Disease Risk", use_container_width=True)

    # Form execution & inference output
    if submit_button:
        if model is None or scaler is None:
            st.error("Cannot perform prediction because the model or scaler artifacts are missing.")
            return

        # 1. Map inputs to match training encoding (Male -> 1, Female -> 0)
        gender_encoded = 1 if gender == "Male" else 0

        # 2. Build DataFrame with exact 6 feature column names and order used in train_model.py
        input_df = pd.DataFrame([{
            "Age": age,
            "Gender": gender_encoded,
            "Total_Bilirubin": total_bilirubin,
            "Alamine_Aminotransferase": alt,
            "Aspartate_Aminotransferase": ast,
            "Total_Protiens": total_proteins
        }])

        # 3. Apply StandardScaler transform and predict
        scaled_features = scaler.transform(input_df)
        prediction = model.predict(scaled_features)[0]
        prediction_probabilities = model.predict_proba(scaled_features)[0]

        # Extract probability of disease class (Class 1)
        disease_prob = prediction_probabilities[1] * 100

        # 4. Render prediction results to user
        st.subheader("Clinical Diagnostic Result")

        if prediction == 1:
            st.error("⚠️ **High Risk:** The model indicates a high likelihood of Liver Disease.")
            st.metric(label="Estimated Probability of Liver Disease", value=f"{disease_prob:.1f}%")
        else:
            healthy_prob = prediction_probabilities[0] * 100
            st.success("✅ **Low Risk:** The model indicates low likelihood of Liver Disease.")
            st.metric(label="Estimated Confidence Score (Healthy)", value=f"{healthy_prob:.1f}%")

        st.divider()

        # 5. Render tailored advice based on Probability of Liver Disease
        render_medical_advice(disease_prob)

if __name__ == "__main__":
    main()