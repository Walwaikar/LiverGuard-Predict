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
    Load the trained Logistic Regression model
    and StandardScaler.
    """

    model_path = "model/liver_model.pkl"
    scaler_path = "model/scaler.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error(
            "⚠️ Model or Scaler file not found. "
            "Please run train_model.py first."
        )
        return None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler


# ==========================================
# 2. USER INPUT MODULES
# ==========================================

def render_patient_information():
    """
    Collect basic patient information.
    """

    st.subheader("1. Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider(
            "Age",
            min_value=1,
            max_value=120,
            value=45,
            help="Enter the patient's age in years."
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            options=["Male", "Female"],
            help="Select the patient's gender."
        )

    return age, gender


def render_bilirubin():
    """
    Collect bilirubin information.
    """

    st.subheader("2. Bilirubin Level")

    total_bilirubin = st.slider(
        "Bilirubin Level (Total Bilirubin)",
        min_value=0.0,
        max_value=50.0,
        value=0.8,
        step=0.1,
        help=(
            "Enter the Total Bilirubin value from the patient's "
            "blood test report. Unit: mg/dL."
        )
    )

    st.caption(
        "💡 Bilirubin is a substance measured in blood that can "
        "provide information about liver function."
    )

    return total_bilirubin


def render_liver_enzymes():
    """
    Collect ALT and AST values.
    """

    st.subheader("3. Liver Enzyme Levels")

    st.caption(
        "💡 ALT and AST are enzymes commonly measured in blood "
        "tests to provide information about liver health."
    )

    col1, col2 = st.columns(2)

    with col1:
        alt = st.slider(
            "ALT",
            min_value=0,
            max_value=2000,
            value=35,
            help=(
                "Enter the ALT value from the blood test report. "
                "Unit: IU/L."
            )
        )

    with col2:
        ast = st.slider(
            "AST",
            min_value=0,
            max_value=2000,
            value=40,
            help=(
                "Enter the AST value from the blood test report. "
                "Unit: IU/L."
            )
        )

    return alt, ast


def render_protein_level():
    """
    Collect total protein value.
    """

    st.subheader("4. Protein Level")

    total_proteins = st.slider(
        "Total Protein Level",
        min_value=0.0,
        max_value=15.0,
        value=6.8,
        step=0.1,
        help=(
            "Enter the Total Protein value from the patient's "
            "blood test report. Unit: g/dL."
        )
    )

    st.caption(
        "💡 Total protein measures the amount of certain proteins "
        "present in the blood."
    )

    return total_proteins


# ==========================================
# 3. HEALTH GUIDANCE
# ==========================================

def render_health_guidance(disease_prob):
    """
    Display general health guidance based on the
    estimated model probability.
    """

    st.subheader("💡 Health Guidance")

    # ==========================================
    # HIGHER PREDICTED RISK GUIDANCE
    # ==========================================

    if disease_prob >= 70.0:

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                **🩺 What you can consider doing:**

                - Consider discussing this result with a qualified healthcare professional.
                - A healthcare professional may recommend additional blood tests or other examinations.
                - Share your complete blood-test report with your doctor.
                - Avoid taking medicines or supplements without appropriate medical advice.
                """
            )

        with col2:
            st.markdown(
                """
                **🏃 General Healthy Habits:**

                - Stay physically active according to your ability.
                - Maintain a balanced and nutritious diet.
                - Avoid excessive alcohol consumption.
                - Maintain adequate hydration unless a healthcare professional has advised otherwise.
                """

            )

    # ==========================================
    # MODERATE PREDICTED RISK GUIDANCE
    # ==========================================

    elif disease_prob >= 30.0:

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                **🩺 What you can consider doing:**

                - Consider discussing this result with a healthcare professional.
                - Keep your blood-test reports available for comparison during future check-ups.
                - Follow up with a healthcare professional if you have symptoms or concerns.
                - Avoid unnecessary self-medication.
                """
            )

        with col2:
            st.markdown(
                """
                **🏃 General Healthy Habits:**

                - Maintain regular physical activity.
                - Follow a balanced diet.
                - Avoid excessive alcohol consumption.
                - Maintain a healthy body weight where appropriate.
                """
            )

    # ==========================================
    # LOWER PREDICTED RISK GUIDANCE
    # ==========================================

    else:

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                **🩺 General Health Guidance:**

                - Continue regular health check-ups.
                - Follow recommendations provided by your healthcare professional.
                - Keep your health and blood-test records for future reference.
                - Contact a healthcare professional if you develop symptoms or concerns.
                """
            )

        with col2:
            st.markdown(
                """
                **🏃 General Healthy Habits:**

                - Stay physically active.
                - Eat a balanced and nutritious diet.
                - Maintain a healthy body weight.
                - Stay adequately hydrated.
                """
            )

    st.caption(
        "⚠️ This application provides an ML-based prediction for "
        "educational and informational purposes. It is not a medical "
        "diagnosis and should not replace advice from a qualified "
        "healthcare professional."
    )


# ==========================================
# 4. MAIN APPLICATION
# ==========================================

def main():

    # ==========================================
    # HEADER
    # ==========================================

    st.title("🫁 LiverGuard Predict")

    st.markdown(
        """
        Enter the patient's information and blood-test values
        to estimate the predicted risk of liver disease.
        """
    )

    st.divider()

    # ==========================================
    # LOAD MODEL
    # ==========================================

    model, scaler = load_model_and_scaler()

    # ==========================================
    # INPUT FORM
    # ==========================================

    with st.form(key="liver_prediction_form"):

        # --------------------------------------
        # 1. Patient Information
        # --------------------------------------

        age, gender = render_patient_information()

        st.divider()

        # --------------------------------------
        # 2. Bilirubin
        # --------------------------------------

        total_bilirubin = render_bilirubin()

        st.divider()

        # --------------------------------------
        # 3. Liver Enzymes
        # --------------------------------------

        alt, ast = render_liver_enzymes()

        st.divider()

        # --------------------------------------
        # 4. Protein
        # --------------------------------------

        total_proteins = render_protein_level()

        st.divider()

        # --------------------------------------
        # PREDICT BUTTON
        # --------------------------------------

        submit_button = st.form_submit_button(
            label="🔍 Check Liver Disease Risk",
            use_container_width=True
        )

    # ==========================================
    # PREDICTION
    # ==========================================

    if submit_button:

        if model is None or scaler is None:

            st.error(
                "Prediction cannot be performed because "
                "the model or scaler is missing."
            )

            return

        # ==========================================
        # 1. ENCODE GENDER
        # ==========================================

        # Same encoding used during model training:
        # Male = 1
        # Female = 0

        gender_encoded = 1 if gender == "Male" else 0

        # ==========================================
        # 2. CREATE MODEL INPUT
        # ==========================================

        # IMPORTANT:
        # These names must remain exactly the same
        # as the training code.

        input_df = pd.DataFrame(
            [{
                "Age": age,
                "Gender": gender_encoded,
                "Total_Bilirubin": total_bilirubin,
                "Alamine_Aminotransferase": alt,
                "Aspartate_Aminotransferase": ast,
                "Total_Protiens": total_proteins
            }]
        )

        # ==========================================
        # 3. SCALE INPUT
        # ==========================================

        scaled_features = scaler.transform(input_df)

        # ==========================================
        # 4. MAKE PREDICTION
        # ==========================================

        prediction_probabilities = model.predict_proba(
            scaled_features
        )[0]

        # Class 1 = Liver Disease

        disease_prob = prediction_probabilities[1] * 100

        # ==========================================
        # 5. DISPLAY RESULT
        # ==========================================

        st.divider()

        st.subheader("📊 Prediction Details")

        # ------------------------------------------
        # Estimated Risk
        # ------------------------------------------

        st.metric(
            label="Estimated Liver Disease Risk",
            value=f"{disease_prob:.1f}%"
        )

        # ------------------------------------------
        # Risk Category
        # ------------------------------------------

        if disease_prob >= 70.0:

            st.error(
                "🔴 **Higher Predicted Risk**\n\n"
                "The model estimates a higher probability "
                "of liver disease."
            )

        elif disease_prob >= 30.0:

            st.warning(
                "🟠 **Moderate Predicted Risk**\n\n"
                "The model estimates a moderate probability "
                "of liver disease."
            )

        else:

            st.success(
                "🟢 **Lower Predicted Risk**\n\n"
                "The model estimates a lower probability "
                "of liver disease."
            )

        # ==========================================
        # 6. HEALTH GUIDANCE
        # ==========================================

        st.divider()

        render_health_guidance(disease_prob)


# ==========================================
# 5. RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    main()