import streamlit as st
import pandas as pd
import numpy as np
import pickle # Changed from joblib to pickle

# --- Streamlit App Layout (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Heart Disease Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💖 Heart Disease Prediction System")
st.markdown("---")

# --- Load the Trained Model and Preprocessors ---
# Ensure these files are in the same directory as your app.py
try:
    with open('gaussian_naive_bayes_model.pkl', 'rb') as file: # Changed to pickle and .pkl
        loaded_model = pickle.load(file)
    with open('minmax_scaler.pkl', 'rb') as file: # Changed to pickle and .pkl
        loaded_scaler = pickle.load(file)
    with open('mi_feature_selector.pkl', 'rb') as file: # Changed to pickle and .pkl
        loaded_selector = pickle.load(file)
    with open('full_encoded_columns.pkl', 'rb') as file: # Changed to pickle and .pkl
        full_encoded_columns = pickle.load(file)
    with open('scaler_fit_columns.pkl', 'rb') as file: # Changed to pickle and .pkl
        scaler_fit_columns = pickle.load(file)
    with open('selector_input_columns.pkl', 'rb') as file: # Changed to pickle and .pkl
        selector_input_columns = pickle.load(file)

    st.sidebar.success("Model, scaler, selector, and column lists loaded successfully!")
except FileNotFoundError:
    st.sidebar.error("Error: Model files not found. Please run 'train_and_save_model.py' first to generate them.")
    st.stop() # Stop the app if model files are not found

# --- Feature Mapping for User Input (Consistent with train_and_save_model.py) ---
# These mappings ensure the user's input is correctly transformed into the format
# expected by the trained model (specifically for one-hot encoding).

# For 'cp' (Chest pain type) - mapping to original integer values
CP_MAPPING = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

# For 'restecg' (Resting electrocardiographic results) - mapping to original integer values
RESTECG_MAPPING = {
    "Normal": 0,
    "ST-T wave abnormality": 1,
    "Left ventricular hypertrophy": 2
}

# For 'slope' (Slope of the peak exercise ST segment) - mapping to original integer values
SLOPE_MAPPING = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

# For 'thal' (Thalassemia) - mapping to original integer values
THAL_MAPPING = {
    "Normal": 0,
    "Fixed Defect": 1,
    "Reversible Defect": 2
}

# For 'sex' (Sex of the patient) - binary
SEX_OPTIONS = {
    "Male": 1,
    "Female": 0
}

# For 'fbs' (Fasting blood sugar > 120 mg/dl) - binary
FBS_OPTIONS = {
    "True": 1,
    "False": 0
}

# For 'exang' (Exercise-induced angina) - binary
EXANG_OPTIONS = {
    "Yes": 1,
    "No": 0
}

# Define the list of categorical columns that were one-hot encoded during training
CATEGORICAL_COLS_TO_ENCODE = ['cp', 'restecg', 'slope', 'thal']


# --- Prediction Function (Using Loaded Model) ---
def predict_heart_disease(features_dict):
    """
    Makes a prediction using the loaded preprocessors and trained model.
    Applies the same preprocessing steps as during training.
    """
    # 1. Create an empty DataFrame with all expected columns from the training data (after OHE), initialized to zeros.
    processed_input_df = pd.DataFrame(0, index=[0], columns=full_encoded_columns)

    # 2. Populate numerical features directly
    processed_input_df['age'] = features_dict['age']
    processed_input_df['sex'] = features_dict['sex']
    processed_input_df['trestbps'] = features_dict['trestbps']
    processed_input_df['chol'] = features_dict['chol']
    processed_input_df['fbs'] = features_dict['fbs']
    processed_input_df['thalach'] = features_dict['thalach']
    processed_input_df['exang'] = features_dict['exang']
    processed_input_df['oldpeak'] = features_dict['oldpeak']
    processed_input_df['ca'] = features_dict['ca']

    # 3. Populate one-hot encoded categorical features
    processed_input_df[f'cp_{features_dict["cp"]}'] = 1
    processed_input_df[f'restecg_{features_dict["restecg"]}'] = 1
    processed_input_df[f'slope_{features_dict["slope"]}'] = 1
    processed_input_df[f'thal_{features_dict["thal"]}'] = 1

    # 4. Apply scaling using the loaded scaler.
    scaled_input_array = loaded_scaler.transform(processed_input_df[scaler_fit_columns])
    
    # 5. Convert the scaled NumPy array back to a DataFrame,
    scaled_input_df = pd.DataFrame(scaled_input_array, columns=selector_input_columns)

    # 6. Apply feature selection using the loaded selector.
    selected_input = loaded_selector.transform(scaled_input_df)

    # Make prediction
    prediction = loaded_model.predict(selected_input)[0]
    prediction_proba = loaded_model.predict_proba(selected_input)[0][1] # Probability of positive class

    return prediction, prediction_proba

# --- Prediction System UI ---
st.header("Predict Heart Disease Risk")
st.markdown("Enter the patient's details below to get a heart disease prediction.")
st.info("This prediction uses the Gaussian Naïve Bayes model trained with Mutual Information feature selection, as detailed in the project report. It is for informational purposes and should not replace professional medical advice.")

# Input form for features
with st.form("prediction_form"):
    st.subheader("Patient Information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 100, 50)
        sex_input_str = st.radio("Sex", list(SEX_OPTIONS.keys()))
        sex = SEX_OPTIONS[sex_input_str] # Get numerical value

        cp_input_str = st.selectbox("Chest Pain Type (cp)", list(CP_MAPPING.keys()))
        cp = CP_MAPPING[cp_input_str] # Get numerical value for processing

        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)

    with col2:
        fbs_input_str = st.radio("Fasting Blood Sugar > 120 mg/dl", list(FBS_OPTIONS.keys()))
        fbs = FBS_OPTIONS[fbs_input_str] # Get numerical value

        restecg_input_str = st.selectbox("Resting Electrocardiographic Results", list(RESTECG_MAPPING.keys()))
        restecg = RESTECG_MAPPING[restecg_input_str] # Get numerical value

        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exang_input_str = st.radio("Exercise-Induced Angina", list(EXANG_OPTIONS.keys()))
        exang = EXANG_OPTIONS[exang_input_str] # Get numerical value

        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.0, step=0.1)

        slope_input_str = st.selectbox("Slope of the Peak Exercise ST Segment", list(SLOPE_MAPPING.keys()))
        slope = SLOPE_MAPPING[slope_input_str] # Get numerical value

        ca = st.slider("Number of Major Vessels Colored by Fluoroscopy (0-3)", 0, 3, 0)

        thal_input_str = st.selectbox("Thalassemia", list(THAL_MAPPING.keys()))
        thal = THAL_MAPPING[thal_input_str] # Get numerical value

    submitted = st.form_submit_button("Get Prediction")

    if submitted:
        # Prepare features for prediction
        input_features = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }

        # Get prediction from the loaded model
        prediction, prediction_proba = predict_heart_disease(input_features)

        st.subheader("Prediction Result:")
        if prediction == 0: # 0 means No Heart Disease
            st.success(f"**Prediction: No Heart Disease**")
            st.info(f"The model predicts a low probability of heart disease ({prediction_proba:.2f}).")
        else: # 1 means Heart Disease Present
            st.error(f"**Prediction: Heart Disease Present**")
            st.warning(f"The model predicts a high probability of heart disease ({prediction_proba:.2f}). Please consult a medical professional for accurate diagnosis.")

        st.markdown("---")
        st.write("Input Features Provided:")
        st.json(input_features

