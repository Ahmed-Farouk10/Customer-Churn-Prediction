# src/app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from typing import Dict, List, Tuple, Optional

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set gloomy theme
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.markdown("""
    <style>
    body { background-color: #1C2526; color: #D3D3D3; }
    .stApp { background-color: #1C2526; }
    .stButton>button { background-color: #2E3B3E; color: #D3D3D3; border: 1px solid #4A5A5C; }
    .stTextInput>div>input, .stNumberInput>div>input, .stSelectbox>div>select { background-color: #2E3B3E; color: #D3D3D3; border: 1px solid #4A5A5C; }
    h1, h2, h3, h4, h5, h6 { color: #A9A9A9; }
    .stAlert { background-color: #2E3B3E; color: #D3D3D3; }
    </style>
""", unsafe_allow_html=True)

# Get the parent directory (project root)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Feature configuration
FEATURE_CONFIG = {
    # Original dataset order
    'original_features': [
        'Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls',
        'Payment Delay', 'Subscription Type', 'Contract Length',
        'Total Spend', 'Last Interaction'
    ],
    'derived_features': [
        'Support_Calls_per_Tenure', 'Payment_Delay_to_Spend'
    ],
    'feature_ranges': {
        'Age': (18, 100),
        'Tenure': (0, 120),
        'Usage Frequency': (0, 100),
        'Support Calls': (0, 50),
        'Payment Delay': (0, 90),
        'Total Spend': (0.0, 10000.0),
        'Last Interaction': (0, 90)
    },
    'categorical_options': {
        'Gender': ['Male', 'Female'],
        'Subscription Type': ['Basic', 'Premium', 'Standard'],
        'Contract Length': ['Monthly', 'Quarterly', 'Annual']
    }
}

def validate_inputs(input_data: Dict) -> Tuple[bool, Optional[str]]:
    """Validate input data against expected ranges and types."""
    try:
        # Validate numerical features
        for feature, (min_val, max_val) in FEATURE_CONFIG['feature_ranges'].items():
            value = input_data[feature]
            if not isinstance(value, (int, float)):
                return False, f"{feature} must be a number"
            if value < min_val or value > max_val:
                return False, f"{feature} must be between {min_val} and {max_val}"
        
        # Validate categorical features
        for feature, options in FEATURE_CONFIG['categorical_options'].items():
            value = input_data[feature]
            if value not in options:
                return False, f"{feature} must be one of {options}"
        
        return True, None
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def create_derived_features(input_data: Dict) -> Dict:
    """Create derived features from input data."""
    try:
        # Calculate derived features
        input_data['Support_Calls_per_Tenure'] = (
            input_data['Support Calls'] / max(input_data['Tenure'], 1e-6)
        )
        input_data['Payment_Delay_to_Spend'] = (
            input_data['Payment Delay'] / max(input_data['Total Spend'], 1e-6)
        )
        return input_data
    except Exception as e:
        st.error(f"Error creating derived features: {str(e)}")
        return None

def create_input_dataframe(input_data: Dict) -> Optional[pd.DataFrame]:
    """Create a DataFrame from input data with correct feature order."""
    try:
        # Debug: Print input data keys
        st.write("Input data keys:", list(input_data.keys()))
        
        # Use the original dataset order for all features
        all_features = FEATURE_CONFIG['original_features'] + FEATURE_CONFIG['derived_features']
        
        # Debug: Print feature order
        st.write("Feature order being used:", all_features)
        
        # Create DataFrame with features in the correct order
        df = pd.DataFrame([input_data])[all_features]
        
        # Debug: Print final DataFrame columns
        st.write("Final DataFrame columns:", list(df.columns))
        
        return df
    except Exception as e:
        st.error(f"Error creating input DataFrame: {str(e)}")
        return None

# Load the model
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(parent_dir, 'models', 'ensemble_model.pkl')
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            # Debug: Print model feature names if available
            if hasattr(model, 'feature_names_in_'):
                st.write("Model feature names:", model.feature_names_in_)
            return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load the scaler and label encoders
@st.cache_resource
def load_preprocessing_objects():
    try:
        scaler_path = os.path.join(parent_dir, 'data', 'scaler.pkl')
        label_encoders_path = os.path.join(parent_dir, 'data', 'label_encoders.pkl')

        if not os.path.exists(scaler_path):
            st.error(f"Scaler file not found at: {scaler_path}")
            return None, None
        if not os.path.exists(label_encoders_path):
            st.error(f"Label encoders file not found at: {label_encoders_path}")
            return None, None

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            # Debug: Print scaler feature names if available
            if hasattr(scaler, 'feature_names_in_'):
                st.write("Scaler feature names:", scaler.feature_names_in_)
        with open(label_encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)
        return scaler, label_encoders
    except Exception as e:
        st.error(f"Error loading preprocessing files: {str(e)}")
        return None, None

# Function to preprocess input data
def preprocess_input(data: pd.DataFrame, scaler, label_encoders) -> Optional[pd.DataFrame]:
    """Preprocess input data using the same steps as training."""
    if scaler is None or label_encoders is None:
        st.error("Preprocessing objects not loaded.")
        return None

    try:
        # Create a copy to avoid modifying original data
        data_processed = data.copy()

        # Debug: Print columns before scaling
        st.write("Columns before scaling:", list(data_processed.columns))

        # Scale all numerical features (original + derived)
        numerical_features = [col for col in data_processed.columns if col not in FEATURE_CONFIG['categorical_options'].keys()]
        # Ensure columns are numeric before scaling
        for col in numerical_features:
             data_processed[col] = pd.to_numeric(data_processed[col], errors='coerce')
             # Handle potential NaNs introduced by coercion - replace with mean/median from training data if possible
             # For simplicity, replacing NaNs with 0 for now. A better approach: Impute with mean/median from training data.
             data_processed[col].fillna(0, inplace=True) # Replace NaN with 0

        data_processed[numerical_features] = scaler.transform(data_processed[numerical_features])

        # Debug: Print columns after scaling
        st.write("Columns after scaling:", list(data_processed.columns))

        # Encode categorical features
        for col in FEATURE_CONFIG['categorical_options'].keys():
            if col in label_encoders:
                # Ensure the value exists in the label encoder classes before transforming
                # Convert the column to string type to avoid issues with mixed types
                data_processed[col] = data_processed[col].astype(str)

                # Check if all unique values in the column are in the label encoder's classes
                unique_values = data_processed[col].unique()
                if all(item in label_encoders[col].classes_ for item in unique_values):
                     data_processed[col] = label_encoders[col].transform(data_processed[col])
                else:
                     unknown_values = [item for item in unique_values if item not in label_encoders[col].classes_]
                     st.error(f"Unknown category/values {unknown_values} for feature '{col}' during encoding. Please check input data and preprocessing.")
                     return None # Stop processing if an unknown category is found
            else:
                st.error(f"Label encoder not found for {col}")
                return None

        # Ensure final DataFrame has the correct column order explicitly
        all_features = FEATURE_CONFIG['original_features'] + FEATURE_CONFIG['derived_features']
        processed_data_final = data_processed[all_features]

        # Debug: Print final columns
        st.write("Final processed columns:", list(processed_data_final.columns))

        return processed_data_final
    except Exception as e:
        st.error(f"Error during preprocessing: {str(e)}")
        return None

# Streamlit app
st.title("Customer Churn Prediction")
st.markdown("Enter customer details to predict churn likelihood with a gloomy twist.")

# Create the form
form = st.form(key="customer_form")

# Input form
with form:
    col1, col2 = st.columns(2)

    with col1:
        # Numerical inputs with validation
        age = st.number_input(
            "Age",
            min_value=FEATURE_CONFIG['feature_ranges']['Age'][0],
            max_value=FEATURE_CONFIG['feature_ranges']['Age'][1],
            value=30
        )
        tenure = st.number_input(
            "Tenure (months)",
            min_value=FEATURE_CONFIG['feature_ranges']['Tenure'][0],
            max_value=FEATURE_CONFIG['feature_ranges']['Tenure'][1],
            value=0
        )
        usage_frequency = st.number_input(
            "Usage Frequency (times/month)",
            min_value=FEATURE_CONFIG['feature_ranges']['Usage Frequency'][0],
            max_value=FEATURE_CONFIG['feature_ranges']['Usage Frequency'][1],
            value=0
        )
        support_calls = st.number_input(
            "Support Calls",
            min_value=FEATURE_CONFIG['feature_ranges']['Support Calls'][0],
            max_value=FEATURE_CONFIG['feature_ranges']['Support Calls'][1],
            value=0
        )
        payment_delay = st.number_input(
            "Payment Delay (days)",
            min_value=FEATURE_CONFIG['feature_ranges']['Payment Delay'][0],
            max_value=FEATURE_CONFIG['feature_ranges']['Payment Delay'][1],
            value=0
        )

    with col2:
        # Categorical inputs
        gender = st.selectbox("Gender", FEATURE_CONFIG['categorical_options']['Gender'])
        subscription_type = st.selectbox(
            "Subscription Type",
            FEATURE_CONFIG['categorical_options']['Subscription Type']
        )
        contract_length = st.selectbox(
            "Contract Length",
            FEATURE_CONFIG['categorical_options']['Contract Length']
        )
        total_spend = st.number_input(
            "Total Spend",
            min_value=float(FEATURE_CONFIG['feature_ranges']['Total Spend'][0]),
            max_value=float(FEATURE_CONFIG['feature_ranges']['Total Spend'][1]),
            value=0.0,
            step=0.01
        )
        last_interaction = st.number_input(
            "Days Since Last Interaction",
            min_value=FEATURE_CONFIG['feature_ranges']['Last Interaction'][0],
            max_value=FEATURE_CONFIG['feature_ranges']['Last Interaction'][1],
            value=0
        )

    # Submit button
    submitted = st.form_submit_button("Predict Churn")

# Process form submission
if submitted:
    # Collect input data
    input_data = {
        'Age': age,
        'Gender': gender,
        'Tenure': tenure,
        'Usage Frequency': usage_frequency,
        'Support Calls': support_calls,
        'Payment Delay': payment_delay,
        'Subscription Type': subscription_type,
        'Contract Length': contract_length,
        'Total Spend': total_spend,
        'Last Interaction': last_interaction
    }

    # Validate inputs
    is_valid, error_message = validate_inputs(input_data)
    if not is_valid:
        st.error(error_message)
    else:
        # Create derived features
        input_data = create_derived_features(input_data)
        if input_data is None:
            st.error("Failed to create derived features")
        else:
            # Create input DataFrame
            input_df = create_input_dataframe(input_data)
            if input_df is None:
                st.error("Failed to create input DataFrame")
            else:
                # Load preprocessing objects
                scaler, label_encoders = load_preprocessing_objects()
                if scaler is not None and label_encoders is not None:
                    # Preprocess input data
                    processed_data = preprocess_input(input_df, scaler, label_encoders)
                    if processed_data is not None:
                        # Load model and make prediction
                        model = load_model()
                        if model is not None:
                            try:
                                prediction = int(model.predict(processed_data)[0]) # Explicitly cast to int

                                # Debug: Print the output of predict_proba
                                proba_output = model.predict_proba(processed_data)
                                st.write("Predict_proba output:", proba_output)
                                st.write("Predict_proba output type:", type(proba_output))
                                st.write("Predict_proba output shape:", proba_output.shape)

                                # Debug: Print the first element of predict_proba output
                                if isinstance(proba_output, np.ndarray) and proba_output.shape[0] > 0:
                                     st.write("First element of predict_proba output:", proba_output[0])
                                     st.write("Type of first element:", type(proba_output[0]))
                                     if isinstance(proba_output[0], (list, np.ndarray)) and len(proba_output[0]) > 1:
                                          st.write("Second element of the first element:", proba_output[0][1])
                                     else:
                                          st.error("First element of predict_proba output is not a subscriptable sequence of sufficient length.")
                                else:
                                     st.error("Predict_proba output is not a valid numpy array with at least one sample.")

                                # Check the structure before accessing elements
                                if isinstance(proba_output, np.ndarray) and proba_output.ndim == 2 and proba_output.shape[0] > 0 and proba_output.shape[1] > 1:
                                    probability = proba_output[0][1]
                                    st.write("Probability extracted:", probability)
                                else:
                                     st.error("Unexpected predict_proba output structure for direct access.")
                                     probability = 0.5 # Default to 0.5 if structure is unexpected

                                # Display prediction
                                st.subheader("Prediction Result")
                                if prediction == 1:
                                    st.error(f"Customer is likely to churn with {probability*100:.2f}% probability.")
                                else:
                                    st.success(f"Customer is unlikely to churn with {1-probability*100:.2f}% probability.")

                                # Feature importance visualization
                                if hasattr(model, 'estimators_'):
                                    base_model = next((est[1] for est in model.estimators_ if est[0] == 'xgboost'), None)
                                    if base_model and hasattr(base_model, 'feature_importances_'):
                                        feature_importances = base_model.feature_importances_
                                        feature_names = processed_data.columns
                                        if len(feature_importances) == len(feature_names):
                                            feature_importance_df = pd.DataFrame({
                                                'Feature': feature_names,
                                                'Importance': feature_importances
                                            }).sort_values(by='Importance', ascending=False)

                                            fig, ax = plt.subplots(figsize=(10, 6))
                                            sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax, palette="Blues_r")
                                            ax.set_title("Feature Importance (XGBoost)", color="#A9A9A9")
                                            ax.set_xlabel("Importance", color="#D3D3D3")
                                            ax.set_ylabel("Feature", color="#D3D3D3")
                                            ax.tick_params(colors="#D3D3D3")
                                            ax.set_facecolor("#1C2526")
                                            fig.set_facecolor("#1C2526")
                                            st.pyplot(fig)
                                        else:
                                            st.warning("Feature importance mismatch.")
                                    else:
                                        st.warning("XGBoost model not found for feature importance.")
                            except Exception as e:
                                st.error(f"Prediction error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("By Ahmed for Customer Churn Prediction.")