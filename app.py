import streamlit as st
import pandas as pd
import joblib

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load("gs_rf.pk1")  # Make sure the model file is in the same directory

model = load_model()

# Streamlit UI
st.title("Credit Card Fraud Detection")
st.write("Upload a CSV file to predict fraud cases.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview:")
    st.write(df.head())
    
    # Ensure the dataset has the correct features
    expected_features = model.feature_names_in_  # Extract features the model expects
    if all(feature in df.columns for feature in expected_features):
        # Make predictions
        predictions = model.predict(df[expected_features])
        df["Prediction"] = predictions
        
        st.write("### Predictions:")
        st.write(df[["Prediction"]].value_counts())
        
        # Allow users to download the predictions
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
    else:
        st.error("Uploaded CSV does not have the required features for prediction.")
