
import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import io  # Import io for capturing df.info()

# Set page configuration (must be the first Streamlit command)
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")


# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load("gs_rf.pk1")  # Ensure the file exists in the directory

model = load_model()

# Streamlit UI
st.title("Credit Card Fraud Detection System")
st.write(
    """
    This is a demo for predicting credit card fraud using a machine learning model, the dataset used 
    for this demo was obtained from Kaggle. The features have been anonymized due to company data privacy 
    policies, recall is at 90% (i.e don't miss any fraud case). Email me at nosakhareasowata94@gmail.com for feedback/remarks 
    as i will be updating my code from the feedbacks i get, thanks in anticipation.
    """
)

st.write("### Download Test File (CSV Format) and make your predictions")
csv_url = "https://docs.google.com/spreadsheets/d/1K3jIpRLh-dwtmMPOD3RsN5k08IJdtRIW-CvH3ETxwt4/export?format=csv"
st.markdown(f"[Download CSV File]({csv_url})")

st.write("### Download Test File results (CSV Format) and compare with model predictions")
csv_url = "https://docs.google.com/spreadsheets/d/1ykMCpFcpVFINL_GPGU_No3t6id85DvLlob1-axWj7ys/export?format=csv"
st.markdown(f"[Download CSV File]({csv_url})")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file to predict fraud cases", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write("### Uploaded Data Preview:")
    st.dataframe(df.head())  # Display the uploaded data as a dataframe

    # Show data information (columns, types, etc.)
    st.write("### Data Information:")
    buffer = io.StringIO()  # Use StringIO to capture df.info() output
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    # Ensure the dataset has the correct features
    expected_features = model.feature_names_in_  # Extract features the model expects
    if all(feature in df.columns for feature in expected_features):
        # Make predictions
        predictions = model.predict(df[expected_features])
        
        # Add fraud prediction to dataframe with 1 for fraud, 0 for non-fraud
        df["Fraud_Prediction"] = predictions
        df["Fraud_Label"] = df["Fraud_Prediction"].apply(lambda x: "Fraud" if x == 1 else "Non-Fraud")

        st.write("### Predictions and Label Counts:")
        st.write(df["Fraud_Label"].value_counts())
        
        # Plot categorical comparisons
        st.write("### Comparison between Categorical Features:")
        categorical_features = df.select_dtypes(include=['object']).columns
        if len(categorical_features) > 0:
            for feature in categorical_features:
                plt.figure(figsize=(8, 5))
                sns.countplot(x=feature, hue="Fraud_Label", data=df)
                plt.title(f"Comparison of {feature} with Fraud Prediction")
                plt.xticks(rotation=45)
                st.pyplot(plt)
        
        # Allow users to download the predictions
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions with Labels", csv, "predictions_with_labels.csv", "text/csv")
    else:
        st.error("Uploaded CSV does not have the required features for prediction.")
