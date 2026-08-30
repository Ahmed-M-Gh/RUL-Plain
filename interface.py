import streamlit as st 
import pandas as pd
from src.preprocessing_pipeline import preprocessing_engine_data
from src.config import MODEL

st.title("Aircraft Engine RUL Prediction")

# About section
st.markdown("""
            ### About this model
            ...
            """)


# file uploader 
uploaded_file = st.file_uploader("Upload engine sensor data (CSV)", type="csv")

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    
    if st.button("Predict RUL"):
        try:
            processed = preprocessing_engine_data(raw_df)
            model_input = processed.drop(columns=["unit", "cycle"])
            predictions = MODEL.predict(model_input)
            
            results_df = processed[["unit", "cycle"]].copy()
            results_df["predicted_RUL"] = predictions
            
            st.dataframe(results_df)
        except ValueError as e:
            st.error(f"Data issue: {e}")