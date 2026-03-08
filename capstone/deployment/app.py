import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="RubeenaNouman/machine_failure_model", filename="best_machine_failure_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Machine Failure Prediction App")
st.write("""
This application predicts the likelihood of a machine failing based on its operational parameters.
Please enter the sensor and configuration data below to get a prediction.
""")

# User input
Engine_rpm = st.number_input("Engine rpm", min_value=61.0, max_value=2239.0, value=70.0, step=1)
lub_oil_press = st.number_input("Lub oil pressure", min_value=0.003384, max_value=7.265566, value=1.0, step=0.0001)
fuel_press = st.number_input("Fuel pressure", min_value=0.003187, max_value=21.389, value=1.0, step=0.0001)
coolant_press = st.number_input("Coolant pressure", min_value=0.002483, max_value=7.4785, value=1.0, step=0.0001)
lub_oil_temp = st.number_input("lub oil temp", min_value=71.3219, max_value=89.5808, value=10,step=0.0001)
coolant_temp = st.number_input("Coolant temp", min_value=61.6733, max_value=195.5279, value=10,step=0.0001)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Engine rpm': Engine_rpm,
    'Lub oil pressure': lub_oil_press,
    'Fuel pressure': fuel_press,
    'Coolant pressure': coolant_press,
    'lub oil temp': lub_oil_temp,
    'Coolant temp': coolant_temp
}])


if st.button("Predict Failure"):
    prediction = model.predict(input_data)[0]
    result = "Machine Failure" if prediction == 1 else "No Failure"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
