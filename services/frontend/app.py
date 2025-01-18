import streamlit as st
import requests
import json
import os

API_URL = os.environ.get("API_URL", "http://localhost:8000")


# Define the features list as per your model
numerical_features = ['cc_num', 'amt', 'zip', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']

# Streamlit UI setup
st.title('Credit Card Transaction Prediction')
st.markdown("""
    This application predicts whether a credit card transaction is valid based on user input.
    Fill in the fields below and click **Predict** to get the result.
""")

# Create input fields for each feature
cc_num = st.text_input("Credit Card Number (cc_num)", "")
amt = st.number_input("Transaction Amount (amt)", min_value=0.0, step=0.01)
zip_code = st.text_input("Zip Code (zip)", "")
lat = st.number_input("Latitude (lat)", min_value=-90.0, max_value=90.0, step=0.0001)
long = st.number_input("Longitude (long)", min_value=-180.0, max_value=180.0, step=0.0001)
city_pop = st.number_input("City Population (city_pop)", min_value=0)
merch_lat = st.number_input("Merchant Latitude (merch_lat)", min_value=-90.0, max_value=90.0, step=0.0001)
merch_long = st.number_input("Merchant Longitude (merch_long)", min_value=-180.0, max_value=180.0, step=0.0001)

# Function to make predictions
def make_prediction(input_data):
    try:
        # Send a POST request to the Flask API with the input data
        response = requests.post(f"{API_URL}/predict", json=input_data)
        
        # If request was successful
        if response.status_code == 200:
            result = response.json()
            return result.get("prediction", "No prediction result")
        else:
            st.error(f"Error: {response.json()['error']}")
            return None
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        return None

# When the user presses the 'Predict' button
if st.button("Predict"):
    # Prepare input data from the Streamlit input fields
    input_data = {
        "cc_num": int(cc_num) if cc_num else 0,
        "amt": amt,
        "zip": int(zip_code) if zip_code else 0,
        "lat": lat,
        "long": long,
        "city_pop": city_pop,
        "merch_lat": merch_lat,
        "merch_long": merch_long
    }
    
    # Make the prediction
    prediction = make_prediction(input_data)
    st.write(prediction)
    
    # # Display the result if prediction was successful
    # if prediction is not None:
    #     st.subheader(f"Prediction: {prediction[0]}")



    # Display the result if prediction was successful
    if prediction is not None:
        # Convert the prediction (0 or 1) to "Not Fraud" or "Fraud"
        if prediction[0] == 0:
            st.subheader("Prediction: Not Fraud")
        elif prediction[0] == 1:
            st.subheader("Prediction: Fraud")
        else:
            st.error("Invalid prediction result.")