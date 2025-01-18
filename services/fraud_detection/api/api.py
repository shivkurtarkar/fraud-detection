import joblib
from flask import Flask, request, jsonify
import numpy as np
import os

# Initialize the Flask application
app = Flask(__name__)

# Load the model
OUTPUT_DIR='api/models'
MODEL_FILE = f"{OUTPUT_DIR}/best_model.pkl"

MODEL_FILE = os.environ.get("MODEL_FILE", f"{OUTPUT_DIR}/best_model.pkl")


# Load the trained model (replace with your actual model file path)
model = joblib.load(MODEL_FILE)

# Define the feature names
numerical_features = ['cc_num', 'amt', 'zip', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Ensure all required features are present in the data
        missing_features = [feature for feature in numerical_features if feature not in data]
        if missing_features:
            return jsonify({'error': f'Missing features: {", ".join(missing_features)}'}), 400

        # Prepare the input features for prediction (as a numpy array)
        input_features = np.array([[
            data['cc_num'],
            data['amt'],
            data['zip'],
            data['lat'],
            data['long'],
            data['city_pop'],
            data['merch_lat'],
            data['merch_long']
        ]])

        # Make the prediction using the model
        prediction = model.predict(input_features)

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction.tolist()}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# API endpoint to check if the service is running
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    # Run the app on port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)
