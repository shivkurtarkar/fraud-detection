import unittest
import json
from api.api import app  # Assuming your Flask app is in a file named 'app.py'

class TestAPI(unittest.TestCase):
    
    # Set up a test client for the Flask application
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True  # Enable testing mode
        
    def test_predict_success(self):
        # Create a valid test data payload
        valid_data = {
            "cc_num": 1234567890,
            "amt": 100.5,
            "zip": 94110,
            "lat": 37.7749,
            "long": -122.4194,
            "city_pop": 870000,
            "merch_lat": 37.8044,
            "merch_long": -122.2711
        }
        
        # Send a POST request to the '/predict' endpoint with the valid data
        response = self.client.post('/predict', 
                                    data=json.dumps(valid_data), 
                                    content_type='application/json')
        
        # Assert that the response status code is 200 (success)
        self.assertEqual(response.status_code, 200)
        
        # Assert that the response contains the prediction result
        response_data = json.loads(response.data)
        self.assertIn('prediction', response_data)
    
    def test_predict_missing_feature(self):
        # Create invalid data (missing the 'cc_num' feature)
        invalid_data = {
            "amt": 100.5,
            "zip": 94110,
            "lat": 37.7749,
            "long": -122.4194,
            "city_pop": 870000,
            "merch_lat": 37.8044,
            "merch_long": -122.2711
        }
        
        # Send a POST request with the invalid data
        response = self.client.post('/predict', 
                                    data=json.dumps(invalid_data), 
                                    content_type='application/json')
        
        # Assert that the response status code is 400 (bad request)
        self.assertEqual(response.status_code, 400)
        
        # Assert that the error message mentions the missing feature
        response_data = json.loads(response.data)
        self.assertIn('Missing features', response_data['error'])
    
    def test_health_check(self):
        # Send a GET request to the '/health' endpoint
        response = self.client.get('/health')
        
        # Assert that the response status code is 200 (OK)
        self.assertEqual(response.status_code, 200)
        
        # Assert that the response contains the 'status' key with 'healthy' value
        response_data = json.loads(response.data)
        self.assertEqual(response_data['status'], 'healthy')

if __name__ == '__main__':
    unittest.main()
