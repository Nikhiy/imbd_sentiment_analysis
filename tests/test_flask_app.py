import unittest
import os
import mlflow

# 🔐 Set DagsHub auth BEFORE importing flask_app
dagshub_token = os.getenv("CAPSTONE_TEST")
if dagshub_token:
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    mlflow.set_tracking_uri("https://dagshub.com/Nikhiy/imbd_sentiment_analysis.mlflow")

from flask_app.app import app   # import AFTER auth

class FlaskAppTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Sentiment Analysis</title>', response.data)

    def test_predict_endpoint(self):
        response = self.client.post('/predict', data=dict(text='This is a test.'))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            b'Positive' in response.data or b'Negative' in response.data
        )

if __name__ == '__main__':
    unittest.main()
