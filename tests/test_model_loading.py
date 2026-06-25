import unittest
import os
import joblib

class TestModelLoading(unittest.TestCase):
    def test_load_latest_model(self):
        # Look for the latest RF model in models/
        model_path = 'models/detection_random_forest_latest.joblib'
        if not os.path.exists(model_path):
            # Fallback to experiment 053 if running in specific directory structure
            model_path = 'experiments/experiment_053/model.joblib'
            
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            self.assertIsNotNone(model)
            # Verify it has standard scikit-learn random forest properties or methods
            self.assertTrue(hasattr(model, 'predict'))
            self.assertTrue(hasattr(model, 'predict_proba'))
        else:
            self.skipTest("No pre-trained model checkpoint found for loading validation.")

if __name__ == '__main__':
    unittest.main()
