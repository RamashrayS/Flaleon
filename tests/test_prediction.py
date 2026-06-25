import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class TestPrediction(unittest.TestCase):
    def test_predict_proba_format(self):
        # Create a dummy model and dummy input
        X_train = np.random.normal(0, 1, (100, 5))
        y_train = np.random.binomial(1, 0.3, 100)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        X_test = np.random.normal(0, 1, (20, 5))
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Verify shape and contents
        self.assertEqual(len(y_prob), 20)
        self.assertTrue(np.all(y_prob >= 0.0))
        self.assertTrue(np.all(y_prob <= 1.0))

if __name__ == '__main__':
    unittest.main()
