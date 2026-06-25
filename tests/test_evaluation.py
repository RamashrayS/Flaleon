import unittest
import numpy as np
from src.utils.metrics import evaluate_detection

class TestEvaluation(unittest.TestCase):
    def test_evaluate_detection_metrics(self):
        # Setup perfect prediction arrays
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.85, 0.15, 0.95])
        
        metrics = evaluate_detection(y_true, y_pred, y_prob)
        
        # Verify perfect performance scores
        self.assertAlmostEqual(metrics['accuracy'], 1.0)
        self.assertAlmostEqual(metrics['precision'], 1.0)
        self.assertAlmostEqual(metrics['recall'], 1.0)
        self.assertAlmostEqual(metrics['f1'], 1.0)
        self.assertAlmostEqual(metrics['mcc'], 1.0)
        
        # Setup imperfect prediction arrays
        y_pred_imp = np.array([0, 1, 1, 0, 0, 1])
        metrics_imp = evaluate_detection(y_true, y_pred_imp, y_prob)
        self.assertLess(metrics_imp['accuracy'], 1.0)

if __name__ == '__main__':
    unittest.main()
