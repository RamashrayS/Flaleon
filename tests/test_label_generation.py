import unittest
import pandas as pd
import numpy as np
from src.labeling.labeler import label_dataset

class TestLabelGeneration(unittest.TestCase):
    def test_label_dataset(self):
        # Create a mock dataframe
        times = np.array([1700000000, 1700000100, 1700000200, 1700000300])
        df = pd.DataFrame({
            'TIME': times,
            'solexs_counts': np.array([100.0, 2000.0, 100.0, 100.0])
        })
        
        # Test label_dataset runs
        df_labeled = label_dataset(df)
        self.assertIn('flare_now', df_labeled.columns)
        self.assertIn('flare_class', df_labeled.columns)
        self.assertIn('flare_future', df_labeled.columns)

if __name__ == '__main__':
    unittest.main()
