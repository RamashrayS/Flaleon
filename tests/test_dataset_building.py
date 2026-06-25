import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from src.preprocessing.dataset_builder import build_dataset

class TestDatasetBuilding(unittest.TestCase):
    @patch('src.preprocessing.dataset_builder.load_solexs_day')
    @patch('src.preprocessing.dataset_builder.load_helios_day')
    @patch('src.preprocessing.dataset_builder.os.path.exists')
    def test_build_dataset(self, mock_exists, mock_load_helios, mock_load_solexs):
        mock_exists.return_value = True
        
        # Mock load outputs
        times = np.arange(1777000000.0, 1777000000.0 + 10.0)
        df_sol = pd.DataFrame({
            'TIME': times,
            'COUNTS': np.random.normal(500, 50, len(times))
        })
        df_hel = pd.DataFrame({
            'TIME': times + 0.1,
            'COUNTS': np.random.normal(100, 10, len(times))
        })
        
        mock_load_solexs.return_value = df_sol
        mock_load_helios.return_value = df_hel
        
        df_final, metadata = build_dataset(['2024-02-16'])
        
        self.assertIsNotNone(df_final)
        self.assertIn('solexs_counts', df_final.columns)
        self.assertIn('helios_counts', df_final.columns)
        self.assertIn('row_count', metadata)
        self.assertEqual(metadata['row_count'], len(df_final))

if __name__ == '__main__':
    unittest.main()
