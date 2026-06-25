import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import os
from src.preprocessing.dataset_builder import build_dataset

class TestCheckpointLoading(unittest.TestCase):
    @patch('src.preprocessing.dataset_builder.pd.read_csv')
    @patch('src.preprocessing.dataset_builder.os.path.exists')
    def test_checkpoint_reused(self, mock_exists, mock_read_csv):
        # Setup mock behavior: raw directory doesn't need to exist, but checkpoint file does
        def exists_side_effect(path):
            if "checkpoints" in path:
                return True
            return True
            
        mock_exists.side_effect = exists_side_effect
        
        # Mock pre-existing checkpoint dataframe
        times = np.arange(1777000000, 1777000000 + 5)
        df_checkpoint = pd.DataFrame({
            'TIME': times,
            'solexs_counts': np.array([100.0, 100.0, 100.0, 100.0, 100.0]),
            'helios_counts': np.array([10.0, 10.0, 10.0, 10.0, 10.0]),
            'flare_now': np.array([0, 0, 0, 0, 0]),
            'flare_class': np.array(['Quiet', 'Quiet', 'Quiet', 'Quiet', 'Quiet']),
            'flare_future': np.array([0, 0, 0, 0, 0])
        })
        mock_read_csv.return_value = df_checkpoint
        
        # Test with FORCE_RERUN set to False to reuse checkpoints
        with patch('src.utils.config.FORCE_RERUN', False):
            df_final, metadata = build_dataset(['2024-02-16'])
            
            # Verify read_csv was called (meaning checkpoint was loaded)
            mock_read_csv.assert_called()
            self.assertEqual(len(df_final), 5)

if __name__ == '__main__':
    unittest.main()
