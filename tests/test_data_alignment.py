import unittest
import pandas as pd
import numpy as np
from src.preprocessing.alignment import align_payloads

class TestDataAlignment(unittest.TestCase):
    def test_align_payloads(self):
        # Create dummy SoLEXS and HEL1OS dataframes
        solexs_times = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        helios_times = np.array([10.1, 11.2, 12.1, 13.3, 14.2])
        
        df_solexs = pd.DataFrame({
            'TIME': solexs_times,
            'COUNTS': np.array([100, 110, 120, 130, 140])
        })
        df_helios = pd.DataFrame({
            'TIME': helios_times,
            'COUNTS': np.array([10, 11, 12, 13, 14])
        })
        
        # Merge with time_offset=0.0 and tolerance=1.0
        df_merged = align_payloads(df_solexs, df_helios, time_offset=0.0, tolerance=1.0)
        
        # Verify alignment columns are correctly renamed and merged
        self.assertIn('solexs_counts', df_merged.columns)
        self.assertIn('helios_counts', df_merged.columns)
        self.assertEqual(len(df_merged), len(df_solexs))
        
        # Check backward merging (e.g. for TIME=11.0, helios TIME=10.1 is nearest past)
        # So helios_counts at TIME=11.0 should be 10.
        row_11 = df_merged[df_merged['TIME'] == 11.0].iloc[0]
        self.assertEqual(row_11['helios_counts'], 10.0)

if __name__ == '__main__':
    unittest.main()
