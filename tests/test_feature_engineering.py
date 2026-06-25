import unittest
import pandas as pd
import numpy as np
from src.features.engineering import add_physics_features

class TestFeatureEngineering(unittest.TestCase):
    def test_add_physics_features(self):
        # Create a dummy dataframe with aligned columns
        times = np.arange(1777000000, 1777000000 + 400)
        df = pd.DataFrame({
            'TIME': times,
            'solexs_counts': np.random.normal(500, 50, len(times)),
            'helios_counts': np.random.normal(100, 10, len(times)),
            'helios_czt_counts': np.random.normal(200, 20, len(times))
        })
        
        df_feat = add_physics_features(df)
        
        # Verify columns generated
        self.assertIn('solexs_dFlux_dt', df_feat.columns)
        self.assertIn('hardness_ratio', df_feat.columns)
        self.assertIn('solexs_counts_ema_30s', df_feat.columns)
        self.assertIn('solexs_counts_roll_mean_60s', df_feat.columns)
        self.assertIn('solexs_counts_lag_300s', df_feat.columns)
        
        # Verify no NaN values exist in the final output
        self.assertEqual(df_feat.isna().sum().sum(), 0)

if __name__ == '__main__':
    unittest.main()
