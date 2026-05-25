import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np

# Adjusted to match your directory structure: pareto-prot/model/mint/mint.py
from model.mint.mint import MINTScorer, MiniDataset


class TestMINTScorer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up paths relative to pareto-prot root and create mock test sequences."""
        # Adjusted paths to look into your model folder and the sibling mint directory
        cls.config_path = "../mint/data/esm2_t33_650M_UR50D.json"
        cls.weights_path = "model/mint/mint.ckpt"
        cls.regressor_path = "model/mint/SKEMPI_v2.joblib"
        
        # Standardized generalized test pairs
        cls.mock_targets = {
            "Target_1": "GEMQKIVFKIPMVDDKSRTKAMSLVASTVGVHSVAIAGDLRDQVVVVGDGIDSINLVSALRKKVGPAMFLEVSQVKED",
            "Target_2": "GLKQKIVIKVAMEGNNCRSKAMALVASTGGVDSVALVGDLRDKIEVVGYGIDPIKLISALRKKVGDAELLQVSQANKD"
        }
        cls.wt_evolving_seq = "ETGNKYIEKRAIDLSRERDPNFFDNADIPVPECFWFMFKNNVRQDAGTCYSSWKMDMKVGPNWVHIKSDDNCNLSGDFPPGWIVLGKKRPGF"
        
        cls.mutant_seqs = [
            "ETGNKYIEKRAIDLSRERDPNFFDNADIPVPECFWFMFKNNVRQDAGTCYSSWKMDKKVGPNWVHIKSDDNCNLSGDFPPGWIVLGKKRPGF",
            "ETGNKYIEKRAIDLSRERDPNFFDNPGIPVPECFWFMFKNNVRQDDGTCYSSWKMDMKVGPNWVHIKSDDNCNLSGDFPPGWIVLGKKRPGF"
        ]

    @patch('joblib.load')
    def test_mint_scorer_pipeline(self, mock_joblib_load):
        """Test the generalized workflow: initialization, target caching, and mutant scoring."""
        # Mock the downstream regressor model
        mock_regressor = MagicMock()
        mock_regressor.predict.return_value = np.array([0.065, 0.067])
        mock_joblib_load.return_value = mock_regressor

        # Initialize the Scorer
        try:
            scorer = MINTScorer(
                mint_config_path=self.config_path,
                mint_weights_path=self.weights_path,
                regressor_path=self.regressor_path
            )
        except Exception as e:
            self.fail(f"MINTScorer initialization failed from the current layout: {e}")

        # Test baseline static target caching
        try:
            scorer.set_wt_baselines(self.mock_targets, self.wt_evolving_seq, batch_size=2)
        except Exception as e:
            self.fail(f"set_wt_baselines raised an exception unexpectedly: {e}")

        # Verify target caching keys and dimension constraints
        for target_name in self.mock_targets.keys():
            self.assertIn(target_name, scorer.wt_cache)
            self.assertEqual(scorer.wt_cache[target_name].shape[-1], 1280)

        # Test mutant scoring execution
        batch_targets = ["Target_1", "Target_2"]
        
        try:
            scores = scorer.score_mutants(
                target_names=batch_targets,
                mutant_seqs=self.mutant_seqs,
                target_dict=self.mock_targets,
                batch_size=2
            )
        except Exception as e:
            self.fail(f"score_mutants raised an exception unexpectedly: {e}")

        # Basic type and structure assertions
        self.assertIsInstance(scores, list)
        self.assertEqual(len(scores), len(self.mutant_seqs))
        self.assertIsInstance(scores[0], float)

        # Verify feature vector properties before regressor step
        self.assertTrue(mock_regressor.predict.called)
        called_features = mock_regressor.predict.call_args[0][0]
        self.assertEqual(called_features.shape, (2, 1280))

    def test_input_validation(self):
        """Ensure mismatch lengths between target names and mutant sequences trigger exceptions."""
        with patch('joblib.load'), patch('model.mint.mint.MINTWrapper'):
            scorer = MINTScorer(self.config_path, self.weights_path, self.regressor_path)
            
            with self.assertRaises(ValueError):
                scorer.score_mutants(
                    target_names=["Target_1"], 
                    mutant_seqs=self.mutant_seqs, 
                    target_dict=self.mock_targets
                )


if __name__ == '__main__':
    unittest.main()