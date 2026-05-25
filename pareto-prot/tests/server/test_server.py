import unittest
from server.server import ScoringServer

class DummyModelEngine:
    """A fake MINTScorer to test the Server's routing logic."""
    def score_mutants(self, target_names, mutant_seqs, batch_size=4):
        # We will just return the length of the string as a fake "score"
        return [float(len(seq)) for seq in mutant_seqs]

class TestScoringServer(unittest.TestCase):
    
    def setUp(self):
        self.engine = DummyModelEngine()
        
        # Preload the server with one known sequence
        self.preloaded = {
            ("OsHIPP20", "KNOWN_SEQ"): 0.99
        }
        
        self.server = ScoringServer(
            model=self.engine, 
            server_name="test_mint",
            preloaded_cache=self.preloaded
        )

    def test_routing_and_cache_hits(self):
        """Ensure the server successfully mixes cached data with novel GPU predictions."""
        targets = ["OsHIPP20", "OsHIPP20", "Pikm-1"]
        variants = ["KNOWN_SEQ", "NEW_SEQ_1", "NEW_SEQ_22"] # Lengths: 9, 9, 10
        
        scores = self.server.request_phenotype_scores(targets, variants)
        
        # The expected output:
        # Index 0: 0.99 (Pulled from historic cache)
        # Index 1: 9.0 (Calculated: length of "NEW_SEQ_1")
        # Index 2: 10.0 (Calculated: length of "NEW_SEQ_22")
        self.assertEqual(scores, [0.99, 9.0, 10.0])
        
        # Telemetry check: 1 cache hit, 2 inferences
        self.assertEqual(self.server.cache_hits, 1)
        self.assertEqual(self.server.inference_calls, 2)

    def test_cache_extraction_for_db(self):
        """Ensure extracting the new cache moves data correctly without duplicating."""
        targets = ["OsHIPP20"]
        variants = ["NOVEL_SEQ"]
        
        # 1. Run inference
        self.server.request_phenotype_scores(targets, variants)
        
        # 2. Extract for DB
        to_flush = self.server.extract_new_cache_for_db()
        
        # It should contain the novel sequence
        self.assertIn(("OsHIPP20", "NOVEL_SEQ"), to_flush)
        
        # The new_cache should now be empty
        self.assertEqual(len(self.server.new_cache), 0)
        
        # The novel sequence should now live in the historic cache
        self.assertIn(("OsHIPP20", "NOVEL_SEQ"), self.server.historic_cache)
        
        # 3. Extract again (should be empty, preventing double-flushing)
        to_flush_again = self.server.extract_new_cache_for_db()
        self.assertEqual(len(to_flush_again), 0)

    def test_oom_protection(self):
        """Ensure the historic cache drops if memory limits are breached."""
        # Set an artificially tiny limit
        tiny_server = ScoringServer(self.engine, "test", self.preloaded, max_cache_size=2)
        
        targets = ["OsHIPP20", "OsHIPP20"]
        variants = ["NEW_1", "NEW_2"]
        
        tiny_server.request_phenotype_scores(targets, variants)
        
        # We had 1 historic + 2 new = 3 items. Limit was 2.
        # The historic cache should have been wiped.
        self.assertEqual(len(tiny_server.historic_cache), 0)
        
        # The new cache must survive so it can be flushed!
        self.assertEqual(len(tiny_server.new_cache), 2)

if __name__ == '__main__':
    unittest.main()