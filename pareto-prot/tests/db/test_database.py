import unittest
from db.manager import DatabaseManager
from db.models import Experiment, Simulation, Variant, Score

class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        # Use an in-memory SQLite database for pure, isolated testing
        self.db = DatabaseManager(db_path="sqlite:///:memory:")
        self.assertTrue(self.db.check_connection())

    def test_experiment_creation(self):
        # Create a new experiment
        exp_id_1 = self.db.get_or_create_experiment("Test_Exp_1", "Desc 1")
        self.assertIsNotNone(exp_id_1)
        
        # Calling it again should return the exact same ID
        exp_id_2 = self.db.get_or_create_experiment("Test_Exp_1")
        self.assertEqual(exp_id_1, exp_id_2)

    def test_simulation_lifecycle(self):
        exp_id = self.db.get_or_create_experiment("Lifecycle_Test")
        
        # Start
        sim_id = self.db.start_simulation(
            exp_id=exp_id, poi_name="AVR", wt_sequence="MADER",
            model_name="Mint", model_params={"weights": "path"},
            mutator_name="Protein", mutator_params={"rate": 0.01},
            objective_name="Escape", objective_params={"slope": 10.0},
            simulator_name="Greedy", simulator_params={"top_k": 5}
        )
        self.assertIsNotNone(sim_id)

        # Verify start timestamp exists, end timestamp is None
        with self.db.SessionLocal() as session:
            sim = session.query(Simulation).filter_by(id=sim_id).first()
            self.assertIsNotNone(sim.start_timestamp)
            self.assertIsNone(sim.end_timestamp)

        # Finish
        self.db.finish_simulation(sim_id)
        
        # Verify end timestamp populated
        with self.db.SessionLocal() as session:
            sim = session.query(Simulation).filter_by(id=sim_id).first()
            self.assertIsNotNone(sim.end_timestamp)

    def test_variant_streaming(self):
        exp_id = self.db.get_or_create_experiment("Variant_Test")
        sim_id = self.db.start_simulation(exp_id, "P", "S", "M", {}, "M", {}, "O", {}, "S", {})
        
        fitness_dict = {"SEQ1": 0.95, "SEQ2": 0.88}
        lineage_graph = {"SEQ1": ["WT"], "SEQ2": ["WT", "ALT_WT"]} # SEQ2 has multiple parents
        
        # Stream one generation
        self.db.save_generation(sim_id, generation=1, fitness_dict=fitness_dict, lineage_graph=lineage_graph)
        
        with self.db.SessionLocal() as session:
            variants = session.query(Variant).all()
            self.assertEqual(len(variants), 2)
            
            # Check parsing logic
            seq2_record = session.query(Variant).filter_by(sequence="SEQ2").first()
            self.assertEqual(seq2_record.parent_sequences, "WT,ALT_WT")

    def test_score_cache_management(self):
        # 1. Flush new scores to the DB
        new_scores = {
            ("OsHIPP", "SEQ_A"): -1.5,
            ("Pikm-1", "SEQ_A"): -2.0
        }
        self.db.flush_scores(model_name="MINT", new_scores=new_scores)
        
        # 2. Try flushing a duplicate to ensure it rolls back gracefully without crashing
        self.db.flush_scores(model_name="MINT", new_scores={("OsHIPP", "SEQ_A"): -1.5})
        
        # 3. Load them back out into the RAM format
        loaded_cache = self.db.load_scores(model_name="MINT", target_names=["OsHIPP", "Pikm-1"])
        
        self.assertEqual(len(loaded_cache), 2)
        self.assertIn(("OsHIPP", "SEQ_A"), loaded_cache)
        self.assertEqual(loaded_cache[("Pikm-1", "SEQ_A")], -2.0)

if __name__ == '__main__':
    unittest.main()