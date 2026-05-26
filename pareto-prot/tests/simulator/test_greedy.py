import unittest
from simulator.greedy import GreedySimulator

class DummyServer:
    def request_phenotype_scores(self, target_names, variant_sequences, batch_size=4):
        return [float(len(v)) for v in variant_sequences]

class DummyMutator:
    def expand_mutant_pool(self, seed_sequences, pool_size):
        # Mocks your actual method's return type: {child: [parents]}
        pool = {}
        for parent in seed_sequences:
            pool[parent + "MUT1"] = [parent]
            pool[parent + "MUT2"] = [parent]
        return pool

class DummyObjective:
    def calculate_fitness(self, score_dict):
        return sum(score_dict.values())

class TestGreedySimulator(unittest.TestCase):
    def setUp(self):
        self.server = DummyServer()
        self.mutator = DummyMutator()
        self.objective = DummyObjective()
        self.target_names = ["Target_A", "Target_B"]
        
        self.simulator = GreedySimulator(
            server=self.server,
            mutator=self.mutator,
            objective_fn=self.objective,
            target_names=self.target_names,
            pool_size=10,
            top_k=2,
            batch_size=4
        )

    def test_uninitialized_execution(self):
        """Ensure the simulator crashes gracefully if seeds are forgotten."""
        with self.assertRaises(ValueError):
            list(self.simulator.simulate_generations(rounds=1))

    def test_generator_yield_structure(self):
        """Verify the data structures match pool-expansion lineage parameters."""
        self.simulator.initialize_seed_sequences(["SEED1"])
        
        results = list(self.simulator.simulate_generations(rounds=1))
        self.assertEqual(len(results), 1)
        
        gen_idx, fitness_dict, lineage_graph = results[0]
        self.assertEqual(gen_idx, 0)
        
        # Verify keys exist and trace correctly
        self.assertIn("SEED1MUT1", fitness_dict)
        self.assertEqual(lineage_graph["SEED1MUT1"], ["SEED1"])

    def test_top_k_selection(self):
        """Ensure selection appropriately retains only top-K strings as subsequent parents."""
        self.simulator.initialize_seed_sequences(["S"])
        
        # FIX: Update the lambda to accept the correct keyword arguments 
        # and return the expected dictionary layout {child: [parents]}
        self.simulator.mutator.expand_mutant_pool = lambda seed_sequences, pool_size: {
            "SHORT": ["S"],
            "LONGER_STRING": ["S"],
            "MAX_LENGTH_WINNER": ["S"]
        }
        
        # Force generator execution
        list(self.simulator.simulate_generations(rounds=1))
        
        # Because top_k is 2, "MAX_LENGTH_WINNER" and "LONGER_STRING" should survive.
        self.assertEqual(len(self.simulator.seed_sequences), 2)
        self.assertIn("MAX_LENGTH_WINNER", self.simulator.seed_sequences)
        self.assertIn("LONGER_STRING", self.simulator.seed_sequences)
        self.assertNotIn("SHORT", self.simulator.seed_sequences)

if __name__ == '__main__':
    unittest.main()