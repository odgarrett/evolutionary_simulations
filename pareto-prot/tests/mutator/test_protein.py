import unittest
from unittest.mock import patch
import random

# Adjust import path based on your project structure
from mutator.protein import ProteinMutator

class TestProteinMutator(unittest.TestCase):
    
    def setUp(self):
        """Initialize the mutator with the unified dynamic rate."""
        self.mutator = ProteinMutator(mutation_rate=0.01)

    def test_expansion_size_and_edit_distance(self):
        """Verify the mutator hits the pool size target and applies exact mutations."""
        seeds = ["AAAAA", "CCCCC"]
        pool_size = 10
        
        # Using your specific method name
        pool = self.mutator.expand_mutant_pool(seeds, pool_size)
        
        # 1. Did it generate exactly 10 unique children?
        self.assertEqual(len(pool), pool_size)
        
        # 2. Check parent mapping and edit distance
        variants_per_seed = pool_size // len(seeds)
        seed_1_count = 0
        seed_2_count = 0
        
        for child, parents in pool.items():
            self.assertEqual(len(parents), 1) # No collisions expected in this small sample
            parent = parents[0]
            
            if parent == "AAAAA":
                seed_1_count += 1
            elif parent == "CCCCC":
                seed_2_count += 1
                
            # Verify exact edit distance of 1
            # (Because max(1, int(5 * 0.01)) == 1)
            differences = sum(1 for c, p in zip(child, parent) if c != p)
            self.assertEqual(differences, 1, f"Child {child} should be exactly 1 mutation away from {parent}")
            
        # 3. Did it distribute the work evenly?
        self.assertEqual(seed_1_count, variants_per_seed)
        self.assertEqual(seed_2_count, variants_per_seed)

    def test_failsafe_limit(self):
        """Verify the mutator does not infinitely loop if the pool size exceeds physical possibilities."""
        # A 1-amino-acid sequence only has 19 possible mutants. 
        seeds = ["A"]
        impossible_pool_size = 50 
        
        pool = self.mutator.expand_mutant_pool(seeds, impossible_pool_size)
        
        # The mutator should gracefully exhaust all 19 options and return them
        self.assertEqual(len(pool), 19)
        self.assertNotIn("A", pool) 

    @patch('random.sample')
    @patch('random.choice')
    def test_convergent_evolution(self, mock_choice, mock_sample):
        """Force two different seeds to mutate into the identical child and verify lineage tracking."""
        seeds = ["CAAAA", "DAAAA"]
        
        # Force a collision:
        # random.sample will always choose index 0 to mutate
        mock_sample.return_value = [0]
        # random.choice will always choose the amino acid 'Y'
        mock_choice.return_value = 'Y'
        
        # Both seeds mutating index 0 to 'Y' will produce the exact same child: "YAAAA"
        pool = self.mutator.expand_mutant_pool(seeds, pool_size=2)
        
        # Because both mutated index 0 to 'Y', the only sequence produced is "YAAAA"
        self.assertEqual(len(pool), 1)
        self.assertIn("YAAAA", pool)
        
        # Verify that BOTH parents are tracked under this single child
        parents = pool["YAAAA"]
        self.assertEqual(len(parents), 2)
        self.assertIn("CAAAA", parents)
        self.assertIn("DAAAA", parents)

if __name__ == '__main__':
    unittest.main()