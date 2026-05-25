import unittest
from objective.pathogen_escape import PathogenEscape

class TestPathogenEscape(unittest.TestCase):
    
    def setUp(self):
        """Initialize standard environments for OsHIPP20 (virulence) and Pikm-1 (immune)."""
        self.virulence_target = "OsHIPP20"
        self.immune_target = "Pikm-1"
        
        # Standard Objective: 
        # 0.04 -> 0% chance of binding. 0.08 -> 100% chance of binding.
        # This applies equally to both the virulence target and the immune target.
        self.objective = PathogenEscape(
            target_a_name=self.virulence_target,
            target_b_name=self.immune_target,
            target_a_bounds=(0.04, 0.08),  
            target_b_bounds=(0.04, 0.08),  
            target_a_slope=10.0,
            target_b_slope=10.0
        )

    def test_sigmoid_boundaries(self):
        """Test that the sigmoid outputs near 0%, 50%, and 100% at the correct bounds."""
        # Test Virulence Binding Bounds
        prob_low = self.objective._sigmoid_probability(0.04, 0.04, 0.08, 10.0)
        prob_mid = self.objective._sigmoid_probability(0.06, 0.04, 0.08, 10.0)
        prob_high = self.objective._sigmoid_probability(0.08, 0.04, 0.08, 10.0)
        
        self.assertLess(prob_low, 0.01)
        self.assertAlmostEqual(prob_mid, 0.5, places=3)
        self.assertGreater(prob_high, 0.99)

        # Test Immune Binding Bounds (Same logic now: high score = high binding probability)
        prob_bind_low = self.objective._sigmoid_probability(0.04, 0.04, 0.08, 10.0)
        prob_bind_high = self.objective._sigmoid_probability(0.08, 0.04, 0.08, 10.0)
        
        self.assertLess(prob_bind_low, 0.01)
        self.assertGreater(prob_bind_high, 0.99)

    def test_composite_fitness_extremes(self):
        """Verify the multiplicative logic correctly penalizes failure on either axis."""
        
        # Scenario 1: High Virulence Binding + Low Immune Binding
        # P(A) ~ 1.0, P(B) ~ 0.0 -> Fitness ~ 1.0
        scores_perfect = {self.virulence_target: 0.10, self.immune_target: 0.02}
        fitness = self.objective.calculate_fitness(scores_perfect)
        self.assertGreater(fitness, 0.99)
        
        # Scenario 2: High Virulence Binding, but HIGH Immune Binding
        # P(A) ~ 1.0, P(B) ~ 1.0 -> Fitness ~ 0.0
        scores_caught = {self.virulence_target: 0.10, self.immune_target: 0.10}
        fitness = self.objective.calculate_fitness(scores_caught)
        self.assertLess(fitness, 0.01)

        # Scenario 3: Low Immune Binding, but LOW Virulence Binding
        # P(A) ~ 0.0, P(B) ~ 0.0 -> Fitness ~ 0.0
        scores_weak = {self.virulence_target: 0.02, self.immune_target: 0.02}
        fitness = self.objective.calculate_fitness(scores_weak)
        self.assertLess(fitness, 0.01)

    def test_slope_sensitivity(self):
        """Verify that a gentler slope creates a wider permissive fitness landscape."""
        # Fixed bounds here to match the new consistent logic: (0.0, 1.0) for both
        steep_objective = PathogenEscape(
            "A", "B", (0.0, 1.0), (0.0, 1.0), 
            target_a_slope=20.0, target_b_slope=20.0
        )
        shallow_objective = PathogenEscape(
            "A", "B", (0.0, 1.0), (0.0, 1.0), 
            target_a_slope=2.0, target_b_slope=2.0
        )
        
        # Evaluate at a slightly suboptimal score for A (0.4) and perfect evasion for B (0.0)
        scores = {"A": 0.4, "B": 0.0}
        
        steep_fitness = steep_objective.calculate_fitness(scores)
        shallow_fitness = shallow_objective.calculate_fitness(scores)
        
        # The steep slope should harshly punish the 0.4 score (well below midpoint 0.5)
        # The shallow slope should be much more forgiving
        self.assertGreater(shallow_fitness, steep_fitness)

    def test_missing_keys(self):
        """Ensure missing keys silently drop to the 0.0 default, triggering low fitness."""
        fitness = self.objective.calculate_fitness({})
        
        # Defaults to 0.0 for A and 0.0 for B. 
        # P(A) for 0.0 is ~0.0. P(B) for 0.0 is ~0.0. Fitness = 0.0 * (1 - 0.0) = 0.0.
        self.assertLess(fitness, 0.01)

if __name__ == '__main__':
    unittest.main()