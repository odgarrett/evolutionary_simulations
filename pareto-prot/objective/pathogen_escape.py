import math
from typing import Dict

class PathogenEscape:
    '''
    Evaluates a composite fitness score from opposing dual objectives, 
    e.g. maintaining binding to a virulence target while evading detection 
    by an immune receptor.

    Requires calibration by bounding values for each objective. Provide
    as tuple (worst_score, best_score) along with directionality flags.
    Interpolates between bounds using a sigmoid, then computes fitness
    as Fitness = P(Target_A_Binding) * (1 - P(Target_B_Binding)).
    '''
    def __init__(
        self, 
        target_a_name: str,
        target_b_name: str,
        target_a_bounds: tuple, 
        target_b_bounds: tuple,
        target_a_slope: float = 10.0,
        target_b_slope: float = 10.0,
        target_a_higher_is_better: bool = True,
        target_b_higher_is_better: bool = True
    ):
        self.name_a = target_a_name
        self.name_b = target_b_name
        
        # Unpack parameters assuming (worst_score, best_score) semantic layout
        a_bound_1, a_bound_2 = target_a_bounds
        b_bound_1, b_bound_2 = target_b_bounds
        
        # Enforce directionality based on the biological reality of the metric
        if target_a_higher_is_better:
            self.a_low, self.a_high = min(a_bound_1, a_bound_2), max(a_bound_1, a_bound_2)
        else:
            self.a_low, self.a_high = max(a_bound_1, a_bound_2), min(a_bound_1, a_bound_2)
            
        if target_b_higher_is_better:
            self.b_low, self.b_high = min(b_bound_1, b_bound_2), max(b_bound_1, b_bound_2)
        else:
            self.b_low, self.b_high = max(b_bound_1, b_bound_2), min(b_bound_1, b_bound_2)

        self.a_slope = target_a_slope
        self.b_slope = target_b_slope
        

    def _sigmoid_probability(self, val: float, low: float, high: float, slope_factor: float) -> float:
        if val <= min(low, high): return 0.0 if low < high else 1.0
        if val >= max(low, high): return 1.0 if low < high else 0.0
        
        midpoint = (low + high) / 2.0
        
        slope = slope_factor / (high - low)
        return 1.0 / (1.0 + math.exp(-slope * (val - midpoint)))

    def calculate_fitness(self, score_dict: Dict[str, float]) -> float:
        '''
        Fitness = P(Target_A_Binding) * (1 - P(Target_B_Binding))
        '''
        raw_score_a = score_dict.get(self.name_a, 0.0)
        raw_score_b = score_dict.get(self.name_b, 0.0)

        prob_a = self._sigmoid_probability(raw_score_a, self.a_low, self.a_high, self.a_slope)
        prob_b = self._sigmoid_probability(raw_score_b, self.b_low, self.b_high, self.b_slope)

        return prob_a * (1.0 - prob_b)