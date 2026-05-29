from typing import List, Dict, Iterator, Tuple
import numpy as np

class GreedySimulator:
    '''
    Executes a greedy search algorithm through the protein fitness landscape.
    1. Mutate seed sequence(s).
    2. Send to server for scoring.
    3. Send scores to objective function for a composite score.
    4. Select Top-K variants.
    5. Repeat.

    Yields data generation-by-generation.
    '''
    def __init__(
        self,
        server,
        mutator,
        objective_fn,
        target_names: List[str],
        pool_size: int = 100,
        top_k: int = 10,
        batch_size: int = 4
    ):
        self.server = server
        self.mutator = mutator
        self.objective_fn = objective_fn
        self.target_names = target_names

        # Hyperparameters
        self.pool_size = pool_size
        self.top_k = top_k
        self.batch_size = batch_size

        self.seed_sequences = []

    def initialize_seed_sequences(self, seeds: List[str]):
        '''Sets the starting sequences for Generation 0.'''
        self.seed_sequences = seeds

    def _print_telemetry(self, gen, fitness_dict, variant_scores):
        """Prints an aesthetic console block of variant statistics."""
        fit_vals = list(fitness_dict.values())
        
        print(f"\n" + "═"*50)
        print(f" GENERATION {gen} STATS ".center(50, "═"))
        print("═"*50)
        print(f" Population Size: {len(fit_vals)} variants".center(50))
        print("─"*50)
        
        # Composite Fitness (Higher is Better)
        print(f" [Composite Fitness]")
        print(f"   Max:    {np.max(fit_vals):>8.4f}")
        print(f"   Mean:   {np.mean(fit_vals):>8.4f}")
        print(f"   Median: {np.median(fit_vals):>8.4f}")
        print(f"   Min:    {np.min(fit_vals):>8.4f}")
        
        # Raw Target Scores (Lower is Better for MINT ddG)
        for target in self.target_names:
            print("─"*50)
            # Extract scores strictly for this target directly from the nested dictionary
            t_vals = [scores[target] for variant, scores in variant_scores.items() if target in scores]
            
            if t_vals:
                print(f" [Target: {target}]")
                print(f"   Min:    {np.min(t_vals):>8.4f}")
                print(f"   Median: {np.median(t_vals):>8.4f}")
                print(f"   Mean:   {np.mean(t_vals):>8.4f}")
                print(f"   Max:    {np.max(t_vals):>8.4f}")

        print("═"*50 + "\n")
    
    def simulate_generations(self, rounds: int) -> Iterator[Tuple[int, Dict[str, float], Dict[str, List[str]]]]:
        '''
        The main evolution loop.
        Yields: (generation_index, fitness_dictionary, lineage_graph)
        '''
        if not self.seed_sequences:
            raise ValueError("Must initialize seeds before executing trajectory.")
        
        for gen_idx in range(rounds):
            # Mutate
            current_lineage_graph = self.mutator.expand_mutant_pool(
                seed_sequences=self.seed_sequences,
                pool_size=self.pool_size
            )

            # Extract unique variants generated
            mutant_list = list(current_lineage_graph.keys())

            # Assemble evaluation pool to include parents, in case they're better
            evaluation_pool = list(set(mutant_list + self.seed_sequences))

            # Prepare batch for the scoring server
            server_targets = []
            server_variants = []

            for variant in evaluation_pool:
                for target in self.target_names:
                    server_targets.append(target)
                    server_variants.append(variant)

            # Score via the scoring server
            flat_scores = self.server.request_phenotype_scores(
                target_names=server_targets,
                variant_sequences=server_variants,
                batch_size=self.batch_size
            )

            # Reconstruct score dictionaries for the objective function
            variant_scores = {v: {} for v in evaluation_pool}
            for target, variant, score in zip(server_targets, server_variants, flat_scores):
                variant_scores[variant][target] = score

            # Evaluate fitness
            current_generation_fitness = {}
            for variant, scores in variant_scores.items():
                fitness = self.objective_fn.calculate_fitness(scores)
                current_generation_fitness[variant] = fitness

            # Select top-k
            sorted_variants = sorted(
                current_generation_fitness.keys(),
                key=lambda v: current_generation_fitness[v],
                reverse=True
            )

            # Move top performers to seeds for the next round
            self.seed_sequences = sorted_variants[:self.top_k]

            # Print the aesthetic block
            self._print_telemetry(gen_idx, current_generation_fitness, variant_scores)

            # Yield data for the database manager to store
            yield gen_idx, current_generation_fitness, current_lineage_graph
    
