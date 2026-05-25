import random
from typing import Dict, List

class ProteinMutator:
    '''
    Handles mutations at the amino acid level.
    '''
    def __init__(self, mutation_rate: float):
        self.mutation_rate = mutation_rate
        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    def expand_mutant_pool(self, seed_sequences: List[str], pool_size: int, max_attempts: int = 100_000) -> Dict[str, List[str]]:
        '''
        Used for greedy algorithm evolution. Provided seed sequences are mutated
        with num_mut_per_seq amino acid substitutions yielding a total of pool_size
        mutant sequences. Tracks parent:child relationships, handling cases where
        multiple parents produce the same child (convergent evolution). 

        Returns:
            Dict[str, List[str]]: A mapping of {child_sequence: [list_of_parent_seeds]}
        '''
        pool = {}
        variants_per_seed = pool_size // len(seed_sequences)

        for parent_seq in seed_sequences:
            seq_length = len(parent_seq)
            mutations_per_sequence = max(1, int(seq_length * self.mutation_rate)) # make sure there's at least one mutation

            variants_generated = 0
            attempts = 0 # safety counter to break infinite loops

            while variants_generated < variants_per_seed and attempts < max_attempts:
                attempts += 1
                seq_chars = list(parent_seq)

                # Randomly select positions in the sequence to mutate
                mutation_indices = random.sample(range(seq_length), mutations_per_sequence)
                for idx in mutation_indices:
                    current_aa = seq_chars[idx]
                    other_aa = [aa for aa in self.amino_acids if aa != current_aa]
                    seq_chars[idx] = random.choice(other_aa)

                mut_seq = ''.join(seq_chars)

                # Check for parent:child uniqueness
                # First, check if the sequence is in the pool
                if mut_seq not in pool:
                    pool[mut_seq] = [parent_seq]
                    variants_generated += 1
                # If it is, make sure its a different parent
                else:
                    if parent_seq not in pool[mut_seq]:
                        pool[mut_seq].append(parent_seq)
                        variants_generated += 1
            if attempts >= max_attempts:
                print(f"Warning: Mutator exhausted its maximum attempts to generate a unique sequence. Successfully generated {variants_generated} new variants.")

        return pool