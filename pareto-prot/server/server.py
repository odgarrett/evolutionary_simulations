from typing import List, Dict, Optional
from collections import OrderedDict

class ScoringServer:
    '''
    Interfaces with the predictive model.
    Maintains a cache for performance and later analysis.
    '''
    def __init__(
        self,
        model,
        server_name: str,
        preloaded_cache: Optional[Dict[tuple, float]] = None,
        max_cache_size: int = 2_000_000
    ):
        self.scorer = model
        self.server_name = server_name
        self.max_cache_size = max_cache_size

        # Use OrderedDict to track cache access history for smart wiping
        self.historic_cache = OrderedDict(preloaded_cache) if preloaded_cache else OrderedDict()
        self.new_cache = {}

        # Metrics to see how useful caching is
        self.cache_hits = 0
        self.inference_calls = 0

    def _check_memory_limit(self, incoming_count: int = 0):
        '''
        A helper function to make sure you don't go over your cache limit.
        The idea is to prevent overdrawing RAM. You'll need to calibrate
        the max for your system.
        Once the max cache is reached, it clears the least recently used 20%
        of the cache.
        '''
        total_size = len(self.historic_cache) + len(self.new_cache) + incoming_count

        if total_size > self.max_cache_size:
            # Drop at least the amount we are over, or 20% of the max cache, whichever is larger
            amount_over = total_size - self.max_cache_size
            drop_count = max(amount_over, int(self.max_cache_size * 0.20))
            for _ in range(drop_count):
                try:
                    # last=False pops item from beginning of the dict,
                    # which was the least recently accessed
                    self.historic_cache.popitem(last=False)
                except KeyError:
                    break # cache is empty
            
            print(f"[{self.server_name}] Max cache reached. Removed the {drop_count} least accessed historical predictions.")
    
    def request_phenotype_scores(
        self,
        target_names: List[str],
        variant_sequences: List[str],
        batch_size: int = 1
    ) -> List[float]:
        '''
        Takes a list of targets and variants, checks for cached pairs,
        routes new pairs to the prediction model, then returns a fully
        assembled list of scores.
        '''
        if len(target_names) != len(variant_sequences):
            raise ValueError("Target names and variant sequences must be the same length.")

        # Initialize and index to update to preserve variant order
        final_scores = [0.0] * len(variant_sequences)

        novel_targets = []
        novel_variants = []
        novel_indices = []

        # Routing logic
        for idx, (target, variant) in enumerate(zip(target_names, variant_sequences)):
            cache_key = (target, variant) # select based on unique target name and variant sequence pair

            # Check historical cache (originally from database)
            if cache_key in self.historic_cache:
                final_scores[idx] = self.historic_cache[cache_key]
                self.cache_hits += 1

                # Move just-queried entries to the end of the dict to
                # protect them from clearing when cache limit is reached
                self.historic_cache.move_to_end(cache_key)

            # Check cache accumulated this run
            elif cache_key in self.new_cache:
                final_scores[idx] = self.new_cache[cache_key]
                self.cache_hits += 1

            # Send to GPU if not in cache
            else:
                novel_targets.append(target)
                novel_variants.append(variant)
                novel_indices.append(idx)

        # GPU inference logic
        if novel_variants:
            self._check_memory_limit(incoming_count=len(novel_variants))
            self.inference_calls += len(novel_variants)

            computed_scores = self.scorer.score_mutants(
                target_names=novel_targets,
                mutant_seqs=novel_variants,
                batch_size=batch_size
            )

            # Update cache and assemble final list
            for idx, score, target, variant in zip(novel_indices, computed_scores, novel_targets, novel_variants):
                self.new_cache[(target, variant)] = score
                final_scores[idx] = score

        return final_scores

    def extract_new_cache_for_db(self) -> Dict[tuple, float]:
        '''
        Returns the new entries for saving to database, rotates them into
        the historic cache, the clears the new cache for future updates.
        '''
        # Copy new cache
        to_flush = self.new_cache.copy()

        # Merge new cache into history
        self.historic_cache.update(to_flush)
        
        # Clear new cache so only new data is in the next batch
        self.new_cache.clear()

        # Return flushed data for the database manager
        return to_flush

    def print_cache_data(self):
        '''Prints caching efficiency.'''
        total = self.cache_hits + self.inference_calls
        if total > 0:
            saved_pct = (self.cache_hits / total) * 100
            print(f"[{self.server_name}] Cache Hits: {self.cache_hits} | "
                  f"GPU Calls: {self.inference_calls} | Compute Saved: {saved_pct:.1f}%")