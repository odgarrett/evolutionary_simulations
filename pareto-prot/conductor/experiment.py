import yaml
import traceback

from model.mint.mint import MINTScorer
from server.server import ScoringServer
from mutator.protein import ProteinMutator
from objective.pathogen_escape import PathogenEscape
from simulator.greedy import GreedySimulator

class Experiment:
    '''
    The top-level unit to run simulations to test a specific hypothesis. Parses
    YAML configs, dynamically instantiates components from these configs, 
    manages the database stream and handles GPU memory teardown.
    '''
    def __init__(self, db_manager):
        self.db = db_manager

        self.registry = {
            "model": {
                "MINTScorer": MINTScorer
            },
            "server": {
                "ScoringServer": ScoringServer
            },
            "mutator": {
                "ProteinMutator": ProteinMutator
            },
            "objective": {
                "PathogenEscape": PathogenEscape
            },
            "simulator": {
                "GreedySimulator": GreedySimulator
            }
        }

    def _instantiate(self, component_type: str, config_block: dict, extra_kwargs: dict = None):
        '''
        Helper to fetch class from registry and unpack params.
        '''
        class_name = config_block['name']
        available_classes = self.registry.get(component_type, {})
        
        if class_name not in available_classes:
            options = ", ".join(available_classes.keys())
            raise KeyError(
                f"Invalid {component_type}: '{class_name}'. "
                f"Available options in registry are: [{options}]"
            )
        
        ComponentClass = available_classes[class_name]
        params = config_block.get('params', {}).copy()
        
        if extra_kwargs:
            params.update(extra_kwargs)
            
        return ComponentClass(**params)

    def run_from_config(self, config_path: str):
        """Executes the full end-to-end pipeline from a single YAML file."""
        print(f"\n[{config_path}] Loading configuration...")
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # Initialize experiment
        exp_id = self.db.get_or_create_experiment(
            name=cfg['experiment_name'], 
            description=cfg.get('experiment_desc', "")
        )
        
        replicates = cfg.get('replicates', 1)
        target_names = list(cfg['targets'].keys())
        target_dict = cfg['targets']
        ram_limit = cfg.get('ram_limit', 2_000_000)

        print(f"Experiment initialized. Queuing {replicates} replicate(s).")
        
        # Execute replicates
        for rep in range(1, replicates + 1):
            print(f"\n--- Starting Replicate {rep}/{replicates} ---")
            
            # Start a new Simulation tied to the parent Experiment
            sim_id = self.db.start_simulation(
                exp_id=exp_id,
                poi_name=cfg['poi_name'],
                wt_sequence=cfg['wt_sequence'],
                model_name=cfg['model']['name'],
                model_params=cfg['model'].get('params', {}),
                mutator_name=cfg['mutator']['name'],
                mutator_params=cfg['mutator'].get('params', {}),
                objective_name=cfg['objective']['name'],
                objective_params=cfg['objective'].get('params', {}),
                simulator_name=cfg['simulator']['name'],
                simulator_params=cfg['simulator'].get('params', {})
            )

            try:
                # Load cache inside the loop so subsequent replicates benefit 
                # from the ML predictions of earlier replicates
                print("Loading historical prediction cache from SQLite...")
                historical_scores = self.db.load_scores(
                    model_name=cfg['model']['name'],
                    target_names=target_names,
                    total_limit=ram_limit
                )

                # Initialize simulation components
                print("Initializing simulation components...")
                model = self._instantiate('model', cfg['model'], extra_kwargs={
                    'wt_sequence': cfg['wt_sequence'],
                    'target_dict': target_dict
                })
                
                server_kwargs = cfg['server'].get('params', {})
                batch_size = server_kwargs.pop('batch_size', 4) 
                server_kwargs.update({
                    'model': model,
                    'preloaded_cache': historical_scores,
                    'max_cache_size': ram_limit
                })
                server = self._instantiate('server', cfg['server'], extra_kwargs=server_kwargs)

                mutator = self._instantiate('mutator', cfg['mutator'])
                objective = self._instantiate('objective', cfg['objective'])

                simulator = self._instantiate('simulator', cfg['simulator'], extra_kwargs={
                    'server': server,
                    'mutator': mutator,
                    'objective_fn': objective,
                    'target_names': target_names,
                    'batch_size': batch_size
                })
                
                simulator.initialize_seed_sequences([cfg['wt_sequence']])

                # Run Generations
                print(f"Beginning evolutionary trajectory for {cfg['generations']} generations...")
                for gen, fitness_dict, lineage_graph in simulator.simulate_generations(rounds=cfg['generations']):
                    print(f"  -> Generation {gen} complete. Evaluated {len(fitness_dict)} unique variants.")
                    self.db.save_generation(sim_id, gen, fitness_dict, lineage_graph)

            except Exception as e:
                print(f"CRITICAL PIPELINE FAILURE during replicate {rep}: {e}")
                traceback.print_exc()
                print("Attempting to salvage and flush cache before moving to next replicate...")

            finally:
                # Teardown & Cache Flush (Per-Replicate)
                print("Executing teardown and cache flushing...")
                self.db.finish_simulation(sim_id)
                
                # Check if server was successfully instantiated before trying to extract from it
                if 'server' in locals():
                    flushed_entries = server.extract_new_cache_for_db()
                    if flushed_entries:
                        self.db.flush_scores(cfg['model']['name'], flushed_entries)
                        print(f"  -> Successfully committed {len(flushed_entries)} novel GPU predictions to SQLite.")
                
        print(f"\nExperiment [{cfg['experiment_name']}] complete. All replicates finished.\n")
    