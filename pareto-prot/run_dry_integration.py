import sys
import os
from typing import Dict, List

# Ensure current directory is on the path so we can import local modules cleanly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.mint.mint import MINTScorer
from server.server import ScoringServer
from mutator.protein import ProteinMutator
from objective.pathogen_escape import PathogenEscape
from simulator.greedy import GreedySimulator

def run_integration_test():
    print("=========================================================")
    print("STARTING FULL END-TO-END PIPELINE INTEGRATION RUN")
    print("=========================================================\n")

    # 1. DEFINE BIOLOGICAL INPUTS (AVR-Pik variants and host targets)
    wt_evolving_effector = (
        "ETGNKYIEKRAIDLSRERDPNFFDNADIPVPECFWFMFKNNVRQDAGTCYSSWKMDMKVGPNWVHIKSDDNCNLSGDFPPGWIVLGKKRPGF"
    )
    
    system_targets = {
        "OsHIPP20": "GEMQKIVFKIPMVDDKSRTKAMSLVASTVGVHSVAIAGDLRDQVVVVGDGIDSINLVSALRKKVGPAMFLEVSQVKED",
        "Pikm-1": "GLKQKIVIKVAMEGNNCRSKAMALVASTGGVDSVALVGDLRDKIEVVGYGIDPIKLISALRKKVGDAELLQVSQANKD"
    }

    # 2. INITIALIZE HARDWARE LAYER & PREDICTIVE MODEL
    # Update these paths relative to your layout
    config_path = "../mint/data/esm2_t33_650M_UR50D.json"
    weights_path = "model/mint/mint.ckpt"
    regressor_path = "model/mint/SKEMPI_v2.joblib"

    print("Step 1: Instantiating deep learning model and pre-computing WT baselines...")
    try:
        mint_model = MINTScorer(
            mint_config_path=config_path,
            mint_weights_path=weights_path,
            regressor_path=regressor_path,
            target_dict=system_targets,
            wt_sequence=wt_evolving_effector
        )
    except Exception as e:
        print(f"CRITICAL ERROR loading MINT model: {e}")
        return

    # 3. INITIALIZE SCORING SERVER (RAM layer wrapper)
    print("\nStep 2: Starting local Scoring Server instance...")
    server = ScoringServer(
        model=mint_model,
        server_name="mint_cuda_server",
        preloaded_cache=None, # Fresh start: no database cache yet
        max_cache_size=50_000 # Keeping it small for integration visibility
    )

    # 4. SET UP EVOLUTIONARY DRIVERS
    print("Step 3: Building biological mutator and scoring objective...")
    # About 1 aa mutation per 93 amino acid sequence
    mutator = ProteinMutator(mutation_rate=(1.0/93.0))
    
    # Pathogen Escape Objective
    # Target_A (OsHIPP20) -> High score = better binding (virulence)
    # Target_B (Pikm-1) -> Low score = less binding (immune evasion)
    objective = PathogenEscape(
        target_a_name="OsHIPP20",
        target_b_name="Pikm-1",
        target_a_bounds=(0.0666, 0.0658),  # best score known to not bind, worst score known to bind
        target_b_bounds=(0.0666, 0.0660),  
        target_a_slope=10.0,
        target_b_slope=10.0,
        target_a_higher_is_better=False, # MINT ddG: lower is better binder
        target_b_higher_is_better=False  # MINT ddG: lower is better binder
    )

    # 5. ASSEMBLE SIMULATOR CORE
    print("Step 4: Assembling Greedy Evolutionary Simulator...")
    simulator = GreedySimulator(
        server=server,
        mutator=mutator,
        objective_fn=objective,
        target_names=list(system_targets.keys()),
        pool_size=1000,   # Generate 40 variants per generation
        top_k=100,        # Beam width of 5 surviving lineages
        batch_size=8    # Run 8 concurrent structures through the GPU
    )
    
    # Inject wild-type sequence as the foundational seed
    simulator.initialize_seed_sequences([wt_evolving_effector])

    # 6. RUN EXHAUSTIVE INTEGRATION LOOP
    generations_to_test = 5
    print(f"\nStep 5: Executing {generations_to_test} evolutionary generations...")
    print("-" * 60)

    try:
        # Step through the generator stream
        for gen, fitness_history, lineage in simulator.simulate_generations(rounds=generations_to_test):
            print(f"Generation {gen} Complete:")
            print(f"  -> Total unique sequences evaluated: {len(fitness_history)}")
            
            # Identify the peak performer in the current population matrix
            best_variant = max(fitness_history.keys(), key=lambda v: fitness_history[v])
            best_fitness = fitness_history[best_variant]
            
            print(f"  -> Peak variant composite fitness: {best_fitness:.4f}")
            
            # Print Server tracking stats for the generation
            server.print_cache_data()
            
            # Simulate what the database manager will look at: extracting data to flush
            flushed_entries = server.extract_new_cache_for_db()
            print(f"  -> Simulated DB cache extraction: {len(flushed_entries)} new entries prepared.")
            print("-" * 60)
            
    except Exception as e:
        print(f"\nCRITICAL PIPELINE FAILURE during simulation run: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n=========================================================")
    print("INTEGRATION SUCCESSFUL: Core engine loop is fully validated.")
    print("=========================================================")

if __name__ == "__main__":
    run_integration_test()