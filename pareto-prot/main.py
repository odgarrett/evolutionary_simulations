import argparse
import os
import glob
from db.manager import DatabaseManager
from conductor.experiment import Experiment

def main():
    parser = argparse.ArgumentParser(description="Pareto-Prot Evolutionary Simulation Engine")
    
    # Define our three command line arguments
    parser.add_argument('-c', '--config', type=str, 
                        help="Path to a single YAML config file.")
    parser.add_argument('-d', '--run_dir', type=str, 
                        help="Path to a directory of YAML config files to run sequentially.")
    parser.add_argument('--db', type=str, default="sqlite:///pareto_prot.db", 
                        help="Custom SQLite database URI (default: sqlite:///pareto_prot.db)")

    args = parser.parse_args()

    # Safety check
    if not args.config and not args.run_dir:
        parser.error("You must provide either a single config file (-c) or a directory of configs (-d).")

    print(f"Connecting to database at: {args.db}")
    db = DatabaseManager(db_path=args.db)
    runner = Experiment(db_manager=db)

    # Gather all targeted files
    configs_to_run = []
    
    if args.config:
        if os.path.exists(args.config):
            configs_to_run.append(args.config)
        else:
            print(f"Error: Config file not found at {args.config}")
            
    if args.run_dir:
        if os.path.isdir(args.run_dir):
            # Grab both .yaml and .yml extensions just in case
            yaml_files = glob.glob(os.path.join(args.run_dir, "*.yaml"))
            yaml_files += glob.glob(os.path.join(args.run_dir, "*.yml"))
            configs_to_run.extend(yaml_files)
        else:
            print(f"Error: Directory not found at {args.run_dir}")

    # Deduplicate in case a user passed the same file in both arguments
    configs_to_run = list(set(configs_to_run))

    if not configs_to_run:
        print("No valid configuration files found to run. Exiting.")
        return

    print(f"Found {len(configs_to_run)} configuration file(s) to execute.")
    print("=" * 60)

    # Run the batch
    for cfg_path in configs_to_run:
        runner.run_from_config(cfg_path)

if __name__ == "__main__":
    main()