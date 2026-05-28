import os
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Base, Experiment, Simulation, Variant, Score

class DatabaseManager:
    '''
    Handles connection lifecyle and data transactions for SQLite.
    '''
    def __init__(self, db_path: str = "sqlite:///pareto_prot.db"):
        # Create the file (named pareto_prot.db by default) in your root directory
        self.engine = create_engine(db_path)

        # Check file and build any missing tables
        Base.metadata.create_all(self.engine)

        # Create a temporary session to interact with the DB
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def check_connection(self):
        try:
            with self.engine.connect() as connection:
                print(f"Successfully connected to {self.engine.url}")
            return True
        except Exception as e:
            print(f"Database connection failed: {e}")
            return False

    ### Experiment and Simulation Tracking ###

    def get_or_create_experiment(self, name: str, description: str = "") -> int:
        '''
        Finds an existing experiment by name, or creates a new one.
        '''
        with self.SessionLocal() as session:
            exp = session.query(Experiment).filter_by(name=name).first()
            if not exp:
                exp = Experiment(name=name, description=description)
                session.add(exp)
                session.commit()
                session.refresh(exp)
            return exp.id

    def start_simulation(
        self, exp_id: int, poi_name: str, wt_sequence: str,
        model_name: str, model_params: dict,
        mutator_name: str, mutator_params: dict,
        objective_name: str, objective_params: dict,
        simulator_name: str, simulator_params: dict
    ) -> int:
        '''
        Logs the start of a simulation and returns its ID.
        '''
        with self.SessionLocal() as session:
            sim = Simulation(
                experiment_id=exp_id,
                poi_name=poi_name,
                wt_sequence=wt_sequence,
                model=model_name,
                model_params=model_params,
                mutator=mutator_name,
                mutator_params=mutator_params,
                objective=objective_name,
                objective_params=objective_params,
                simulator=simulator_name,
                simulator_params=simulator_params
            )
            session.add(sim)
            session.commit()
            session.refresh(sim)
            return sim.id

    def finish_simulation(self, sim_id: int):
        '''
        Marks the simulation complete with a final timestamp.
        '''
        with self.SessionLocal() as session:
            sim = session.query(Simulation).filter_by(id=sim_id).first()
            if sim:
                sim.end_timestamp = datetime.utcnow()
                session.commit()

    ### Trajectory Tracking ###

    def save_generation(self, sim_id: int, generation: int, fitness_dict: dict, lineage_graph: dict):
        '''
        Inserts a single generation of variants.
        '''
        with self.SessionLocal() as session:
            variants_to_insert = []

            for seq, fitness in fitness_dict.items():
                # Extract parents and join into a comma-separated string
                parents = lineage_graph.get(seq, [])
                parent_str = ','.join(parents) if parents else ''

                variants_to_insert.append(
                    Variant(
                        simulation_id=sim_id,
                        generation=generation,
                        sequence=seq,
                        fitness=fitness,
                        parent_sequences=parent_str
                    )
                )

            # Bulk save
            session.bulk_save_objects(variants_to_insert)
            session.commit()

    ### Score / Cache Management ###

    def load_scores(self, model_name: str, target_names: list, total_limit: int = 2_000_000) -> dict:
        '''
        Pulls relevant historical predictions from SQLite for the server's RAM.
        '''
        with self.SessionLocal() as session:
            # We split the cache between the first mutants scored (likely single and double mutants) and the most
            # recent scored
            basal_limit = int(total_limit * 0.25)
            recent_limit = total_limit - basal_limit

            base_query = session.query(Score).filter(
                Score.model == model_name,
                Score.target_name.in_(target_names)
            )

            oldest_records = base_query.order_by(Score.id.asc()).limit(basal_limit).all()
            newest_records = base_query.order_by(Score.id.desc()).limit(recent_limit).all()
            combined_results = oldest_records + newest_records

        # Return format expected by ScoringServer: {(target, variant): score}
        return {(row.target_name, row.mut_seq): row.score for row in combined_results}

    def flush_scores(self, model_name: str, new_scores: dict):
        '''
        Bulk inserts novel predictions back into the database.
        '''
        if not new_scores:
            return

        with self.SessionLocal() as session:
            scores_to_insert = [
                Score(
                    model=model_name,
                    target_name=target,
                    mut_seq=variant,
                    score=score
                )
                for (target, variant), score in new_scores.items()
            ]

            try: 
                session.bulk_save_objects(scores_to_insert)
                session.commit()
            except Exception as e:
                session.rollback()
                print(f"Warning: Failed to flush scores (likely a duplicate constraint). Error: {e}")
