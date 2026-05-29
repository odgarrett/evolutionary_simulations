from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, UniqueConstraint
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Experiment(Base):
    __tablename__ = 'experiments'

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(String, nullable=True)

    simulations = relationship("Simulation", back_populates="experiment")

class Simulation(Base):
    __tablename__ = 'simulations'

    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey('experiments.id'), nullable=False)

    condition_name = Column(String, nullable=False)
    replicate_num = Column(Integer, nullable=False)

    start_timestamp = Column(DateTime, default=datetime.utcnow)
    end_timestamp = Column(DateTime, nullable=True)

    poi_name = Column(String, nullable=False)
    wt_sequence = Column(String, nullable=False)

    model = Column(String, nullable=False)
    model_params = Column(JSON, nullable=False)

    mutator = Column(String, nullable=False)
    mutator_params = Column(JSON, nullable=False)

    objective = Column(String, nullable=False)
    objective_params = Column(JSON, nullable=False)

    simulator = Column(String, nullable=False)
    simulator_params = Column(JSON, nullable=False)

    experiment = relationship("Experiment", back_populates="simulations")
    variants = relationship("Variant", back_populates="simulation")

class Variant(Base):
    __tablename__ = 'variants'

    id = Column(Integer, primary_key=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"), nullable=False)

    generation = Column(Integer, nullable=False)
    sequence = Column(String, nullable=False)
    fitness = Column(Float, nullable=False)
    parent_sequences = Column(String, nullable=False) # Comma-separated parent strings

    simulation = relationship("Simulation", back_populates="variants")

class Score(Base):
    __tablename__ = 'scores'

    id = Column(Integer, primary_key=True)
    model = Column(String, nullable=False, index=True)
    target_name = Column(String, nullable=False, index=True)
    mut_seq = Column(String, nullable=False, index=True)
    score = Column(Float, nullable=False)

    # Protect against duplicating the exact same prediction
    __table_args__ = (
        UniqueConstraint('model', 'target_name', 'mut_seq', name='_model_target_seq_uc'),
    )
