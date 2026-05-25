# Evolutionary Simulations (Pareto-Prot)
In this repo, I aim to build a system for in silico directed evolution of AVR-Pik. The system will be modular, allowing the plug-and-play of any scoring function or model. It should be parallelizable and utilize caching to avoid duplicate predictions, both optimizing performance. It will also include analysis functions to characterize the fitness landscape and validate the evolutionary trajectories. 

## Initial notes
### Prediction model
To start, I'll use zero shot [MINT](https://www.nature.com/articles/s41467-025-67971-3) predictions of AVR-Pik mutations with OsHIPP20 and Pikm-1 HMA domains as binding partners to generate binding scores.

### Mutation
For mutations I'll keep it as simple SNP-based mutation for now. To make it more realistic, I'll employ the Ts:Tv rates established in [this paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0065416). But, I'll be mutating the underlying DNA sequence, translating, then scoring.

### Fitness
To define fitness, I'll calibrate to naturally occurring alleles. Pikm-1 detects AVR-PikA but not C. Assuming these are the thresholds of detection vs non-detection, I'll score the likelihood of these and calibrate the probability of being detected to be near one for A and near zero for C. Between, I'll interpolate a sigmoid function. For OsHIPP20, the bounds aren't well-defined because there aren't any naturally occurring AVR-Pik alleles that completely lose affinity that I've tested. However, AVR-PikK is the lowest I've found, so I'll calibrate to that as lost effective binding and AVR-PikF is the lowest in my batch screen that has been verified as good enough, so I'll calibrate that as my effective binding boundary. Again, I'll interpolate a sigmoid between the two.

To join into a fitness score, I'll use the following formula. P is the probability of effective binding, F is fitness. $F = P_{OsHIPP} * (1 - P_{Pik-1})$. When P_{OsHIPP} is low or P_{Pik-1} is high, fitness is low. Only when P_{OsHIPP} is high and P_{Pik-1} is low can fitness be high.

## Architecture
The project will be divided into the following modules:

### Prediction models
Configures models to output required scores given sequence. Wraps each model in a custom class handling the initialization and converting a common score request format into something compatible with each specific model. 

### Objective function
Defines the function converting input model scores into an output fitness score.

### Mutator
Handles the mutational constraints. Should at least support simple amino acid random mutagenesis and DNA sequence mutagenesis with configurable nucleotide substitution frequencies. May also support recombination.

### Simulator
Runs the main loop. At each step, it applies the mutator to the current set of sequences, sends a request to the controller (see controller below), receives fitness scores, then updates allele frequencies.

### Server
Acts as a kind of internal server. Holds a common instance of the prediction model. Receives requests for model scores from simulators. It retrieves requests that are present in the cache (see database below) and predicts and stores those that aren't. Sends scores back to the simulators.

### Database
SQLite database with tables for the simulation cache and simulation results. It will have three tables: cache, simulations, and trajectories. Cache stores the aa sequence hash for fast look-up, the full amino acid sequence, and the model scores. Simulations keeps track of the different simulations ran with all of their parameters (e.g. analyzed protein, target proteins, mutator parameters, prediction model, population size, etc.), along with metadata like start_time and a UUID. Trajectories stores the stepwise data collected during the simulations.

### Experiment
Orchestrates multiple simulation runs to test a specific hypothesis. Manages configs and disk I/O.

### Analysis
Somehow makes sense of all of these data, doing at least the following:
1. UMAP projection of the embeddings of AVR-Pik variants with their frequencies in the simulation as the hue
2. Interpolate UMAP projection for 2D then a consolidated fitness score as the Z-axis and frequency in simulation to create the fitness landscape
3. Project the frequency of mutations onto an AVR-Pik 3D structure
4. Evaluate degree to which natural evolutionary trajectories are replicated

## Dev Log
### 26/05/20
- Cloned mint repo into this repo and attempted to install their conda environment, but it failed due to some package conflicts. Asked Gemini to revise the environment.yml file, yielding `environment_revised.yml`. The environment installed fine using this updated environment file.
- Followed mutational-ppi/prepare_data.ipynb. Had to make some adjustments for my desired file organization. Wrote a bash script to collect the required data. There were some apparently out-of-date column names and dict keys in their script to access the dataframe, so I had to edit these to get the notebook to work properly.
- With data prepared, I tried to run their embedding script, but ran into cuda issues. Realized its because I'm running on much newer hardware (laptop 5070Ti), so I need a newer CUDA version. Gemini assisted again and `environment_revised.yml` was again updated. CUDA is now available.
- Ran into an `UnpicklingError` because of the upgrade to Pytorch 2.6. Error gave the solution; had to modify mint/helpers/extract.py.
- Once I verified I could extract embeddings, I moved to generating embeddings with the goal of running their finetune_general script. To get it to work, I had to edit the weights_only arguments again because of the Pytorch upgrade, and I had to set the batch size to 1. Let it run over night, as it took at least an hour.

### 26/05/21
- Embedding generation for MutationalPPI worked. I then attempted to run their finetune script, but it assumed I was using weights and biases. Didn't want to make an account or anything, so I scrubbed that from their script. I also had to make several other edits to the script to save the models while training.
- "Fine-tuning" didn't take long at all. Then I realized that this is the wrong task. I want to be doing the SKEMPI procedure, not MutationalPPI.
- Successfully did fine-tuning with SKEMPI. Generating the embeddings actually took less than an hour. Had to make a few more modifications to the finetuning script to save the best models. Used the --use_mlp_for_cv flag to get a deep learning network for use.
- Used SKEMPI fine-tuned mint to make predictions on validated AVR-Pik-HMA interactions. Doesn't really work well, so I guess that's why we need a new model! But, it should be of similar size to what I will want to work with in the future, and it outputs a numerical predictor that I can use, so I'll give myself the greenlight to move forward building out the evolutionary simulation.

### 26/05/22
- Took a deeper look at the source code of MINT to see how it was calculating the feature vectors for ddG prediction. Turns out they subtract the mutant vector from the wild-type vector, not the other way around, so I needed to correct this in my ddG prediction notebook.

### 26/05/23
- Before getting started on the evolutionary simulation framework, I was thinking about names to give it. I think I've landed on playing on the concept of the Pareto frontier, which is the set of solutions to a multi-objective optimization function that are Pareto efficient (when there are no ways to improve one objective without hurting another). pareto-prot will be the tentative name.

### 26/05/24
- Worked out basic model architecture and dataflow (see above). Shaping up to be quite the project.
- Modularized the MINT prediction logic. Used Gemini to develop a test to make sure everything works. Probably a good habit to do this along the way.

### 25/05/25
- Starting fleshing out the main prediction loop assuming a basic greedy algorithm approach with mutations on the protein level. Built the corresponding mutator class and function.
- Built the objective function for pathogen escape. It has configurable parameters to set the bounds and the slope of the sigmoid function. The slope determines essentially how sensitive the composite fitness score is to binding each of the targets, so for AVR-Pik, it will be interesting to play with. For example, one would expect the immune receptor to be on a hair-trigger, while the virulence function of OsHIPP binding may be less so. By changing the slope of the sigmoid function and seeing how the evolutionary landscape changes, we can play out different scenarios.
- Also noticed that I was including my MINT scoring logic in the .gitignore, so I fixed that today and issued a corrective commit.