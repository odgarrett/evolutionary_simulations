# Evolutionary Simulations
In this repo, I aim to build a system for in silico directed evolution of AVR-Pik. The system will be modular, allowing the plug-and-play of any scoring function or model. It should be parallelizable and utilize caching to avoid duplicate predictions, both optimizing performance. It will also include analysis functions to characterize the fitness landscape and validate the evolutionary trajectories. 

## Engine
To start, I'll use zero shot [MINT](https://www.nature.com/articles/s41467-025-67971-3) predictions of AVR-Pik mutations with OsHIPP20 and Pikm-1 HMA domains as binding partners to generate binding scores.

## Simulation
In simulating evolution, I'll keep it as simple SNP-based mutation for now. To make it more realistic, I'll employ the Ts:Tv rates established in [this paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0065416). But, I'll be mutating the underlying DNA sequence, translating, then scoring.

To define fitness, I'll calibrate to naturally occurring alleles. Pikm-1 detects AVR-PikA but not C. Assuming these are the thresholds of detection vs non-detection, I'll score the likelihood of these and calibrate the probability of being detected to be near one for A and near zero for C. Between, I'll interpolate a sigmoid function. For OsHIPP20, the bounds aren't well-defined because there aren't any naturally occurring AVR-Pik alleles that completely lose affinity that I've tested. However, AVR-PikK is the lowest I've found, so I'll calibrate to that as lost effective binding and AVR-PikF is the lowest in my batch screen that has been verified as good enough, so I'll calibrate that as my effective binding boundary. Again, I'll interpolate a sigmoid between the two.

To join into a fitness score, I'll use the following formula. P is the probability of effective binding, F is fitness. $F = P_{OsHIPP} * (1 - P_{Pik-1})$. When P_{OsHIPP} is low or P_{Pik-1} is high, fitness is low. Only when P_{OsHIPP} is high and P_{Pik-1} is low can fitness be high.

## Analysis
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
