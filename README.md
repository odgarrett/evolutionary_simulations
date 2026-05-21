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