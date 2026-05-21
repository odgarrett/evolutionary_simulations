#!/bin/bash
# Be sure to give executable permission: chmod +x ./get_data.sh
wget https://github.com/jishnu-lab/SWING/raw/refs/heads/main/Data/MutInt_Model/Mutation_perturbation_model.csv
wget https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
gunzip uniprot_sprot.fasta.gz