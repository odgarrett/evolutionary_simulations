import torch
import joblib
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mint.helpers.extract import load_config, MINTWrapper, CollateFn

MAX_SEQ = 1024 # Maximum sequence length before the collate function starts truncating

class MiniDataset(Dataset):
    '''
    An in-memory torch Dataset for rapid sequence loading.
    '''
    def __init__(self, seq1_list, seq2_list):
        self.seq1 = seq1_list
        self.seq2 = seq2_list

    def __len__(self):
        return len(self.seq1)

    def __getitem__(self, idx):
        return self.seq1[idx], self.seq2[idx]


class MINTScorer:
    '''
    A wrapper around SKEMPI_v2-finetuned MINT for zero-shot ddG predictions
    '''
    def __init__(
        self, 
        mint_config_path: str, 
        mint_weights_path: str, 
        regressor_path: str, 
        target_dict: dict,
        wt_sequence: str,
        device: str = None
    ):
        self.device = device if device else ('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Load MINT
        cfg = load_config(mint_config_path)
        self.wrapper = MINTWrapper(cfg, mint_weights_path, sep_chains=False, device=self.device)
        self.wrapper.eval()

        # Load the regression head
        self.predictor = joblib.load(regressor_path)

        # Initialize the WT embedding cache
        self.target_dict = target_dict
        self.wt_sequence = wt_sequence
        self.wt_cache = {}
        self._set_wt_baselines()
        
        print(f"MINT initialized on {self.device}")

    def _set_wt_baselines(self, batch_size: int = 1):
        '''
        Calculates and caches the embeddings for the WT complexes.
        Call this just once at the start of the simulation from the Controller.

        Args:
            target_dict (dict): Dictionary mapping {target_name: static_target_sequence}.
            wt_evolving_seq (str): The base sequence of the protein being evolved.
            batch_size (int): DataLoader batch size. Adjust as needed for your GPU's VRAM.
        '''
        target_names = list(self.target_dict.keys())
        target_seqs = list(self.target_dict.values())

        print(f"Caching embeddings for WT complexed with {target_names}...")

        # Initialize in-memory dataset and dataloader
        dataset = MiniDataset(target_seqs, [self.wt_sequence]*len(target_seqs))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=CollateFn(MAX_SEQ))

        # Get embeddings
        wt_embs = []
        with torch.inference_mode():
            for chains, chain_ids in tqdm(loader, desc="Caching WT embeddings"):
                chains = chains.to(self.device)
                chain_ids = chain_ids.to(self.device)
                embs = self.wrapper(chains, chain_ids)
                wt_embs.append(embs.cpu())
        
        # Remove batch dimension
        wt_embs = torch.cat(wt_embs, dim=0)

        # Store in a dictionary keyed by the target name for lookup
        for name, emb in zip(target_names, wt_embs):
            self.wt_cache[name] = emb

    def score_mutants(self, target_names: list, mutant_seqs: list, batch_size: int = 1) -> list:
        '''
        Takes a batch of mutant sequences and their corresponding targets.
        Calculates the mutant embeddings, fetches the cached WT embeddings, and predicts the phenotypic score.

        Args:
            target_names (list): List of target names corresponding to each mutant.
            mutant_seqs (list): List of mutant sequences to score.
            target_dict (dict): Dictionary mapping {target_name: static_target_sequence}.
            batch_size (int): DataLoader batch size. Adjust as needed for your GPU's VRAM.

        Returns:
            list: The predicted phenotypic scores (e.g. ddG).
        '''
        if len(target_names) != len(mutant_seqs):
            raise ValueError("Target names list and mutant sequences list must be the same length.")

        # Retrieve target seqs
        target_seqs = [self.target_dict[name] for name in target_names]

        # Initialize in-memory dataset and dataloader
        dataset = MiniDataset(target_seqs, mutant_seqs)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=CollateFn(MAX_SEQ))

        # Get mutant embeddings
        mut_embs = []
        with torch.inference_mode():
            for chains, chain_ids in tqdm(loader, desc=f"Scoring novel variants"):
                chains = chains.to(self.device)
                chain_ids = chain_ids.to(self.device)
                embs = self.wrapper(chains, chain_ids)
                mut_embs.append(embs.cpu())

        # Remove batch dimension
        mut_embs = torch.cat(mut_embs, dim=0)

        # Pair each mutant embedding with its respective WT embedding
        paired_wt_embs = torch.stack([self.wt_cache[name] for name in target_names])

        # Compute feature vectors by subtracting mutant from wt embeddings
        feature_vectors = (paired_wt_embs - mut_embs).numpy()

        # Make ddG predictions
        predicted_scores = self.predictor.predict(feature_vectors)
        return predicted_scores.tolist()