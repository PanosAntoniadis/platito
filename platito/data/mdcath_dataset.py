import os
import h5py
import torch
import random
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset

from platito.data import FrameData
from platito.utils.amino_acid_vocab import AA_1_TO_ID, AA_3_TO_1
from platito.data.constants import (
    MDCATH_ALL_TEMPS,
    MDCATH_FRAME_SPACING,
)


class MDCATH(Dataset):
    """PyTorch Dataset for training a coarse-grained ITO model on mdCATH.

    Loads all C_{\alpha} coordinates from HDF5 files into memory during init and
    samples random (x0, xt) frame pairs during data loading.
    The effective dataset size is controlled by `samples_per_epoch` rather
    than the number of stored frames.

    Args:
        dataset_path: Root directory of the mdCATH dataset. Expected layout:
            <dataset_path>/data/mdcath_dataset_<domain>.h5
            <dataset_path>/embeddings/<seq_emb_name>.pt
        protein_names: List of domain identifiers to load.
        seq_emb_name: Filename of the pre-computed sequence embedding file
            inside <dataset_path>/embeddings/ (e.g. "esmc_6b.pt").
            The file should be a dict mapping domain name -> tensor [L, D].
        temperatures: List of temperature strings to use (e.g. ["320", "450"]).
            Defaults to all five mdCATH temperatures.
        max_lag: Maximum lag time in ns. Lag is sampled uniformly in
            [1, min(max_lag, available_frames - idx0 - 1)].
        samples_per_epoch: Number of samples returned per epoch (controls
            how many gradient steps are taken per epoch).
    """

    def __init__(
        self,
        dataset_path,
        protein_names,
        seq_emb_name,
        temperatures: list = MDCATH_ALL_TEMPS,
        max_lag: int = 200,  # in ns
        samples_per_epoch: int = 10000,
    ):
        super().__init__()

        self.dataset_path = dataset_path
        self.protein_names = protein_names
        self.seq_emb_name = seq_emb_name
        self.temperatures = temperatures
        self.max_lag = max_lag
        self.samples_per_epoch = samples_per_epoch

        # Convert max lag from ns to frames
        self.max_lag_frames = self.max_lag // MDCATH_FRAME_SPACING

        # Load pre-computed sequence embeddings: dict {domain -> tensor [L, D]}
        self.seq_emb_per_domain = torch.load(
            os.path.join(
                dataset_path, "embeddings", self.seq_emb_name + ".pt"
            ),
            weights_only=False,
        )

        # Load all CA coordinates into memory:
        # {domain -> {temp -> {repl -> tensor [T, L, 3]}}}
        # Coordinates are converted from Å (HDF5) to nm (divide by 10).
        self.all_coords = {}
        self.sequences = {}
        for protein_name in tqdm(self.protein_names, desc="Loading mdCATH"):
            h5_path = f"{dataset_path}/data/mdcath_dataset_{protein_name}.h5"
            self.all_coords[protein_name] = {}
            with h5py.File(h5_path, "r") as f:
                grp = f[protein_name]
                resname = grp["resname"][:]
                resid = grp["resid"][:]
                _, idx = np.unique(resid, return_index=True)
                self.sequences[protein_name] = [
                    AA_3_TO_1[r.decode()] for r in resname[np.sort(idx)]
                ]
                for temp in self.temperatures:
                    self.all_coords[protein_name][temp] = {}
                    for repl in grp[temp].keys():
                        coords_all = grp[temp][repl]["ca_coords"][:]
                        coords = torch.from_numpy(np.array(coords_all)) / 10.0
                        self.all_coords[protein_name][temp][repl] = coords

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Sample a random protein, temperature, and replica
        protein = random.choice(self.protein_names)
        temp = random.choice(self.temperatures)
        repl = random.choice(list(self.all_coords[protein][temp].keys()))
        n_frames = self.all_coords[protein][temp][repl].shape[0]

        # used by baseline TITO model
        residue_ids = [AA_1_TO_ID[aa] for aa in self.sequences[protein]]

        # Sample a random starting frame and lag within the valid range
        idx0 = random.randint(0, n_frames - 2)
        max_valid_lag = min(n_frames - idx0 - 1, self.max_lag_frames)
        lag = random.randint(1, max_valid_lag)

        x0 = self.all_coords[protein][temp][repl][idx0]
        xt = self.all_coords[protein][temp][repl][idx0 + lag]

        return FrameData(
            id=protein,
            x0=x0,
            xt=xt,
            lag=lag,
            temp=torch.tensor(int(temp), dtype=torch.long),
            residue_ids=torch.tensor(residue_ids, dtype=torch.long),
            replica=torch.tensor(int(repl), dtype=torch.long),
            sequence_emb=self.seq_emb_per_domain[protein],
        )
