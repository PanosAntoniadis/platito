import torch
import numpy as np

from typing import Optional
from dataclasses import dataclass


@dataclass
class FrameData:
    """A single training sample: a pair of backbone frames.

    Attributes:
        id: Domain identifier.
        x0: C_{\alpha} coordinates at time t, shape [L, 3].
        xt: C_{\alpha} coordinates at time t + lag, shape [L, 3].
        lag: Lag time between x0 and xt, in number of frames (1 frame = 1 ns).
        temp: Simulation temperature in Kelvin.
        replica: Replica index from the MD simulation.
        mask: Boolean mask of valid residues, shape [L]. True = real residue,
            False = padding. Set by the collator; None at the single-sample level.
        residue_ids: Integer amino acid type indices, shape [L].
            Indices follow AA_1_TO_ID in platito/utils/amino_acid_vocab.py.
        sequence_emb: Pre-computed protein language model embeddings, shape [L, D].
    """

    id: str
    x0: torch.Tensor
    xt: torch.Tensor
    lag: int
    temp: int
    replica: Optional[int] = 0
    mask: Optional[torch.Tensor] = None
    residue_ids: Optional[torch.Tensor] = None
    sequence_emb: Optional[torch.Tensor] = None


class FrameDataCollator:
    def __init__(self, pad_to=None):
        """
        Collator for FrameData objects.
        Args:
            pad_to: int or None. If set, pad to this length. If None, pad to
                max length in batch.
        """
        self.pad_to = pad_to

    def __call__(self, batch):
        """
        Pads all tensors in the batch to the same length (max in batch or
        pad_to), and returns a mask indicating non-padded regions.
        Args:
            batch: list of FrameData objects
        Returns:
            dict with padded tensors and a 'mask' key (bool tensor, True for
            real, False for pad).
        """
        lengths = [item.x0.shape[0] for item in batch]
        max_len = max(lengths) if self.pad_to is None else self.pad_to

        def pad_tensor(t, pad_value=0):
            if t is None:
                return None
            pad_size = [max_len - t.shape[0]] + list(t.shape[1:])
            if pad_size[0] == 0:
                return t
            padding = torch.full(
                pad_size, pad_value, dtype=t.dtype, device=t.device
            )
            return torch.cat([t, padding], dim=0)

        tensor_fields = [
            "x0",
            "xt",
            "residue_ids",
            "sequence_emb",
        ]
        scalar_fields = [
            "lag",
            "temp",
            "replica",
        ]
        list_fields = [
            "id",
        ]

        out = {}
        for field in tensor_fields:
            values = [getattr(item, field) for item in batch]
            if all(v is not None for v in values):
                out[field] = torch.stack([pad_tensor(v) for v in values])

        for field in scalar_fields:
            values = [getattr(item, field) for item in batch]
            if all(v is not None for v in values):
                out[field] = torch.tensor(values)

        for field in list_fields:
            values = [getattr(item, field) for item in batch]
            if all(v is not None for v in values):
                out[field] = np.array(values)

        mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
        for i, l in enumerate(lengths):
            mask[i, :l] = True
        out["mask"] = mask

        return out
