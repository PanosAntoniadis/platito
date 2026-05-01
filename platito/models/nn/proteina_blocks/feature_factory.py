# Code based on original Proteina code:
# github.com/NVIDIA-Digital-Bio/proteina/

from typing import Dict, List, Literal

import torch


from platito.models.nn.proteina_blocks.ff_utils import (
    get_time_embedding,
    get_index_embedding,
)
from platito.utils.amino_acid_vocab import MAX_RES_ID


# ################################
# # # Some auxiliary functions # #
# ################################


def bin_pairwise_distances(x, min_dist, max_dist, dim):
    """
    Takes coordinates and bins the pairwise distances.

    Args:
        x: Coordinates of shape [b, n, 3]
        min_dist: Right limit of first bin
        max_dist: Left limit of last bin
        dim: Dimension of the final one hot vectors

    Returns:
        Tensor of shape [b, n, n, dim] consisting of one-hot vectors
    """
    pair_dists_nm = torch.norm(
        x[:, :, None, :] - x[:, None, :, :], dim=-1
    )  # [b, n, n]
    bin_limits = torch.linspace(
        min_dist, max_dist, dim - 1, device=x.device
    )  # Open left and right
    return bin_and_one_hot(
        pair_dists_nm, bin_limits
    )  # [b, n, n, pair_dist_dim]


def bin_and_one_hot(tensor, bin_limits):
    """
    Converts a tensor of shape [*] to a tensor of shape [*, d]
    using the given bin limits.

    Args:
        tensor (Tensor): Input tensor of shape [*]
        bin_limits (Tensor): bin limits [l1, l2, ..., l_{d-1}]. d-1 limits
            define d-2 bins, and the first one is <l1, the last one is >l_{d-1},
            giving a total of d bins.

    Returns:
        torch.Tensor: Output tensor of shape [*, d] where d = len(bin_limits) + 1
    """
    bin_indices = torch.bucketize(tensor, bin_limits)
    return torch.nn.functional.one_hot(bin_indices, len(bin_limits) + 1) * 1.0


def indices_force_start_w_one(pdb_idx, mask):
    """
    Takes a tensor with pdb indices for a batch and forces them all to start
    with the index 1. Masked elements are still assigned the index -1.

    Args:
        pdb_idx: tensor of increasing integers (except masked ones fixed to -1), shape [b, n]
        mask: binary tensor, shape [b, n]

    Returns:
        pdb_idx but now all rows start at 1, masked elements are still set to -1.
    """
    first_val = pdb_idx[:, 0][:, None]  # min val is the first one
    pdb_idx = pdb_idx - first_val + 1
    pdb_idx = torch.masked_fill(
        pdb_idx, ~mask, -1
    )  # set masked elements to -1
    return pdb_idx


################################
# # Classes for each feature # #
################################


class Feature(torch.nn.Module):
    """Base class for features."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def get_dim(self):
        return self.dim

    def forward(self, batch: Dict):
        pass  # Implemented by each class


class ZeroFeat(Feature):
    """Computes empty feature (zero) of shape [b, n, dim] or [b, n, n, dim],
    depending on sequence or pair features."""

    def __init__(self, dim_feats_out, mode: Literal["seq", "pair"]):
        super().__init__(dim=dim_feats_out)
        self.mode = mode

    def forward(self, batch):
        xt = batch["x_t"]  # [b, n, 3]
        b, n = xt.shape[0], xt.shape[1]
        if self.mode == "seq":
            return torch.zeros((b, n, self.dim), device=xt.device)
        elif self.mode == "pair":
            return torch.zeros((b, n, n, self.dim), device=xt.device)
        else:
            raise IOError(f"Mode {self.mode} wrong for zero feature")


class TimeEmbeddingSeqFeat(Feature):
    """Computes time embedding and returns as sequence feature of shape [b, n, t_emb_dim]."""

    def __init__(self, t_emb_dim, **kwargs):
        super().__init__(dim=t_emb_dim)

    def forward(self, batch):
        t = batch["t"]  # [b]
        xt = batch["x_t"]  # [b, n, 3]
        n = xt.shape[1]
        t_emb = get_time_embedding(t, edim=self.dim)  # [b, t_emb_dim]
        t_emb = t_emb[:, None, :]  # [b, 1, t_emb_dim]
        return t_emb.expand(
            (t_emb.shape[0], n, t_emb.shape[2])
        )  # [b, n, t_emb_dim]


class TimeEmbeddingPairFeat(Feature):
    """Computes time embedding and returns as pair feature of shape [b, n, n, t_emb_dim]."""

    def __init__(self, t_emb_dim, **kwargs):
        super().__init__(dim=t_emb_dim)

    def forward(self, batch):
        t = batch["t"]  # [b]
        xt = batch["x_t"]  # [b, n, 3]
        n = xt.shape[1]
        t_emb = get_time_embedding(t, edim=self.dim)  # [b, t_emb_dim]
        t_emb = t_emb[:, None, None, :]  # [b, 1, 1, t_emb_dim]
        return t_emb.expand(
            (t_emb.shape[0], n, n, t_emb.shape[3])
        )  # [b, n, t_emb_dim]


class IdxEmbeddingSeqFeat(Feature):
    """Computes index embedding and returns sequence feature of shape [b, n, idx_emb]."""

    def __init__(self, idx_emb_dim, **kwargs):
        super().__init__(dim=idx_emb_dim)

    def forward(self, batch):
        # If it has the actual residue indices
        if "residue_pdb_idx" in batch:
            inds = batch["residue_pdb_idx"]  # [b, n]
            inds = indices_force_start_w_one(inds, batch["mask"])
        else:
            xt = batch["x_t"]  # [b, n, 3]
            b, n = xt.shape[0], xt.shape[1]
            inds = torch.Tensor(
                [[i + 1 for i in range(n)] for _ in range(b)]
            ).to(
                xt.device
            )  # [b, n]
        return get_index_embedding(
            inds, edim=self.dim
        )  # [b, n, idx_embed_dim]


class CondSeqFeat(Feature):
    def __init__(self, cond_emb_dim, **kwargs):
        super().__init__(dim=cond_emb_dim)

    def forward(self, batch):
        if "cond_emb" in batch:
            assert batch["cond_emb"].shape[-1] == self.dim
            return batch["cond_emb"]  # [b, n, cond_emb_dim]
        else:
            raise ValueError("No cond_emb feature provided.")


class EsmSeqFeat(Feature):
    def __init__(self, sequence_emb_dim, **kwargs):
        super().__init__(dim=sequence_emb_dim)

    def forward(self, batch):
        if "sequence_emb" in batch:
            return batch["sequence_emb"]  # [b, n, D]
        else:
            raise ValueError("No sequence_emb feature provided.")


class ResidueIdSeqFeat(Feature):
    def __init__(self, res_emb_dim, **kwargs):
        super().__init__(dim=res_emb_dim)
        self.res_id_emb = torch.nn.Embedding(
            num_embeddings=MAX_RES_ID, embedding_dim=res_emb_dim
        )

    def forward(self, batch):
        if "residue_ids" in batch:
            return self.res_id_emb(batch["residue_ids"])
        else:
            raise ValueError("No residue id feature provided.")


class LagEmbeddingSeqFeat(Feature):
    def __init__(self, lag_emb_dim, max_lag, **kwargs):
        super().__init__(dim=lag_emb_dim)
        self.max_lag = max_lag  # Maximum lag value
        self.lag_emb_max_positions = kwargs.get(
            "lag_emb_max_positions", max_lag
        )

    def forward(self, batch):
        lag = batch["lag"] / self.max_lag  # [b]
        xt = batch["x_t"]  # [b, n, 3]
        n = xt.shape[1]
        lag_emb = get_time_embedding(
            lag, edim=self.dim, max_positions=self.lag_emb_max_positions
        )  # [b, lag_emb_dim]
        lag_emb = lag_emb[:, None, :]  # [b, 1, lag_emb_dim]
        return lag_emb.expand(
            (lag_emb.shape[0], n, lag_emb.shape[2])
        )  # [b, n, lag_emb_dim]


class TempEmbeddingSeqFeat(Feature):
    def __init__(
        self,
        temp_emb_dim,
        temp_max,
        temp_min,
        **kwargs,
    ):
        super().__init__(dim=temp_emb_dim)
        self.temp_max = temp_max
        self.temp_min = temp_min
        self.temp_emb_max_positions = kwargs.get("temp_emb_max_positions", 5)

    def forward(self, batch):
        temp = batch["temp"]  # [b]
        temp_norm = (temp - self.temp_min) / (self.temp_max - self.temp_min)
        xt = batch["x_t"]  # [b, n, 3]
        n = xt.shape[1]
        temp_emb = get_time_embedding(
            temp_norm, edim=self.dim, max_positions=self.temp_emb_max_positions
        )  # [b, temp_emb_dim]
        temp_emb = temp_emb[:, None, :]  # [b, 1, temp_emb_dim]
        return temp_emb.expand(
            (temp_emb.shape[0], n, temp_emb.shape[2])
        )  # [b, n, temp_emb_dim]


class ConnectivityPairFeat(Feature):
    """Encodes pairwise residue connectivity as a one-hot pair feature.

    Each residue pair (i, j) is assigned one of four edge types:
        0 — self-edge           (i == j)
        1 — sequential neighbor (|i - j| == 1)
        2 — spatial neighbor    (distance < connectivity_cutoff, non-self, non-sequential)
        3 — no connection       (all other pairs)

    Args:
        connectivity_dim: Number of edge-type classes. Must be 4. Default: 4.
        connectivity_cutoff: Distance threshold (nm) for spatial neighbors. Default: 0.8.

    Returns:
        Pair feature tensor of shape [b, n, n, connectivity_dim].
    """

    def __init__(self, connectivity_dim=4, connectivity_cutoff=0.8, **kwargs):
        super().__init__(dim=connectivity_dim)
        self.connectivity_cutoff = connectivity_cutoff

    def forward(self, batch):
        x = batch["x_t"]  # [b, n, 3]
        b, n, _ = x.shape
        device = x.device

        edge_type = torch.full(
            (b, n, n), 3, dtype=torch.long, device=device
        )  # [b, n, n], default class 3 = no connection

        # Class 0: self-edges (i == j)
        eye = torch.eye(n, device=device).bool()
        edge_type[:, eye] = 0

        # Class 1: sequential neighbors (|i - j| == 1)
        idx = torch.arange(n, device=device)
        seq_mask = (idx[None] - idx[:, None]).abs() == 1  # [n, n]
        edge_type[:, seq_mask] = 1

        # Class 2: spatial neighbors within cutoff (non-self, non-sequential)
        dist = torch.cdist(x, x)  # [b, n, n]
        spatial_mask = (dist < self.connectivity_cutoff) & ~eye & ~seq_mask
        edge_type[spatial_mask] = 2

        return torch.nn.functional.one_hot(
            edge_type, num_classes=self.dim
        ).float()  # [b, n, n, connectivity_dim]


class SequenceSeparationPairFeat(Feature):
    """Computes sequence separation and returns feature of shape [b, n, n, seq_sep_dim]."""

    def __init__(self, seq_sep_dim, **kwargs):
        super().__init__(dim=seq_sep_dim)

    def forward(self, batch):
        if "residue_pdb_idx" in batch:
            # no need to force 1 since taking difference
            inds = batch["residue_pdb_idx"]  # [b, n]
        else:
            xt = batch["x_t"]  # [b, n, 3]
            b, n = xt.shape[0], xt.shape[1]
            inds = torch.Tensor(
                [[i + 1 for i in range(n)] for _ in range(b)]
            ).to(
                xt.device
            )  # [b, n]

        seq_sep = inds[:, :, None] - inds[:, None, :]  # [b, n, n]

        # Dimension should be odd, bins limits [-(dim/2-1), ..., -1.5, -0.5, 0.5, 1.5, ..., dim/2-1]
        # gives dim-2 bins, and the first and last for values beyond the bin limits
        assert (
            self.dim % 2 == 1
        ), "Relative seq separation feature dimension must be odd and > 3"

        # Create bins limits [..., -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.3, 3.5, ...]
        # Equivalent to binning relative sequence separation
        low = -(self.dim / 2.0 - 1)
        high = self.dim / 2.0 - 1
        bin_limits = torch.linspace(
            low, high, self.dim - 1, device=inds.device
        )

        return bin_and_one_hot(seq_sep, bin_limits)  # [b, n, n, seq_sep_dim]


class XtPairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(
        self, xt_pair_dist_dim, xt_pair_dist_min, xt_pair_dist_max, **kwargs
    ):
        super().__init__(dim=xt_pair_dist_dim)
        self.min_dist = xt_pair_dist_min
        self.max_dist = xt_pair_dist_max

    def forward(self, batch):
        return bin_pairwise_distances(
            x=batch["x_t"],
            min_dist=self.min_dist,
            max_dist=self.max_dist,
            dim=self.dim,
        )  # [b, n, n, pair_dist_dim]


####################################
# # Class that produces features # #
####################################


class FeatureFactory(torch.nn.Module):
    def __init__(
        self,
        feats: List[str],
        dim_feats_out: int,
        use_ln_out: bool,
        mode: Literal["seq", "pair"],
        **kwargs,
    ):
        """
        Sequence features:
            - "res_seq_pdb_idx"  -> sinusoidal position index embedding
            - "time_emb"         -> sinusoidal flow-matching time embedding
            - "cond_seq_feat"    -> conditioning embedding from the condition network
            - "seq_emb_esm3"     -> pre-computed ESM sequence embedding
            - "res_seq_id"       -> learned amino acid type embedding
            - "lag_emb"          -> sinusoidal lag time embedding
            - "temp_emb"         -> sinusoidal temperature embedding

        Pair features:
            - "xt_pair_dists"      -> binned pairwise CA distances
            - "rel_seq_sep"        -> binned relative sequence separation
            - "time_emb"           -> sinusoidal flow-matching time embedding
            - "connectivity_pair"  -> one-hot edge type (self / sequential / spatial / none)
        """
        super().__init__()
        self.mode = mode

        self.ret_zero = True if (feats is None or len(feats) == 0) else False
        if self.ret_zero:
            self.zero_creator = ZeroFeat(
                dim_feats_out=dim_feats_out, mode=mode
            )
            return

        self.feat_creators = torch.nn.ModuleList(
            [self.get_creator(f, **kwargs) for f in feats]
        )
        self.ln_out = (
            torch.nn.LayerNorm(dim_feats_out)
            if use_ln_out
            else torch.nn.Identity()
        )
        self.linear_out = torch.nn.Linear(
            sum([c.get_dim() for c in self.feat_creators]),
            dim_feats_out,
            bias=False,
        )

    def get_creator(self, f, **kwargs):
        """Returns the right class for the requested feature f (a string)."""

        if self.mode == "seq":
            if f == "time_emb":
                return TimeEmbeddingSeqFeat(**kwargs)
            elif f == "res_seq_pdb_idx":
                return IdxEmbeddingSeqFeat(**kwargs)
            elif f == "cond_seq_feat":
                return CondSeqFeat(**kwargs)
            elif f == "res_seq_id":
                return ResidueIdSeqFeat(**kwargs)
            elif f == "seq_emb_esm3":
                return EsmSeqFeat(**kwargs)
            elif f == "lag_emb":
                return LagEmbeddingSeqFeat(**kwargs)
            elif f == "temp_emb":
                return TempEmbeddingSeqFeat(**kwargs)
            else:
                raise IOError(f"Sequence feature {f} not implemented.")

        elif self.mode == "pair":
            if f == "xt_pair_dists":
                return XtPairwiseDistancesPairFeat(**kwargs)
            elif f == "rel_seq_sep":
                return SequenceSeparationPairFeat(**kwargs)
            elif f == "time_emb":
                return TimeEmbeddingPairFeat(**kwargs)
            elif f == "connectivity_pair":
                return ConnectivityPairFeat(**kwargs)
            else:
                raise IOError(f"Pair feature {f} not implemented.")

        else:
            raise IOError(
                f"Wrong feature mode (creator): {self.mode}. Should be 'seq' or 'pair'."
            )

    def apply_padding_mask(self, feature_tensor, mask):
        """
        Applies mask to features.

        Args:
            feature_tensor: tensor with requested features, shape [b, n, d] of
                [b, n, n, d] depending on self.mode ('seq' or 'pair')
            mask: Binary mask, shape [b, n]

        Returns:
            Masked features, same shape as input tensor.
        """
        if self.mode == "seq":
            return feature_tensor * mask[..., None]  # [b, n, d]
        elif self.mode == "pair":
            mask_pair = mask[:, None, :] * mask[:, :, None]  # [b, n, n]
            return feature_tensor * mask_pair[..., None]  # [b, n, n, d]
        else:
            raise IOError(
                f"Wrong feature mode (pad mask): {self.mode}. Should be 'seq' or 'pair'."
            )

    def forward(self, batch):
        """Returns masked features, shape depends on mode, either 'seq' or 'pair'."""
        # If no features requested just return the zero tensor of
        # appropriate dimensions
        if self.ret_zero:
            return self.zero_creator(batch)

        # Compute requested features
        feature_tensors = []
        for fcreator in self.feat_creators:
            feature_tensors.append(
                fcreator(batch)
            )  # [b, n, dim_f] or [b, n, n, dim_f] if seq or pair mode

        # Concatenate features and mask
        features = torch.cat(
            feature_tensors, dim=-1
        )  # [b, n, dim_f] or [b, n, n, dim_f]
        features = self.apply_padding_mask(
            features, batch["mask"]
        )  # [b, n, dim_f] or [b, n, n, dim_f]

        # Linear layer and mask
        features_proc = self.ln_out(
            self.linear_out(features)
        )  # [b, n, dim_f] or [b, n, n, dim_f]
        return self.apply_padding_mask(
            features_proc, batch["mask"]
        )  # [b, n, dim_f] or [b, n, n, dim_f]
