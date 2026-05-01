# Code based on original Proteina code:
# github.com/NVIDIA-Digital-Bio/proteina/blob/main/proteinfoundation/flow_matching/r3n_fm.py

import torch
import einops

from torch import Tensor
from typing import Optional


class R3NFlowMatcher:
    def __init__(
        self,
        sigma: float = 0.1,
        scale_ref: float = 1.0,
    ):
        self.dim = 3
        self.scale_ref = scale_ref
        self.sigma = sigma

    def _mean_w_mask(self, a, mask, keepdim=True):
        """
        Computes the mean of point cloud accounting for the mask.

        Args:
            a: Input point cloud of shape [*, n, d]
            mask: Input mask of shape [*, n] of boolean values
            keepdim: whether to keep the dimension across which
                we're computing the mean like normal pytorch mean

        Returns:
            Masked mean of a across dimension -2 (or n)
        """
        mask = mask[..., None]  # [*, n, 1]
        num_elements = torch.sum(mask, dim=-2, keepdim=True)  # [*, 1, 1]
        num_elements = torch.where(
            num_elements == 0, torch.tensor(1.0), num_elements
        )  # [*, 1, 1]
        a_masked = torch.masked_fill(a, ~mask, 0.0)  # [*, n, d]
        mean = (
            torch.sum(a_masked, dim=-2, keepdim=True) / num_elements
        )  # [*, 1, d]
        mean = torch.masked_fill(mean, num_elements == 0, 0.0)  # [*, 1, d]
        if not keepdim:
            mean = einops.rearrange(mean, "... () d -> ... d")
        return mean

    def _force_zero_com(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Centers tensor over n dimension.

        Args:
            x: Tensor of shape [*, n, 3]
            mask (optional): Binary mask of shape [*, n]

        Returns:
            Centered x = x - mean(x, dim=-2), shape [*, n, 3].
        """
        if mask is None:
            x = x - torch.mean(x, dim=-2, keepdim=True)
        else:
            x = (x - self._mean_w_mask(x, mask, keepdim=True)) * mask[
                ..., None
            ]
        return x

    def _apply_mask(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Applies mask to x. Sets masked elements to zero.

        Args:
            x: Tensor of shape [*, n, 3]
            mask (optional): Binary mask of shape [*, n]

        Returns:
            Masked x of shape [*, n, 3]
        """
        if mask is None:
            return x
        return x * mask[..., None]  # [*, n, 3]

    def _mask_and_zero_com(self, x, mask: Optional[Tensor] = None) -> Tensor:
        """
        Applies mask to and centers x if needed (if zero_com=True).

        Args:
            x: Batch of samples, batch shape *
            mask (optional): Binary mask of shape [*, n]

        Returns:
            Masked (and possibly center) samples.
        """
        x = self._apply_mask(x, mask)
        x = self._force_zero_com(x, mask)
        return x

    def _extend_t(self, n: int, t: Tensor) -> Tensor:
        """
        Extends t shape with n. Needed to use flow matching utils.

        Args:
            n (int): Number of elements per sample (e.g. number of residues)
            t: Float vector, shape [*]

        Returns:
            Extended t vector of shape [*, n] compatible with flow
            matching utils.
        """
        return t[..., None].expand(t.shape + (n,))

    def sample_noise(
        self,
        n: int,
        b: int,
        device: Optional[torch.device] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Samples noise distribution std Gaussian (possibly centered).

        Args:
            n: number of frames in a single sample
            b: number of samples
            device (optional): torch device used
            mask (optional): Binary mask of shape [*, n]

        Returns:
            Samples from refenrece [N(0, I_3)]^n shape [*shape, n, 3]
        """
        x = (
            torch.randn(
                (b, n, self.dim),
                device=device,
            )
            * self.scale_ref
        )
        return self._mask_and_zero_com(x, mask)

    def interpolate(
        self,
        x0: Tensor,
        x1: Tensor,
        t: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Interpolates between rigids x_0 (base) and x_1 (data) using t.

        Args:
            x0: Tensor sampled from reference, shape [*, n, 3]
            x1: Tensor sampled from target, shape [*, n, 3]
            t: Interpolation times, shape [*]
            mask (optional): Binary mask, shape [*, n]

        Returns:
            x_t: Interpolated tensor, shape [*, n, 3]
        """
        x0, x1 = map(
            lambda args: self._mask_and_zero_com(*args),
            ((x0, mask), (x1, mask)),
        )

        n = x0.shape[-2]
        t = self._extend_t(n, t)  # [*, n]
        t = t[..., None]  # [*, n, 1]
        mu_t = (1.0 - t) * x0 + t * x1
        eps_t = torch.randn_like(mu_t)
        eps_t = self._mask_and_zero_com(eps_t, mask)
        x = mu_t + self.sigma * eps_t
        return x, mu_t, eps_t
