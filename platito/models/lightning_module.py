"""
PLaTITO: Protein Language-aware Transferable Implicit Transfer Operator.

Implements a conditional flow matching generative model that approximates the
long-timescale transition density p(x_{t+Δt} | x_t, S, T, Δt) for protein
molecular dynamics, where x_t are C_{\alpha} backbone coordinates, S is the
amino-acid sequence, T is the simulation temperature, and Δt is the physical
lag time.

The model follows a two-stage architecture (Figure 1 of the paper):
  1. A conditioning network f_c encodes the current state (x_t, S, T, Δt) into
     a per-residue representation c.
  2. A velocity network f_v predicts the flow-matching velocity field given the
     interpolated sample z_s and the condition embedding c.

To generate long trajectories, we iteratively sample from the learned
transition operator.
"""

import torch
import lightning as L

from tqdm import tqdm
from typing import Union, Optional

from platito.models.fm.r3n_fm import R3NFlowMatcher


class PLaTITO(L.LightningModule):
    """Lightning module for PLaTITO training and inference.

    Args:
        fm: Handles CFM framework for molecules (noise sampling, interpolation
            and center-of-mass centering in R^{3L}).
        velocity_net: Neural network f_v that predicts the velocity field
            v^θ(z_s; s, c) at flow-matching time s given conditioning c.
        condition_net: Neural network f_c that encodes the current molecular
            state into a per-residue conditioning representation c.
        optimizer: Optimizer constructor (receives model parameters).
        ode_solver: ODE solver used during sampling.
        lr_scheduler: Optional learning-rate scheduler.
    """

    def __init__(
        self,
        fm: R3NFlowMatcher,
        velocity_net: torch.nn.Module,
        condition_net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        ode_solver: torch.nn.Module,
        lr_scheduler: Union[torch.optim.lr_scheduler.LRScheduler, None] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.fm = fm
        self.velocity_net = velocity_net
        self.condition_net = condition_net
        self.ode_solver = ode_solver(
            velocity_model=self.velocity_net, fm=self.fm
        )

    def get_velocity_field(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> torch.Tensor:
        """Target (rectified flow) velocity field: v = x1 - x0."""
        return x1 - x0

    def model_step(self, batch: dict) -> dict:
        """Single forward pass: interpolate, predict velocity, compute loss.

        Args:
            batch: Dictionary with keys:
                - "x0": current backbone coordinates [B, L, 3]
                - "xt": future backbone coordinates at lag Δt [B, L, 3]
                - "mask": boolean padding mask [B, L]
                - plus any conditioning keys (lag, temp, sequence_emb, ...)

        Returns:
            Dictionary with scalar "loss".
        """
        if "mask" not in batch:
            mask = torch.ones(batch["x0"].shape[:2], dtype=torch.bool).to(
                batch["x0"].device
            )
        else:
            mask = batch["mask"]

        # Center-of-mass centering and masking
        x0 = self.fm._mask_and_zero_com(batch["x0"], mask=mask)
        xt = self.fm._mask_and_zero_com(batch["xt"], mask=mask)

        # Sample flow-matching time s ~ U(0,1) and Gaussian noise z0 ~ N(0,I)
        s = torch.rand(xt.shape[:-2], device=xt.device)
        z0 = self.fm.sample_noise(
            n=xt.shape[1],
            b=xt.shape[0],
            device=xt.device,
            mask=mask,
        )

        rest_conditions = {
            key: value
            for key, value in batch.items()
            if key not in ["xt", "x0", "mask"]
        }

        # Interpolate: z_s = s * x_t + (1-s) * z0  (rectified flow)
        zs, _, _ = self.fm.interpolate(x0=z0, x1=xt, t=s, mask=mask)
        us = self.get_velocity_field(x0=z0, x1=xt)

        # Encode conditioning: c = f_c(x0, Δt, S, T)
        cond_emb = self.condition_net(
            {
                "x_t": x0,
                "mask": mask,
                **rest_conditions,
            }
        )["out_feat"]

        # Predict velocity: v^θ(z_s; s, c)
        nn_out = self.velocity_net(
            {
                "x_t": zs,
                "t": s,
                "mask": mask,
                "cond_emb": cond_emb,
                **rest_conditions,
            }
        )

        vs = nn_out["coors_pred"]
        vs = self.fm._mask_and_zero_com(vs, mask=mask)
        loss = self.compute_fm_loss(vs, us, mask)
        return {"loss": loss}

    def training_step(self, batch: dict) -> torch.Tensor:
        results = self.model_step(batch)
        self.log_dict(
            {"train/loss": results["loss"]},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["x0"].shape[0],
            sync_dist=True,
        )
        return results["loss"]

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        """Dummy validation step to ensure the validation loop runs."""
        return 0

    def sample(
        self,
        x0: torch.Tensor,
        ode_steps: int = 50,
        ode_method: str = "euler",
        mask: Optional[torch.Tensor] = None,
        **rest_conditions,
    ) -> torch.Tensor:
        """Sample x_{t+Δt} from the learned transition density p(x_{t+Δt} | x_t, ...).

        Implements Algorithm 2: encodes x0 into conditioning c, then integrates
        the learned ODE from Gaussian noise to the predicted future conformation.

        Args:
            x0: Current backbone coordinates [B, L, 3].
            ode_steps: Number of ODE integration steps.
            ode_method: ODE solver method (e.g. "euler", "rk4", "midpoint").
            mask: Boolean padding mask [B, L]. If None, all residues are valid.
            **rest_conditions: Additional conditioning tensors passed to both
                networks (e.g. lag, temp, sequence_emb, residue_ids).

        Returns:
            Predicted future coordinates x_{t+Δt} [B, L, 3] in nm.
        """
        if mask is None:
            mask = (
                torch.ones(x0.shape[0], x0.shape[1])
                .long()
                .bool()
                .to(self.device)
            )
        x0 = self.fm._mask_and_zero_com(x0, mask=mask)
        t_span = torch.linspace(0, 1, ode_steps + 1).to(self.device)
        x_noise = self.fm.sample_noise(
            n=x0.shape[1], b=x0.shape[0], device=self.device, mask=mask
        )

        with torch.no_grad():
            cond_emb = self.condition_net(
                {
                    "x_t": x0,
                    "mask": mask,
                    **rest_conditions,
                }
            )["out_feat"]
            extra_kwargs = {
                "mask": mask,
                "cond_emb": cond_emb,
                **rest_conditions,
            }
            x_pred = self.ode_solver.sample(
                x_init=x_noise,
                time_grid=t_span,
                method=ode_method,
                return_intermediates=False,
                **extra_kwargs,
            )
            x_pred = self.fm._mask_and_zero_com(x_pred, mask=mask)
            x_pred = x_pred.view(x_pred.shape[0], -1, 3)

        return x_pred

    def compute_fm_loss(
        self,
        vt: torch.Tensor,
        ut: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Conditional Flow Matching loss.

        Args:
            vt: Predicted velocity field [B, L, 3].
            ut: Target velocity field x1 - x0 [B, L, 3].
            mask: Boolean mask [B, L].

        Returns:
            Scalar mean loss.
        """
        nres = torch.sum(mask, dim=-1) * 3
        err = (vt - ut) * mask[..., None]
        loss = torch.sum(err**2, dim=(-1, -2)) / nres
        return loss.mean()

    def generate_trajectory(
        self,
        x0: torch.Tensor,
        trajectory_steps: int,
        ode_steps: int = 50,
        ode_method: str = "euler",
        mask: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        **rest_conditions,
    ) -> torch.Tensor:
        """Generate a long trajectory via iterative rollout (Algorithm 3).

        Repeatedly applies the transition operator x_{k+1} ~ p(·| x_k, ...)
        to produce a trajectory of physical duration trajectory_steps * Δt.

        Args:
            x0: Initial backbone coordinates [B, L, 3].
            trajectory_steps: Number of rollout steps K.
            ode_steps: ODE integration steps per transition sample.
            ode_method: ODE solver method.
            mask: Boolean padding mask [B, L].
            return_intermediates: If True, returns all K+1 frames stacked along
                dim=1; otherwise returns only the final frame.
            **rest_conditions: Conditioning tensors (lag, temp, sequence_emb, ...).

        Returns:
            If return_intermediates=True: tensor of shape [B, K+1, L, 3].
            Otherwise: tensor of shape [B, L, 3].
        """
        x0 = self.fm._mask_and_zero_com(x0, mask=mask)
        all_frames = [x0]

        for _ in tqdm(
            range(trajectory_steps), desc="Generating trajectory", unit="step"
        ):
            x_t = self.sample(
                x0=x0,
                **rest_conditions,
                ode_steps=ode_steps,
                ode_method=ode_method,
                mask=mask,
            )
            if return_intermediates:
                all_frames.append(x_t)
            x0 = x_t.clone()

        if return_intermediates:
            return torch.stack(all_frames, dim=1)
        else:
            return x_t

    def configure_optimizers(self):
        """Configure optimizer and optional learning-rate scheduler."""
        optimizer = self.hparams.optimizer(
            params=self.trainer.model.parameters()
        )
        if self.hparams.lr_scheduler is None:
            return {"optimizer": optimizer}

        scheduler = self.hparams.lr_scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",
            },
        }
