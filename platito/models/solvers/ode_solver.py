import torch
import torch.nn as nn  # used for type hints

from torch import Tensor
from torchdiffeq import odeint_adjoint
from typing import Optional, Union, Sequence

from platito.models.fm.r3n_fm import R3NFlowMatcher


class _VelocityDynamics(nn.Module):
    """Wraps the velocity network as an ODE dynamics function for torchdiffeq.

    At each ODE step, evaluates the velocity field v^θ(x; t, c) and returns
    the time-derivative dx/dt = v^θ.
    """

    def __init__(
        self, velocity_model: nn.Module, fm: R3NFlowMatcher, model_extras: dict
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.fm = fm
        self.model_extras = model_extras

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        # torchdiffeq passes t as a scalar; repeat it across the batch
        t_batch = t.repeat_interleave(x.shape[0])
        vt_input = {"x_t": x, "t": t_batch, **self.model_extras}
        ut = self.velocity_model(vt_input)["coors_pred"]
        ut = self.fm._mask_and_zero_com(ut, self.model_extras["mask"])
        return ut.view(x.shape)


class ODESolver:
    """Integrates the learned velocity field from noise to a predicted sample.

    Args:
        velocity_model: Neural network that predicts the velocity field
            v^θ(x; t, ...) given coordinates, time and extra inputs.
        fm: Flow matcher used for center-of-mass centering after each step.
    """

    def __init__(self, velocity_model: nn.Module, fm: R3NFlowMatcher):
        self._velocity_model = velocity_model
        self._fm = fm

    def sample(
        self,
        x_init: Tensor,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        method: str = "euler",
        step_size: Optional[float] = None,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        return_intermediates: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ) -> Union[Tensor, Sequence[Tensor]]:
        """Solve the flow ODE from x_init to produce a predicted sample.

        Args:
            x_init: Initial noise sample, shape [B, L, 3].
            time_grid: Time points to solve over. If step_size is None the
                grid defines the discretisation. Defaults to [0, 1].
            method: ODE solver method supported by torchdiffeq (e.g. "euler",
                "midpoint", "rk4", "dopri5").
            step_size: Fixed step size. Must be None for adaptive solvers.
            atol: Absolute tolerance for adaptive solvers.
            rtol: Relative tolerance for adaptive solvers.
            return_intermediates: If True, return all time steps in time_grid;
                otherwise return only the final state.
            enable_grad: Whether to enable gradients during integration.
            **model_extras: Additional inputs forwarded to the velocity network
                at every step (e.g. mask, cond_emb, lag, temp).

        Returns:
            Final state tensor of shape [B, L, 3] when return_intermediates is
            False, or a tensor of shape [T, B, L, 3] when True.
        """
        time_grid = time_grid.to(x_init.device)
        ode_opts = {"step_size": step_size} if step_size is not None else {}

        dynamics = _VelocityDynamics(
            self._velocity_model, self._fm, model_extras
        )

        with torch.set_grad_enabled(enable_grad):
            result = odeint_adjoint(
                dynamics,
                x_init,
                time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )

        return result if return_intermediates else result[-1]
