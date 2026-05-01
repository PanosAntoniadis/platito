# Helper functions adapted from the bioemu-benchmarks codebase:
# https://github.com/microsoft/bioemu-benchmarks

from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.stats import binned_statistic_2d
from dataclasses import dataclass

from scipy.optimize import bisect

K_BOLTZMANN = 0.001987203599772605


# https://github.com/microsoft/bioemu-benchmarks/blob/main/bioemu_benchmarks/eval/md_emulation/plot.py
def plot_free_energy_on_axes(
    axes: Axes,
    projections: np.ndarray,
    num_bins: int = 20,
    max_energy: float = 10.0,
    levels: int = 10,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    add_colorbar: bool = False,
    colorbar_axis: Axes | None = None,
    kBT: float = 1.0,
) -> None:
    """
    Convert sample projections to free energy surfaces and plot a contour plot on a given axes
    object.

    Args:
        axes: matplotlib axes object.
        projections: Projections to plot. Should be array of shape [n_samples x 2].
        num_bins: Number of bins used to discretize the free energy surface.
        max_energy: Clamp the free energy surface to this value. Units are the same as `kBT`.
        levels: Number of levels used in the contour plot.
        x_range: Optional range of x axis values.
        y_range: Optional range of y axis values.
        add_colorbar: Add color bar to plot (requires `colorbar_axis`).
        colorbar_axis: Axes object to which color bar should be added.
        kBT: Product of Boltzmann constant and temperature. Defines units in the plot.
    """
    # Extract x and y coordinates.
    projections_x = projections[:, 0]
    projections_y = projections[:, 1]
    # Set up plot ranges if none are provided.
    if x_range is None:
        x_range = (min(projections_x), max(projections_x))
    if y_range is None:
        y_range = (min(projections_y), max(projections_y))

    # Set up x and y grids for plot.
    grid_x, grid_y = np.mgrid[
        x_range[0] : x_range[1] : num_bins * 1j, y_range[0] : y_range[1] : num_bins * 1j  # type: ignore
    ]
    # Discretize data distribution.
    binned = binned_statistic_2d(
        projections_x,
        projections_y,
        None,
        "count",
        bins=[num_bins, num_bins],
        range=[x_range, y_range],
    )

    # Generate the contour plot. The offset here is for numerical stability, the energy limit used
    # for plotting is handled purely by `max_energy`.
    # p = exp(-e/kT) => e = -kT ln(p)
    energy = -kBT * np.log(binned.statistic + 1e-12)

    # Remove minimum energy and clamp to maximum for consistent plotting.
    energy -= energy.min()
    energy = np.minimum(energy, max_energy + 1)

    # Change cutoff color to white.
    cmap = copy.copy(plt.cm.turbo)
    cmap.set_over(color="w")

    # Add contours.
    contours = axes.contourf(
        grid_x,
        grid_y,
        energy,
        cmap=cmap,
        levels=levels,
        vmin=0,
        vmax=max_energy,
    )
    contours.set_clim(0, max_energy)

    # Add color bar if requested.
    if add_colorbar:
        assert (
            colorbar_axis is not None
        ), "Colorbar axis required for color bar."
        cbar_ = axes.figure.colorbar(contours, cax=colorbar_axis)
        cbar_.ax.set_ylim(0, max_energy)


# https://github.com/microsoft/bioemu-benchmarks/blob/main/bioemu_benchmarks/eval/md_emulation/state_metric.py
@dataclass
class DistributionMetricSettings:
    """
    Data class for collecting distribution metric settings.

    Attributes:
        n_resample: Resample projections using this many points.
        sigma_resample: Standard deviation of noise added to resamples points (mainly used to
            improve binning stability. Formally corresponds to Gaussian convolution of discretized
            density.
        num_bins: Number of bins used for discretizing density.
        energy_cutoff: Energy cutoff used for computing metric (units are kcal/mol).
        padding: Padding used for discretization grid.
    """

    n_resample: int = 1000000
    sigma_resample: float = 0.25
    num_bins: int = 50
    energy_cutoff: float = 4.0
    padding: float = 0.5


def histogram_bin_edges(
    x: np.ndarray, num_bins: int, padding: float | None = 0.5
) -> np.ndarray:
    """
    Generate histogram bin edges for provided 1D array. Creates `num_bins` + 1 edges between minimum
    and maximum of array using optional padding.

    Args:
        x: Array used for bin edge computation.
        num_bins: Number of bins.
        padding: If provided, upper and lower limits will be extended by this value times the grid
          spacing.

    Returns:
        Array of bin edges.
    """
    x_min = np.min(x)
    x_max = np.max(x)

    if padding is not None:
        delta_x = (x_max - x_min) / (num_bins + 1)
        x_min = x_min - padding * delta_x
        x_max = x_max + padding * delta_x

    return np.linspace(x_min, x_max, num_bins + 1)


def compute_density_2D(
    x: np.ndarray, edges_x: np.ndarray, edges_y: np.ndarray
) -> np.ndarray:
    """
    Auxiliary function for computing normalized density from a two dimensional array of the shape
    given bin edges.

    Args:
        x: Array of the shape [num_samples x 2].
        edges_x: Bin edges along first dimension.
        edges_y: Bin edges along second dimension.

    Returns:
        Discretized density of the shape [num_bins_x x num_bins_y].
    """
    density, _, _ = np.histogram2d(
        x[:, 0], x[:, 1], bins=(edges_x, edges_y), density=True
    )
    return density


def resample_with_noise(
    x: np.ndarray,
    num_samples: int,
    sigma: float,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """
    Resample `num_samples` points from `x` adding Gaussian noise with standard deviation `sigma`.
    This is e.g. used to make binning for statistics more robust.

    Args:
        x: Array of data to resample. Each row corresponds to a data point (shape [N x d]).
        num_samples: Number of samples to redraw.
        sigma: Standard deviation of Gaussian noise added to resamples points.
        rng: Optional random generator or integer seed for reproducibility.

    Returns:
        Array of resamples, noised points with shape [num_samples x d].
    """
    rng = np.random.default_rng(rng)
    indices = np.arange(x.shape[0])
    sel = rng.choice(indices, size=num_samples)
    return x[sel] + sigma * rng.standard_normal((num_samples, x.shape[1]))


def compute_rmse(
    energies_pred: np.ndarray,
    energies_target: np.ndarray,
    minimize: bool = True,
) -> float:
    """
    Compute root mean squared error between two arrays containing energies. Optionally, energy
    difference can be minimized with respect to an integration constant between both arrays.

    Args:
        energies_pred: Array of predicted energies (shape [num_grid]).
        energies_target: Array of reference energies (shape [num_grid]).
        minimize: Minimize error by finding an optimal scalar shift between values.

    Returns:
        Computed error.
    """

    if minimize:
        # Optimal shift for RMSE / MSE is difference in means.
        energy_shift: float = np.mean(energies_target) - np.mean(energies_pred)
    else:
        energy_shift = 0.0

    energies_difference = energies_pred - energies_target + energy_shift

    return np.sqrt(np.mean(energies_difference**2))


def compute_mae(
    energies_pred: np.ndarray,
    energies_target: np.ndarray,
    minimize: bool = True,
) -> float:
    """
    Compute mean absolute error between two arrays containing energies. Optionally, energy
    difference can be minimized with respect to an integration constant between both arrays.

    Args:
        energies_pred: Array of predicted energies (shape [num_samples]).
        energies_target: Array of reference energies (shape [num_samples]).
        minimize: Minimize error by finding an optimal scalar shift between values.

    Returns:
        Computed error.
    """
    if minimize:
        # Optimal shift needs to be determined with a short numerical optimization.
        def mae_derivative(delta_energies: float) -> float:
            return np.sum(
                np.sign(energies_pred - energies_target + delta_energies)
            )

        limit_lower = np.min(energies_pred) - np.max(energies_target)
        limit_upper = np.max(energies_pred) - np.min(energies_target)
        energy_shift = bisect(
            mae_derivative, limit_lower, limit_upper, disp=False
        )
    else:
        energy_shift = 0.0

    energies_difference = energies_pred - energies_target + energy_shift

    return np.mean(np.abs(energies_difference))


class DistributionMetrics2D:
    def __init__(
        self,
        reference_projections: np.ndarray,
        n_resample: int = 1000000,
        sigma_resample: float = 0.25,
        num_bins: int = 50,
        energy_cutoff: float = 4.0,
        temperature_K: float = 300.0,
        padding: float = 0.5,
        random_seed: int | None = None,
    ):
        """
        Class for computing free energy mean absolute and root mean squared errors between reference
        and sample densities computed from low dimensional projections based on protein structures.

        This routine follows the overall procedure:

            1) Resample data and add Gaussian noise.
            2) Discretize resampled data and normalize to density.
            3) Either select bin coordinates where reference probabilities are greater than a cutoff
               and clamp low sample probabilities to this cutoff (`score`) or
               select bin coordinates where there are reference and sample densities
               (`score_nonzero`).
            4) Compute free energies on those bins.
            5) Compute minimized mean absolute error and root mean squared error (optimizing global
               energy offset).

        Args:
            reference_projections: Array of reference projections (shape [N x 2]).
            n_resample: Resample projections using this many points.
            sigma_resample: Standard deviation of noise added to resamples points (mainly used to
                improve binning stability. Formally corresponds to Gaussian convolution of discretized
                density.
            num_bins: Number of bins used for discretizing density.
            energy_cutoff: Energy cutoff used for computing metric (units are kcal/mol).
            temperature_K: Temperature used for analysis in Kelvin.
            padding: Padding used for discretization grid.
            random_seed: Random seed for resampling.
        """
        self.n_resample = n_resample
        self.sigma_resample = sigma_resample
        self.kBT = temperature_K * K_BOLTZMANN
        self.energy_cutoff = energy_cutoff
        self.random_seed = random_seed

        # Resample reference projections.
        reference_projections_noised = resample_with_noise(
            reference_projections,
            n_resample,
            sigma_resample,
            rng=self.random_seed,
        )
        # Get bin edges for discretization.
        self.edges_x = histogram_bin_edges(
            reference_projections_noised[:, 0], num_bins, padding=padding
        )
        self.edges_y = histogram_bin_edges(
            reference_projections_noised[:, 1], num_bins, padding=padding
        )
        # Compute discretized density.
        self.density_ref = compute_density_2D(
            reference_projections_noised, self.edges_x, self.edges_y
        )

        # Compute reference density mask based on energy cutoff,
        p_cutoff = self._compute_density_cutoff(self.density_ref)
        self.low_energy_mask = self.density_ref > p_cutoff

    def _compute_density_cutoff(self, density: np.ndarray) -> float:
        """
        Auxiliary function for computing probability cutoff based on energy threshold.

        Args:
            density: Density for which cutoff should be computed.

        Returns:
            Density cutoff corresponding to energy cutoff.
        """
        energy_min = -self.kBT * np.log(np.max(density))
        return np.exp(-(energy_min + self.energy_cutoff) / self.kBT)

    def score_nonzero(
        self, sample_projections: np.ndarray
    ) -> tuple[float, float, float]:
        """
        Compute free energy errors between sample and reference densities scores in regions where
        there are reference data points.

        Args:
            sample_projections: Sample projections with shape [M x 2].

        Returns:
            Optimized mean absolute, root mean squared errors and coverage (fraction of grid states
            where there are samples).
        """
        # Resample sample projections and compute discretized density.
        sample_projections_noised = resample_with_noise(
            sample_projections,
            self.n_resample,
            self.sigma_resample,
            rng=self.random_seed,
        )
        sample_density = compute_density_2D(
            sample_projections_noised, self.edges_x, self.edges_y
        )

        # Select subset of states where the reference energy is low and where there are samples in
        # the model density.
        common_mask = np.logical_and(self.low_energy_mask, sample_density > 0)

        # Compute free energy surfaces based on common mask.
        energy_ref = -self.kBT * np.log(self.density_ref[common_mask])
        energy_samples = -self.kBT * np.log(sample_density[common_mask])

        # Compute minimal errors between surfaces and coverage of samples counted based on above
        # threshold criterion.
        mae_min = compute_mae(energy_samples, energy_ref, minimize=True)
        rmse_min = compute_rmse(energy_samples, energy_ref, minimize=True)
        coverage = np.count_nonzero(common_mask) / np.count_nonzero(
            self.low_energy_mask
        )

        return mae_min, rmse_min, coverage
