# python3.8
"""Contains the function of ray marching.

Ray marching focuses on a single marching ray, which goes through a collection
of particles (points). Each point in the 3D space is represented by emitted
color and volume density. The final color to appear for each ray can be obtained
by accumulating the per-point color regarding the per-point density.

Ray marching is an important step for Neural Radiance Field (NeRF).

Paper: https://arxiv.org/pdf/2003.08934.pdf
"""

import torch
import torch.nn.functional as F

__all__ = ['PointIntegrator']

_DENSITY_CLAMP_MODES = ['relu', 'softplus', 'mipnerf']
_COLOR_CLAMP_MODES = ['none', 'widen_sigmoid']

EPS = 1e-3


class PointIntegrator(torch.nn.Module):
    """Defines the class to accumulate points along each ray.

    This class implements the `forward()` function for ray marching, which
    includes the following steps:

    1. Get the color and density of the points for each ray.
    2. Get alpha values for alpha compositing.
    3. Get accumulated transmittances.
    4. Get composite color and density with weighted sum (i.e., integration).

    More details can be found in Section 4 of paper

    https://arxiv.org/pdf/2003.08934.pdf
    """

    def __init__(self,
                 use_mid_point=True,
                 use_dist=True,
                 max_radial_dist=1e10,
                 density_noise_std=0.0,
                 density_clamp_mode='relu',
                 color_clamp_mode='none',
                 normalize_color=False,
                 delta_modulate_scalar=1.0,
                 use_white_background=False,
                 scale_color=True,
                 normalize_radial_dist=True,
                 clip_radial_dist=True):
        """Initializes hyper-parameters for ray marching.

        Args:
            use_mid_point: Whether to use the middle point between two adjacent
                points on each ray for accumulation. Defaults to `True`.
            use_dist: Whether to consider the distance between two adjacent
                points on each ray for accumulation. If set as `False`, the
                distance between two adjacent points is constantly set as `1`.
                Defaults to `True`.
            max_radial_dist: The maximum radial distance between a particular
                point to the camera. This argument is used to prevent the ray
                from going too far away. Defaults to `1e10`.
            density_noise_std: Standard deviation of the gaussian noise added to
                densities.
            density_clamp_mode: Mode of clamping densities. Defaults to `relu`.
            color_clamp_mode: Mode of clamping colors. Defaults to `none`.
            normalize_color: Whether to normalize the output composite color per
                ray. Defaults to `False`.
            delta_modulate_scalar: Scalar value to modulate delta of radial
                distance.
            use_white_background: Whether to use white background. Defaults to
                `False`.
            scale_color: Whether to scale the output composite color to range
                (-1, 1). Defaults to `True`.
            normalize_radial_dist: Whether to normalize the output composite
                radial distance per ray. Defaults to `True`.
            clip_radial_dist: Whether to clip the output composite radial
                distance. Defaults to `True`.
        """
        super().__init__()
        self.use_mid_point = use_mid_point
        self.use_dist = use_dist
        self.max_radial_dist = max_radial_dist
        self.density_noise_std = density_noise_std
        self.density_clamp_mode = density_clamp_mode
        self.color_clamp_mode = color_clamp_mode
        self.normalize_color = normalize_color
        self.delta_modulate_scalar = delta_modulate_scalar
        self.use_white_background = use_white_background
        self.scale_color = scale_color
        self.normalize_radial_dist = normalize_radial_dist
        self.clip_radial_dist = clip_radial_dist

    def forward(self, colors, densities, radii, **kwargs):
        """Integrates points along each ray.

        For simplicity, we define the following notations:

        `N` denotes batch size.
        `R` denotes the number of rays, which usually equals `H * W`.
        `K` denotes the number of points on each ray.

        Args:
            colors: Per-point emitted color, with shape [N, R, K, C]. Here `C`
                denotes the number of color channels. Note that, the color can
                be represented by gray value (`C = 1`), RGB values (`C = 3`), or
                a feature vector (such as `C = 64`).
            densities: Per-point volume density, with shape [N, R, K, 1]. Here,
                the density can be roughly interpreted as how likely a ray will
                be blocked by this point.
            radii: Per-point radial distance, with shape [N, R, K, 1]. Here, the
                distance is measured by treating the camera as the origin.
            **kwargs: Additional keyword arguments to override the variables
                initialized in `__init__()`.

        Returns:
            A dictionary, containing
                - `composite_color`: The final per-ray composite color (or
                    color feature), with shape [N, R, C].
                - `composite_radial_dist`: The final per-ray composite radial
                    distance, with shape [N, R, 1].
                - `weights`: Per-point weight for integral, with shape
                    [N, R, K, 1].
                - `T_end`: The accumulated transmittance along the ray from
                    the start point `p_s` to the end point `p_e` in the
                    foreground scene. This can be interpreted as the probability
                    of the ray travelling from `p_s` to `p_e` without hitting
                    any other particles in the foreground scene. This variable
                    is with shape [N, R, 1].
        """
        # Parse arguments.
        use_mid_point = kwargs.get('use_mid_point', self.use_mid_point)
        use_dist = kwargs.get('use_dist', self.use_dist)
        max_radial_dist = kwargs.get('max_radial_dist', self.max_radial_dist)
        density_noise_std = kwargs.get('density_noise_std',
                                       self.density_noise_std)
        density_clamp_mode = kwargs.get(
            'density_clamp_mode', self.density_clamp_mode)
        color_clamp_mode = kwargs.get('color_clamp_mode', self.color_clamp_mode)
        normalize_color = kwargs.get('normalize_color', self.normalize_color)
        delta_modulate_scalar = kwargs.get(
            'delta_modulate_scalar', self.delta_modulate_scalar)
        use_white_background = kwargs.get(
            'use_white_background', self.use_white_background)
        scale_color = kwargs.get('scale_color', self.scale_color)
        normalize_radial_dist = kwargs.get(
            'normalize_radial_dist', self.normalize_radial_dist)
        clip_radial_dist = kwargs.get('clip_radial_dist', self.clip_radial_dist)

        # Check inputs.
        assert colors.ndim == 4
        N, R, K, _ = colors.shape
        assert densities.shape == (N, R, K, 1)
        assert radii.shape == (N, R, K, 1)
        density_clamp_mode = density_clamp_mode.lower()
        if density_clamp_mode not in _DENSITY_CLAMP_MODES:
            raise ValueError(f'Invalid clamp mode: `{density_clamp_mode}`!\n'
                             f'Modes allowed: {_DENSITY_CLAMP_MODES}.')
        color_clamp_mode = color_clamp_mode.lower()
        if color_clamp_mode not in _COLOR_CLAMP_MODES:
            raise ValueError(f'Invalid clamp mode: `{color_clamp_mode}`!\n'
                             f'Modes allowed: {_COLOR_CLAMP_MODES}.')

        # Compute distances between adjacent points on each ray. Such a distance
        # is termed as `delta` in the paper (Eq. (3)).
        deltas = radii[:, :, 1:, :] - radii[:, :, :-1, :]  # [N, R, K-1, 1]

        if delta_modulate_scalar != 1:
            deltas = torch.ones_like(deltas) * delta_modulate_scalar

        if use_mid_point:  # Using K-1 points on each ray.
            colors = (colors[:, :, :-1, :] + colors[:, :, 1:, :]) / 2
            densities = (densities[:, :, :-1, :] + densities[:, :, 1:, :]) / 2
            radii = (radii[:, :, :-1, :] + radii[:, :, 1:, :]) / 2
        else:  # Using K points on each ray.
            # Append a maximum distance to make sure all points have reference.
            delta_last = max_radial_dist * torch.ones_like(deltas[:, :, :1, :])
            deltas = torch.cat([deltas, delta_last], dim=2)  # [N, R, K, 1]

        ray_dirs = kwargs.get('ray_dirs')
        if ray_dirs is not None:  # [N, R, 3]
            assert ray_dirs.shape == (N, R, 3)
            ray_dirs = ray_dirs.unsqueeze(-1)  # [N, R, 3, 1]
            deltas = deltas * torch.norm(ray_dirs, dim=-2,
                                         keepdim=True)  # [N, R, K, 1]

        if not use_dist:
            deltas[:] = 1

        if 'bg_index' in kwargs:
            bg_index = F.one_hot(kwargs['bg_index'].squeeze(-1),
                                 num_classes=deltas.shape[-2]).to(torch.bool)
            bg_index = bg_index.unsqueeze(-1)
            deltas[bg_index] = max_radial_dist

        if density_noise_std > 0:
            densities = densities + density_noise_std * torch.randn_like(
                densities)

        if density_clamp_mode == 'relu':
            densities = F.relu(densities)
        elif density_clamp_mode == 'softplus':
            densities = F.softplus(densities)
        elif density_clamp_mode == 'mipnerf':
            densities = F.softplus(densities - 1)
        else:
            raise ValueError(f'Not implemented clamping mode: '
                             f'`{density_clamp_mode}`!\n')

        if color_clamp_mode == 'widen_sigmoid':
            colors = torch.sigmoid(colors) * (1 + 2 * EPS) - EPS

        # Compute per-point alpha values. See Eq. (3) in the paper.
        alphas = 1 - torch.exp(- deltas * densities)
        if not use_mid_point and max_radial_dist > 0:
            alphas[:, :, -1, :] = 1

        if 'is_valid' in kwargs:
            alphas = alphas * kwargs['is_valid']

        # Compute per-point accumulated transmittance. See Eq. (3) in the paper.
        # Here, we shift `alpha` forward by one index, because the transmittance
        # of each point is only related to its previous points, excluding
        # itself.
        alphas_shifted = torch.cat(
            [torch.ones_like(alphas[:, :, :1, :]), 1 - alphas + 1e-10], dim=2)
        T = torch.cumprod(alphas_shifted, dim=2)[:, :, :-1, :]  # Transmittance.

        # Compute per-point integral weights.
        weights = alphas * T
        weights_sum = weights.sum(dim=2)

        # Get per-ray color.
        composite_color = torch.sum(weights * colors, dim=2)
        if normalize_color:
            composite_color = composite_color / weights_sum
        if use_white_background:
            composite_color = composite_color + 1 - weights_sum
        if scale_color:
            composite_color = composite_color * 2 - 1

        # Get per-ray radial distance.
        composite_radial_dist = torch.sum(weights * radii, dim=2)
        if normalize_radial_dist:
            composite_radial_dist = composite_radial_dist / weights_sum
        if clip_radial_dist:
            composite_radial_dist = torch.nan_to_num(
                composite_radial_dist, float('inf'))
            composite_radial_dist = torch.clip(
                composite_radial_dist, torch.min(radii), torch.max(radii))

        results = {
            'composite_color': composite_color,
            'composite_radial_dist': composite_radial_dist,
            'weight': weights,
            'T_end': T[:, :, -1, :]
        }

        return results
