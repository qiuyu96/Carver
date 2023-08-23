# python3.8
"""Utility functions used for rendering module."""

import math
import torch
import torch.nn.functional as F

EPS = 1e-6


def sample_importance(radial_dists,
                      weights,
                      num_importance,
                      smooth_weights=False):
    """Implements importance sampling, which is the crucial step in hierarchical
    sampling of NeRF. Hierarchical volume sampling mainly includes the following
    steps:

    1. Sample a set of `Nc` points using stratified sampling.
    2. Evaluate the 'coarse' network at locations of these points as described
       in Eq. (2) & (3) in the paper.
    3. Normalize the output weights to get a piecewise-constant PDF (probability
       density function) along the ray.
    4. Sample a second set of `Nf` points from this distribution using inverse
       transform sampling.

    And importance sampling refers to step 4 specifically.

    Code is borrowed from:

    https://github.com/NVlabs/eg3d/blob/main/eg3d/training/volumetric_rendering/renderer.py

    Args:
        radial_dists: Radial distances, with shape [N, R, K, 1]
        weights: Per-point weight for integral, with shape [N, R, K, 1].
        num_importance: Number of points for importance sampling.
        smooth_weights: Whether to smooth weights. Defaults to `False`.

    Returns:
        importance_radial_dists: Radial distances of importance sampled points
            along rays.
    """
    with torch.no_grad():
        batch_size, num_rays, samples_per_ray, _ = radial_dists.shape
        radial_dists = radial_dists.reshape(batch_size * num_rays,
                                            samples_per_ray)
        weights = weights.reshape(batch_size * num_rays, -1) + 1e-5

        # Smooth weights.
        if smooth_weights:
            weights = F.max_pool1d(weights.unsqueeze(1).float(),
                                   2, 1, padding=1)
            weights = F.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

        radial_dists_mid = 0.5 * (radial_dists[:, :-1] + radial_dists[:, 1:])
        importance_radial_dists = sample_pdf(radial_dists_mid, weights[:, 1:-1],
                                             num_importance)
        importance_radial_dists = importance_radial_dists.detach().reshape(
            batch_size, num_rays, num_importance, 1)

    return importance_radial_dists


def sample_pdf(bins, weights, num_importance, det=False, eps=1e-5):
    """Sample `num_importance` samples from `bins` with distribution defined
        by `weights`. Borrowed from:

        https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py

    Args:
        bins: Bins distributed along rays, with shape (N * R, K - 1).
        weights: Per-point weight for integral, with shape [N * R, K].
        num_importance: The number of samples to draw from the distribution.
        det: Deterministic or not. Defaults to `False`.
        eps: A small number to prevent division by zero. Defaults to `1e-5`.

    Returns:
        samples: The sampled samples.
    """
    n_rays, n_samples_ = weights.shape
    weights = weights + eps
    # Prevent division by zero (don't do inplace op!).
    pdf = weights / torch.sum(weights, -1,
                              keepdim=True)  # (n_rays, n_samples_)
    cdf = torch.cumsum(pdf, -1)  # (n_rays, N_samples),
    # Cumulative distribution function.
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf],
                    -1)  # (n_rays, n_samples_+1)

    if det:
        u = torch.linspace(0, 1, num_importance, device=bins.device)
        u = u.expand(n_rays, num_importance)
    else:
        u = torch.rand(n_rays, num_importance, device=bins.device)
    u = u.contiguous()

    indices = torch.searchsorted(cdf, u)
    below = torch.clamp_min(indices - 1, 0)
    above = torch.clamp_max(indices, n_samples_)

    indices_sampled = torch.stack([below, above], -1).view(n_rays,
                                                           2 * num_importance)
    cdf_g = torch.gather(cdf, 1, indices_sampled)
    cdf_g = cdf_g.view(n_rays, num_importance, 2)
    bins_g = torch.gather(bins, 1, indices_sampled).view(n_rays,
                                                         num_importance, 2)

    # `denom` equals 0 means a bin has weight 0, in which case it will not be
    # sampled anyway, therefore any value for it is fine (set to 1 here).
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < eps] = 1

    samples = (bins_g[..., 0] + (u - cdf_g[..., 0]) /
               denom * (bins_g[..., 1] - bins_g[..., 0]))

    return samples


def unify_attributes(radial_dists1,
                     colors1,
                     densities1,
                     radial_dists2,
                     colors2,
                     densities2,
                     points1=None,
                     points2=None):
    """Unify attributes of point samples according to their radial distances.

    Args:
        radial_dists1: Radial distances of the first pass, with shape
            [N, R, K1, 1].
        colors1: Colors or features of the first pass, with shape [N, R, K1, C].
        densities1: Densities of the first pass, with shape [N, R, K1, 1].
        radial_dists2: Radial distances of the second pass, with shape
            [N, R, K2, 1].
        colors2: Colors or features of the second pass, with shape
            [N, R, K2, C].
        densities2: Densities of the second pass, with shape [N, R, K2, 1].
        points1 (optional): Point coordinates of the first pass,
            with shape [N, R, K1, 3].
        points2 (optional): Point coordinates of the second pass,
            with shape [N, R, K2, 3].

    Returns:
        all_radial_dists: Unified radial distances, with shape [N, R, K1+K2, 1].
        all_colors: Unified colors or features, with shape [N, R, K1+K2, C].
        all_densities: Unified densities, with shape [N, R, K1+K2, 1].
    """
    all_radial_dists = torch.cat([radial_dists1, radial_dists2], dim=-2)
    all_colors = torch.cat([colors1, colors2], dim=-2)
    all_densities = torch.cat([densities1, densities2], dim=-2)

    _, indices = torch.sort(all_radial_dists, dim=-2)
    all_radial_dists = torch.gather(all_radial_dists, -2, indices)
    all_colors = torch.gather(
        all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
    all_densities = torch.gather(all_densities, -2,
                                 indices.expand(-1, -1, -1, 1))

    if points1 is not None and points2 is not None:
        all_points = torch.cat([points1, points2], dim=-2)
        all_points = torch.gather(
            all_points, -2, indices.expand(-1, -1, -1, all_points.shape[-1]))
        return all_radial_dists, all_colors, all_densities, all_points

    return all_radial_dists, all_colors, all_densities


def depth2pts_outside(ray_o, ray_d, depth):
    """Compute point coordinates in the inverted sphere parameterization.

    This function is borrowed from the official code of NeRF++:

    https://github.com/Kai-46/nerfplusplus

    Args:
        ray_o (torch.Tensor): Ray origins, with shape [N, R, K, 3].
        ray_d (torch.Tensor): Ray directions, with shape [N, R, K, 3].
        depth (torch.Tensor): Inverse of distance to sphere origin,
            with shape [N, R, K].

    Returns:
        pts (torch.Tensor): Sampled points with inversed sphere parametrization,
            denoted as (x', y', z', 1/r), with shape [N, R, K, 4].
        depth_real (torch.Tensor): Depth in Euclidean space.
    """

    # Note: d1 becomes negative if this mid point is behind camera.
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p_mid = ray_o + d1.unsqueeze(-1) * ray_d
    p_mid_norm = torch.norm(p_mid, dim=-1)
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    d2 = torch.sqrt(1. - p_mid_norm * p_mid_norm) * ray_d_cos
    p_sphere = ray_o + (d1 + d2).unsqueeze(-1) * ray_d

    rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
    rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
    phi = torch.asin(p_mid_norm)
    theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
    rot_angle = (phi - theta).unsqueeze(-1)  # [..., 1]

    # Rotate p_sphere using Rodrigues formula:
    # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    p_sphere_new = (
        p_sphere * torch.cos(rot_angle) +
        torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) +
        rot_axis * torch.sum(rot_axis * p_sphere, dim=-1, keepdim=True) *
        (1. - torch.cos(rot_angle)))
    p_sphere_new = p_sphere_new / torch.norm(
        p_sphere_new, dim=-1, keepdim=True)
    pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    # Calculate conventional depth.
    depth_real = 1. / (depth + EPS) * torch.cos(theta) * ray_d_cos + d1

    return pts, depth_real


class PositionEncoder(torch.nn.Module):
    """Implements the class for positional encoding."""

    def __init__(self,
                 input_dim,
                 max_freq_log2,
                 num_freqs,
                 log_sampling=True,
                 factor=1.0,
                 include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
        """Initializes with basic settings.

        Args:
            input_dim: Dimension of input to be embedded.
            max_freq_log2: `log2` of max freq; min freq is 1 by default.
            num_freqs: Number of frequency bands.
            log_sampling: If True, frequency bands are linerly sampled in
                log-space.
            factor: Factor of the frequency bands.
            include_input: If True, raw input is included in the embedding.
                Defaults to True.
            periodic_fns: Periodic functions used to embed input.
                Defaults to (torch.sin, torch.cos).
        """
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * num_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2.**torch.linspace(0., max_freq_log2,
                                                 num_freqs) * factor
        else:
            self.freq_bands = torch.linspace(2.**0., 2.**max_freq_log2,
                                             num_freqs) * factor

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input):
        """Forward function of positional encoding.

        Args:
            input: Input tensor with shape [..., input_dim]

        Returns:
            output: Output tensor with shape [..., out_dim]
        """
        output = []
        if self.include_input:
            output.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                output.append(p_fn(input * freq))
        output = torch.cat(output, dim=-1)

        return output

    def get_out_dim(self):
        return self.out_dim


class GaussianCameraPoseSampler:
    """
    Samples pitch and yaw from a Gaussian distribution and returns a camera
    pose. Camera is specified as looking at the origin. If horizontal and
    vertical stddev (specified in radians) are zero, gives a deterministic
    camera pose with yaw=horizontal_mean, pitch=vertical_mean. The coordinate
    system is specified with y-up, z-forward, x-left. Horizontal mean is the
    azimuthal angle (rotation around y axis) in radians, vertical mean is the
    polar angle (angle from the y axis) in radians. A point along the z-axis
    has azimuthal_angle=0, polar_angle=pi/2.

    Example:
    For a camera looking at the origin with the camera at position [0, 0, 1]:
    cam2world = GaussianCameraPoseSampler.sample(math.pi/2,
                                                 math.pi/2,
                                                 radius=1)
    """

    @staticmethod
    def sample(horizontal_mean,
               vertical_mean,
               horizontal_stddev=0,
               vertical_stddev=0,
               radius=1,
               batch_size=1,
               device='cpu'):
        h = torch.randn((batch_size, 1),
                        device=device) * horizontal_stddev + horizontal_mean
        v = torch.randn(
            (batch_size, 1), device=device) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2 * v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius * torch.sin(phi) * torch.cos(math.pi -
                                                                     theta)
        camera_origins[:, 2:3] = radius * torch.sin(phi) * torch.sin(math.pi -
                                                                     theta)
        camera_origins[:, 1:2] = radius * torch.cos(phi)

        forward_vectors = normalize_vecs(-camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins)


class LookAtPoseSampler:
    """
    Same as GaussianCameraPoseSampler, except the
    camera is specified as looking at 'lookat_position', a 3-vector.

    Example:
    For a camera pose looking at the origin with the camera at position [
        0, 0, 1]:
    cam2world = LookAtPoseSampler.sample(
        math.pi/2, math.pi/2, torch.tensor([0, 0, 0]), radius=1)
    """

    @staticmethod
    def sample(horizontal_mean,
               vertical_mean,
               lookat_position,
               horizontal_stddev=0,
               vertical_stddev=0,
               radius=1,
               batch_size=1,
               device='cpu'):
        h = torch.randn((batch_size, 1),
                        device=device) * horizontal_stddev + horizontal_mean
        v = torch.randn(
            (batch_size, 1), device=device) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2 * v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius * torch.sin(phi) * torch.cos(math.pi -
                                                                     theta)
        camera_origins[:, 2:3] = radius * torch.sin(phi) * torch.sin(math.pi -
                                                                     theta)
        camera_origins[:, 1:2] = radius * torch.cos(phi)

        # forward_vectors = normalize_vecs(-camera_origins)
        forward_vectors = normalize_vecs(lookat_position - camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins)


class UniformCameraPoseSampler:
    """
    Same as GaussianCameraPoseSampler, except the pose is sampled from a
    uniform distribution with range +-[horizontal/vertical]_stddev.

    Example:
    For a batch of random camera poses looking at the origin with yaw sampled
    from [-pi/2, +pi/2] radians:

    cam2worlds = UniformCameraPoseSampler.sample(math.pi/2,
                                                 math.pi/2,
                                                 horizontal_stddev=math.pi/2,
                                                 radius=1,
                                                 batch_size=16)
    """

    @staticmethod
    def sample(horizontal_mean,
               vertical_mean,
               horizontal_stddev=0,
               vertical_stddev=0,
               radius=1,
               batch_size=1,
               device='cpu'):
        h = (torch.rand((batch_size, 1), device=device) * 2 -
             1) * horizontal_stddev + horizontal_mean
        v = (torch.rand((batch_size, 1), device=device) * 2 -
             1) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2 * v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius * torch.sin(phi) * torch.cos(math.pi -
                                                                     theta)
        camera_origins[:, 2:3] = radius * torch.sin(phi) * torch.sin(math.pi -
                                                                     theta)
        camera_origins[:, 1:2] = radius * torch.cos(phi)

        forward_vectors = normalize_vecs(-camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins)


def create_cam2world_matrix(forward_vector, origin):
    """
    Takes in the direction the camera is pointing and the camera origin and
    returns a cam2world matrix. Works on batches of forward_vectors, origins.
    Assumes y-axis is up and that there is no camera roll.
    """

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0],
                             dtype=torch.float,
                             device=origin.device).expand_as(forward_vector)

    right_vector = -normalize_vecs(
        torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize_vecs(
        torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(
        forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack(
        (right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=origin.device)
    translation_matrix = translation_matrix.unsqueeze(0).repeat(
        forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert (cam2world.shape[1:] == (4, 4))
    return cam2world


def compute_camera_origins(angles, radius):
    yaw = angles[:, [0]]  # [batch_size, 1]
    pitch = angles[:, [1]]  # [batch_size, 1]

    assert yaw.ndim == 2, f"Wrong shape: {yaw.shape}, {pitch.shape}"
    assert yaw.shape == pitch.shape, f"Wrong shape: {yaw.shape}, {pitch.shape}"

    origins = torch.zeros((yaw.shape[0], 3), device=yaw.device)
    origins[:, [0]] = radius * torch.sin(pitch) * torch.cos(yaw)
    origins[:, [2]] = radius * torch.sin(pitch) * torch.sin(yaw)
    origins[:, [1]] = radius * torch.cos(pitch)

    return origins


def compute_cam2world_matrix(camera_angles, radius):
    """
    Takes in the direction the camera is pointing and the camera origin and
    returns a cam2world matrix.

    Note: `camera_angles` should be provided in the "yaw/pitch/roll" format,
    and with shape [batch_size, 3]
    """
    camera_origins = compute_camera_origins(camera_angles,
                                            radius)  # [batch_size, 3]
    forward_vector = normalize_vecs(-camera_origins)  # [batch_size, 3]
    batch_size = forward_vector.shape[0]
    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor(
        [0, 1, 0], dtype=torch.float,
        device=forward_vector.device).expand_as(forward_vector)
    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector,
                                             dim=-1))
    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector,
                                           dim=-1))

    rotation_matrix = torch.eye(
        4, device=forward_vector.device).unsqueeze(0).repeat(batch_size, 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack(
        (-left_vector, up_vector, -forward_vector), axis=-1)

    translation_matrix = torch.eye(
        4, device=forward_vector.device).unsqueeze(0).repeat(batch_size, 1, 1)
    translation_matrix[:, :3, 3] = camera_origins

    cam2world = translation_matrix @ rotation_matrix

    return cam2world


def FOV_to_intrinsics(fov_degrees, device='cpu'):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view,
    specified in degrees. Note the intrinsics are returned as normalized by
    image size, rather than in pixel units. Assumes principal point is at image
    center.
    """

    focal_length = float(1 / (math.tan(fov_degrees * 3.14159 / 360) * 1.414))
    intrinsics = torch.tensor(
        [[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]],
        device=device)
    return intrinsics


def normalize_vecs(vectors, dim=-1):
    """Normalize vectors."""
    return vectors / (torch.norm(vectors, dim=dim, keepdim=True))


def dividable(n, k=2):
    if k == 2:
        for i in range(int(math.sqrt(n)), 0, -1):
            if n % i == 0:
                break
        return i, n // i
    elif k == 3:
        for i in range(int(float(n) ** (1/3)), 0, -1):
            if n % i == 0:
                b, c = dividable(n // i, 2)
                return i, b, c
    else:
        raise NotImplementedError