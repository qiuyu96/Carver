# python3.8
"""Contains the functions to represent a point in 3D space.

Typically, a point can be represented by its 3D coordinates, by retrieving from
a feature volume, or by combining triplane features.

Paper (coordinate): https://arxiv.org/pdf/2003.08934.pdf
Paper (feature volume): https://arxiv.org/pdf/2112.10759.pdf
Paper (triplane): https://arxiv.org/pdf/2112.07945.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['PointRepresenter']

_REPRESENTATION_TYPES = ['coordinate', 'volume', 'triplane', 'hybrid', 'mpi']


class PointRepresenter(nn.Module):
    """Defines the class to get per-point representation.

    This class implements the `forward()` function to get the representation
    based on the per-point 3D coordinates and the reference representation (such
    as a feature volume or triplane features).
    """

    def __init__(self,
                 representation_type='coordinate',
                 triplane_axes=None,
                 mpi_levels=None,
                 coordinate_scale=None,
                 bound=None):
        """Initializes hyper-parameters for getting point representations.

        NOTE:

        When using triplane representation, the three planes are defaulted as
        follows:

        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
            [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
        ]

        where for each plane, the first two rows stand for the plane axes while
        the third row stands for the plane normal.

        Args:
            representation_type: Type of representation used to describe a point
                in the 3D space. Defaults to `coordinate`.
            coordinate_scale: Scale factor to normalize coordinates.
                Defaults to `None`.
            bound: Bound used to normalize coordinates, with shape [1, 2, 3].
                Defaults to `None`.

        Note that only one of the above two parameters used for normalizing
        coordinates can be available.
        """
        super().__init__()

        self.coordinate_scale = None
        if (coordinate_scale is not None) and (coordinate_scale > 0):
            self.coordinate_scale = coordinate_scale
        if bound is not None:
            self.register_buffer('bound', bound)
        else:
            self.bound = None

        representation_type = representation_type.lower()
        if representation_type not in _REPRESENTATION_TYPES:
            raise ValueError(f'Invalid representation type: '
                             f'`{representation_type}`!\n'
                             f'Types allowed: {_REPRESENTATION_TYPES}.')

        self.representation_type = representation_type
        if self.representation_type in ['coordinate', 'volume']:
            pass
        elif self.representation_type in ['triplane', 'hybrid']:
            if triplane_axes is None:
                self.register_buffer(
                    'triplane_axes',
                    torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                  [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
                                  [[0, 0, 1], [0, 1, 0], [1, 0, 0]]],
                                 dtype=torch.float32))
            else:
                self.register_buffer('triplane_axes', triplane_axes)
        elif self.representation_type == 'mpi':
            self.register_buffer('mpi_levels', mpi_levels)
        else:
            raise NotImplementedError(f'Not implemented representation type: '
                                      f'`{self.representation_type}`!\n')

    def forward(self,
                points,
                ref_representation=None,
                align_corners=False):
        """Gets per-point representation based on its coordinates.

        For simplicity, we define the following notations:

        `N` denotes batch size.
        `R` denotes the number of rays, which usually equals `H * W`.
        `K` denotes the number of points on each ray.
        `C` denotes the dimension of per-point representation.

        Args:
            points: Per-point 3D coordinates, with shape [N, R * K, 3].
            ref_representation: The reference representation, depending on the
                representation type used. For example, this field will be
                ignored if `self.representation_type` is set as `coordinate`,
                a feature volume is expected if `self.representation_type` is
                set as `volume`, while triplane features are expected if
                `self.representation_type` is set as `triplane`. Defaults to
                `None`.

        Returns:
            Per-point representation, with shape [N, R * K, C].
        """
        if self.representation_type == 'coordinate':
            return points
        if self.representation_type == 'mpi':
            return retrieve_from_mpi(points=points,  # [N, R, K, 3]
                                     isosurfaces=ref_representation,
                                     levels=self.mpi_levels)

        # Normalize point coordinates to the desired range, typically [-1, 1].
        if self.coordinate_scale is not None:
            normalized_points = (2 / self.coordinate_scale) * points
        elif self.bound is not None:
            normalized_points = (points - self.bound[:, :1]) / (
                self.bound[:, 1:] - self.bound[:, :1])  # To range [0, 1].
            normalized_points = 2 * normalized_points - 1  # To range [-1, 1].
        else:
            normalized_points = points

        if self.representation_type == 'volume':
            return retrieve_from_volume(
                coordinates=normalized_points,
                volume=ref_representation)
        if self.representation_type == 'triplane':
            return retrieve_from_planes(
                plane_axes=self.triplane_axes.to(points.device),
                plane_features=ref_representation,
                coordinates=normalized_points,
                align_corners=align_corners)
        if self.representation_type == 'hybrid':
            assert (isinstance(ref_representation, list)
                        or isinstance(ref_representation, tuple))
            triplane = ref_representation[0]
            feature_volume = ref_representation[1]
            point_features_triplane = retrieve_from_planes(
                plane_axes=self.triplane_axes.to(points.device),
                plane_features=triplane,
                coordinates=normalized_points,
                align_corners=align_corners)
            point_features_volume = retrieve_from_volume(
                coordinates=normalized_points,
                volume=feature_volume)
            point_features = torch.cat(
                [point_features_volume, point_features_triplane], dim=-1)
            return point_features

        raise NotImplementedError(f'Not implemented representation type: '
                                  f'`{self.representation_type}`!\n')


def grid_sample_3d(volume, coordinates):
    """Performs grid sample in 3D space. Given 3D point coordinates, sample
    values from the volume. Note that this function is similar to function
    `torch.nn.functional.grid_sample()` in the case of 5-D inputs.

    Args:
        volume: The given volume, with shape [N, C, D, H, W].
        coordinates: Input 3D point coordinates, with shape
            [N, 1, 1, d * h * w, 3].

    Returns:
        sampled_vals: Sampled values, with shape [N, C, d * h * w, 1, 1].
    """
    N, C, ID, IH, IW = volume.shape
    _, D, H, W, _ = coordinates.shape

    ix = coordinates[..., 0]
    iy = coordinates[..., 1]
    iz = coordinates[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    iz = ((iz + 1) / 2) * (ID - 1)
    with torch.no_grad():
        ix_tnw = torch.floor(ix)
        iy_tnw = torch.floor(iy)
        iz_tnw = torch.floor(iz)

        ix_tne = ix_tnw + 1
        iy_tne = iy_tnw
        iz_tne = iz_tnw

        ix_tsw = ix_tnw
        iy_tsw = iy_tnw + 1
        iz_tsw = iz_tnw

        ix_tse = ix_tnw + 1
        iy_tse = iy_tnw + 1
        iz_tse = iz_tnw

        ix_bnw = ix_tnw
        iy_bnw = iy_tnw
        iz_bnw = iz_tnw + 1

        ix_bne = ix_tnw + 1
        iy_bne = iy_tnw
        iz_bne = iz_tnw + 1

        ix_bsw = ix_tnw
        iy_bsw = iy_tnw + 1
        iz_bsw = iz_tnw + 1

        ix_bse = ix_tnw + 1
        iy_bse = iy_tnw + 1
        iz_bse = iz_tnw + 1

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)

    with torch.no_grad():
        torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
        torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
        torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

        torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
        torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
        torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

        torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
        torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
        torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

        torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
        torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
        torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

    volume = volume.view(N, C, ID * IH * IW)

    tnw_val = torch.gather(volume, 2,
                           (iz_tnw * IW * IH + iy_tnw * IW +
                            ix_tnw).long().view(N, 1,
                                                D * H * W).repeat(1, C, 1))
    tne_val = torch.gather(volume, 2,
                           (iz_tne * IW * IH + iy_tne * IW +
                            ix_tne).long().view(N, 1,
                                                D * H * W).repeat(1, C, 1))
    tsw_val = torch.gather(volume, 2,
                           (iz_tsw * IW * IH + iy_tsw * IW +
                            ix_tsw).long().view(N, 1,
                                                D * H * W).repeat(1, C, 1))
    tse_val = torch.gather(volume, 2,
                           (iz_tse * IW * IH + iy_tse * IW +
                            ix_tse).long().view(N, 1,
                                                D * H * W).repeat(1, C, 1))
    bnw_val = torch.gather(volume, 2,
                           (iz_bnw * IW * IH + iy_bnw * IW +
                            ix_bnw).long().view(N, 1,
                                                D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(volume, 2,
                           (iz_bne * IW * IH + iy_bne * IW +
                            ix_bne).long().view(N, 1,
                                                D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(volume, 2,
                           (iz_bsw * IW * IH + iy_bsw * IW +
                            ix_bsw).long().view(N, 1,
                                                D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(volume, 2,
                           (iz_bse * IW * IH + iy_bse * IW +
                            ix_bse).long().view(N, 1,
                                                D * H * W).repeat(1, C, 1))

    sampled_vals = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
                    tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
                    tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
                    tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
                    bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
                    bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
                    bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
                    bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

    return sampled_vals


def retrieve_from_volume(coordinates, volume):
    """Samples point features from feature volume.

    Args:
        coordinates: Coordinate of input 3D points, with shape [N, R * K, 3].
        volume: Feature volume, with shape [N, C, D, H, W].

    Returns:
        output_features: Output sampled point features, with shape
            [N, R * K, C].
    """
    grid_coords = coordinates[:, None, None]  # [N, 1, 1, R * K, 3]
    output_features = grid_sample_3d(volume, grid_coords)  # [N, C, R * K, 1, 1]
    output_features = output_features[:, :, 0, 0]  # [N, C, R * K]
    output_features = output_features.permute(0, 2, 1)  # [N, R * K, C]

    return output_features


def project_points_onto_planes(points, planes):
    """
    Projects 3D points onto a batch of 2D planes.

    To project a 3D point `P` onto a 2D plane defined by a normal vector `n`
    and a point `Q` that lies on the plane, one can use the following formula:

        P_proj = P - dot(P-Q, n) * n / dot(n, n)

    where:
        `P_proj` is the projected point on the plane;
        `dot()` is the dot product.

    And `Q` can be chosen as the origin (0, 0, 0) of the coordinate system.
    Meanwhile, if n` is a normalized vector, then the projection formula is
    simplified as:

        P_proj = P - dot(P, n) * n

    Args:
        points: Point coordinates, with shape [N, M, 3], where `M` is the
            number of points in each batch and equals `R * K`.
        planes: Planes, with shape [n_planes, 3, 3], where `n_planes`
            is the number of planes. Here, a plane is represented by two vector
            axes and one normal vector. For instance, if a plane is
            represented by:
                `[[0, 0, 1],
                 [0, 1, 0],
                 [1, 0, 0]]`,
            which means that its axes are the third and second axes of the
            coordinate system, and its normal vector is `[1, 0, 0]`.

    Returns:
        projections: Projections, with shape [N * n_planes, R * K, 2].
    """
    plane_normals = planes[:, 2]
    N, M, _ = points.shape  # `M` equals `R * K`.
    n_planes, _ = plane_normals.shape

    # Normalize the normals to unit vectors.
    plane_normals = F.normalize(plane_normals, dim=1)

    # Unsqueeze, expand and reshape tensors.
    points = points.unsqueeze(1).expand(
        -1, n_planes, -1, -1).reshape(N * n_planes, M,
                                      3)  # [N * n_planes, R * K , 3]
    plane_normals = plane_normals.unsqueeze(0).expand(N, -1, -1).reshape(
        N * n_planes, 3)  # [N * n_planes, 3]
    plane_normals = plane_normals.unsqueeze(1).expand(
        -1, M, -1)  # [N * n_planes, R * K, 3]

    # Compute the projections.
    projections = points - torch.sum(points * plane_normals,
                                     dim=-1).unsqueeze(-1) * plane_normals

    # Extract the projection values from different planes.
    plane_axes = planes.unsqueeze(0).expand(N, -1, -1, -1).reshape(
        N * n_planes, 3, 3)
    projections = torch.bmm(projections, plane_axes.permute(0, 2, 1))[..., :2]

    return projections


def retrieve_from_planes(plane_axes,
                         plane_features,
                         coordinates,
                         mode='bilinear',
                         align_corners=False):
    """Samples point features from triplane. Borrowed from

    https://github.com/NVlabs/eg3d/blob/main/eg3d/training/volumetric_rendering/renderer.py

    Args:
        plane_axes: Axes of triplane, with shape [n_planes, 3, 3].
        plane_features: Triplane features, with shape [N, n_planes, C, H, W].
        coordinates: Coordinate of input 3D points, with shape [N, R * K, 3].
        mode: Interpolation mode.

    Returns:
        output_features: Output sampled point features, with shape
            [N, R * K, C].
    """
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape  # `M` equals `R * K`.
    plane_features = plane_features.view(N * n_planes, C, H, W)

    projected_coordinates = project_points_onto_planes(
        coordinates,
        plane_axes).unsqueeze(1)  # [N * n_planes, 1, R * K, 2]
    output_features = F.grid_sample(
        plane_features,
        projected_coordinates.float(),
        mode=mode,
        padding_mode='zeros',
        align_corners=align_corners)  # [N * n_planes, C, 1, R * K]
    output_features = output_features.permute(
        0, 3, 2, 1)  # [N * n_planes, R * K, 1, C]
    output_features = output_features.reshape(N, n_planes, M,
                                              C)  # [N, 3, R * K, C]
    output_features = output_features.mean(1)  # [N, R * K, C]

    return output_features


def retrieve_from_mpi(points, isosurfaces, levels):
    """Get intersections between camera rays and levels.

    Args:
        points : Coordinate of input 3D points, with shape [N, R, K, 3].
        isosurfaces : Isosurface scalars predicted by MPIPredictor.
        levels: Predefined level set values.

    Returns:
        intersections: The intersections between camera rays and the levels,
            with shape [N, R, num_levels - 1, 3]
        is_valid: Whether a level is valid or not, boolean tensor with shape
            [N, R, num_levels - 1, 1]
    """

    s_l = isosurfaces[:, :, :-1]
    s_h = isosurfaces[:, :, 1:]

    K = points.shape[2]
    cost = torch.linspace(K - 1, 0, K - 1).float()
    cost = cost.to(points.device).reshape(1, 1, -1, 1)

    x_interval = []
    s_interval = []
    for l in levels:
        r = (s_h - l <= 0) * (l - s_l <= 0) * 2 - 1
        r = r * cost
        _, indices = torch.max(r, dim=-2, keepdim=True)
        x_l_select = torch.gather(points, -2, indices.expand(-1, -1, -1, 3))
        x_h_select = torch.gather(points, -2, indices.expand(-1, -1, -1, 3) + 1)
        s_l_select = torch.gather(s_l, -2, indices)
        s_h_select = torch.gather(s_h, -2, indices)
        x_interval.append(torch.cat([x_l_select, x_h_select], dim=-2))
        s_interval.append(torch.cat([s_l_select, s_h_select], dim=-2))

    intersections = []
    is_valid = []
    for interval, val, l in zip(x_interval, s_interval, levels):
        x_l = interval[:, :, 0]
        x_h = interval[:, :, 1]
        s_l = val[:, :, 0]
        s_h = val[:, :, 1]
        scale = torch.where(
            torch.abs(s_h - s_l) > 0.05, s_h - s_l,
            torch.ones_like(s_h) * 0.05)
        intersect = torch.where(
            ((s_h - l <= 0) * (l - s_l <= 0)) & (torch.abs(s_h - s_l) > 0.05),
            ((s_h - l) * x_l + (l - s_l) * x_h) / scale, x_h)
        intersections.append(intersect)
        is_valid.append(((s_h - l <= 0) * (l - s_l <= 0)).to(intersect.dtype))

    intersections = torch.stack(intersections, dim=-2)
    is_valid = torch.stack(is_valid, dim=-2)

    return intersections, is_valid
