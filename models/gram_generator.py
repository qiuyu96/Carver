# python3.8
"""Contains the implementation of generator described in GRAM.

Paper: https://arxiv.org/pdf/2112.08867.pdf

Official PyTorch implementation: https://github.com/microsoft/GRAM
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from models.rendering import PointSampler
from models.rendering import PointRepresenter
from models.rendering import PointIntegrator


class GRAMGenerator(nn.Module):
    """Defines the generator network in GRAM."""

    def __init__(
            self,
            # Settings for mapping network.
            z_dim=256,
            w_dim=256,
            repeat_w=False,
            normalize_z=False,
            mapping_layers=3,
            mapping_hidden_dim=256,
            # Settings for conditional generation.
            label_dim=0,
            embedding_dim=512,
            normalize_embedding=True,
            normalize_embedding_latent=False,
            # Settings for multi-plane image (MPI).
            levels_start=23,
            levels_end=8,
            num_planes=24,
            hidden_dim_sample=128,
            center_z=-1.5,
            use_mpi_background=True,
            # Settings for synthesis network.
            resolution=256,
            synthesis_input_dim=3,
            synthesis_output_dim=256,
            synthesis_layers=8,
            grid_scale=0.24,
            eps=1e-8,
            # Settings for background depth.
            bg_depth_start=-0.12,
            bg_depth_end=-0.12,
            num_bg_depth=1,
            # Settings for rendering.
            rendering_resolution=256,  # Resolution of NeRF rendering.
            point_sampling_kwargs=None,
            ray_marching_kwargs=None):
        """Initializes with basic settings."""

        super().__init__()

        self.z_dim = z_dim
        self.w_dim = w_dim
        self.repeat_w = repeat_w
        self.normalize_z = normalize_z
        self.mapping_layers = mapping_layers

        self.latent_dim = (z_dim, )
        self.label_dim = label_dim
        self.embedding_dim = embedding_dim
        self.normalize_embedding = normalize_embedding
        self.normalize_embedding_latent = normalize_embedding_latent

        self.resolution = resolution
        self.num_layers = synthesis_layers
        self.eps = eps
        self.use_mpi_background = use_mpi_background

        if self.repeat_w:
            self.mapping_space_dim = self.w_dim
        else:
            self.mapping_space_dim = self.w_dim * (self.num_layers + 1)

        if point_sampling_kwargs is None:
            point_sampling_kwargs = {}
        if ray_marching_kwargs is None:
            ray_marching_kwargs = {}

        self.point_sampler = PointSampler(**point_sampling_kwargs)
        self.point_integrator = PointIntegrator(**ray_marching_kwargs)
        if self.use_mpi_background:
            mpi_levels = torch.linspace(levels_start, levels_end,
                                        num_planes - 1)
        else:
            mpi_levels = torch.linspace(levels_start, levels_end, num_planes)
        self.point_representer = PointRepresenter(representation_type='mpi',
                                                  mpi_levels=mpi_levels)

        # Mapping Network to tranform latent codes from Z-Space into W-Space.
        self.mapping = CustomMappingNetwork(
            z_dim=z_dim,
            map_hidden_dim=mapping_hidden_dim,
            map_output_dim=(synthesis_layers + 1) * 2 * mapping_hidden_dim)

        # Set up the reference representation generator.
        self.ref_representation_generator = MPIPredictor(
            hidden_dim_sample=hidden_dim_sample, center=(0, 0, center_z))

        # Set up the mlp network.
        self.mlp = MLPNetwork(w_dim=w_dim,
                              in_channels=synthesis_input_dim,
                              num_layers=synthesis_layers,
                              out_channels=synthesis_output_dim,
                              grid_scale=grid_scale)

        # This is used for truncation trick.
        if self.repeat_w:
            self.register_buffer('w_avg', torch.zeros(w_dim))
        else:
            self.register_buffer('w_avg', torch.zeros(self.num_layers * w_dim))

        # Some other arguments.
        self.rendering_resolution = rendering_resolution
        bg_depth_world = torch.linspace(bg_depth_start, bg_depth_end,
                                        num_bg_depth)
        self.bg_depth_world = bg_depth_world.view(1, 1, num_bg_depth)

    def forward(
        self,
        z,
        label=None,
        lod=None,
        w_moving_decay=None,
        sync_w_avg=False,
        style_mixing_prob=None,
        noise_std=None,
        trunc_psi=None,
        trunc_layers=None,
        enable_amp=False):

        mapping_results = self.mapping(z)
        wp = mapping_results.pop('wp')

        with autocast(enabled=enable_amp):
            batch_size = wp.shape[0]

            point_sampling_result = self.point_sampler(
                batch_size=batch_size,
                image_size=int(self.rendering_resolution),
                cam2world_matrix=None)

            points = point_sampling_result['points_world']  # [N, H, W, K, 3]
            ray_dirs = point_sampling_result['rays_world']  # [N, H, W, 3]
            radii_coarse = point_sampling_result['radii']  # [N, H, W, K]
            ray_origins = point_sampling_result[
                'cam2world_matrix'][:, :3, -1]  # [N, 3]

            # Vertical angle of camera in world coordinate.
            camera_polar = point_sampling_result['camera_polar']  # [N]
            # Horizontal angle of camera in world coordinate.
            camera_azimuthal = point_sampling_result['camera_azimuthal']  # [N]
            camera_polar = camera_polar.unsqueeze(-1)
            camera_azimuthal = camera_azimuthal.unsqueeze(-1)

            N, H, W, K, _ = points.shape
            assert N == batch_size
            R = H * W

            points = points.reshape(N, R, K, -1)  # [N, R, K, 3]
            ray_dirs = ray_dirs.reshape(N, R, -1)  # [N, R, 3]
            ray_origins = ray_origins.unsqueeze(1).repeat(1, R, 1)  # [N, R, 3]
            radii_coarse = radii_coarse.reshape(N, R, K, -1)  # [N, R, K, 1]

            isosurfaces = self.ref_representation_generator(points)

            transformed_points, is_valid = self.point_representer(
                    points, ref_representation=isosurfaces)

            if self.use_mpi_background:
                bg_depth_world = self.bg_depth_world.to(z.device)
                bg_depth_cam =  bg_depth_world - ray_origins[..., -1:]
                bg_radii = bg_depth_cam / ray_dirs[..., -1:]
                bg_radii = bg_radii.unsqueeze(-1)
                transformed_points_bg = ray_origins.unsqueeze(
                    2) + ray_dirs.unsqueeze(2) * bg_radii
                transformed_points = torch.cat(
                    [transformed_points, transformed_points_bg], dim=-2)
                is_valid = torch.cat(
                    [is_valid,
                     torch.ones(N, R, 1, 1).to(is_valid.device)],
                    dim=-2)

            color_density_result = self.mlp(transformed_points, wp, ray_dirs)

            all_radiis = torch.sqrt(
                torch.sum((transformed_points - ray_origins.unsqueeze(2))**2,
                          dim=-1, keepdim=True))
            _, indices = torch.sort(all_radiis, dim=-2)
            radii_coarse = torch.gather(all_radiis, -2, indices)
            colors_coarse = torch.gather(color_density_result['color'], -2,
                                         indices.expand(-1, -1, -1, 3))
            densities_coarse = torch.gather(color_density_result['density'], -2,
                                            indices)
            is_valid = torch.gather(is_valid, -2, indices)
            bg_index = torch.argmax(indices, dim=-2)

            rendering_result = self.point_integrator(
                colors_coarse,
                densities_coarse,
                radii_coarse,
                is_valid=is_valid,
                bg_index=bg_index)

        image = rendering_result['composite_color'].reshape(
            z.shape[0], self.resolution, self.resolution,
            -1).permute(0, 3, 1, 2)

        camera = torch.cat([camera_polar, camera_azimuthal], -1)

        return {
            **mapping_results, 'image': image,
            'camera': camera,
            'latent': z
        }


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight,
                                      a=0.2,
                                      mode='fan_in',
                                      nonlinearity='leaky_relu')


class CustomMappingNetwork(nn.Module):

    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()

        self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(map_hidden_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(map_hidden_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(map_hidden_dim, map_output_dim))

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[
            ..., :frequencies_offsets.shape[-1] // 2]
        phase_shifts = frequencies_offsets[...,
                                           frequencies_offsets.shape[-1] // 2:]
        results = {
            'frequencies': frequencies,
            'phase_shifts': phase_shifts,
            'w': torch.cat([frequencies, phase_shifts], axis=1),
            'wp': torch.cat([frequencies, phase_shifts], axis=1),
            'label': None,
            'z': z
        }
        return results


class MLPNetwork(nn.Module):
    """Defines MLP Network in Pi-GAN, including fully-connected layer head."""

    def __init__(self,
                 w_dim,
                 in_channels,
                 num_layers,
                 out_channels,
                 grid_scale=0.24,
                 hidden_dim=256):
        super().__init__()

        self.in_channels = in_channels
        self.w_dim = w_dim
        self.out_channels = out_channels

        self.grid_warper = UniformBoxWarp(grid_scale)

        self.mlp_network = nn.ModuleList([
            FiLMLayer_GRAM(3, hidden_dim),
            FiLMLayer_GRAM(hidden_dim, hidden_dim),
            FiLMLayer_GRAM(hidden_dim, hidden_dim),
            FiLMLayer_GRAM(hidden_dim, hidden_dim),
            FiLMLayer_GRAM(hidden_dim, hidden_dim),
            FiLMLayer_GRAM(hidden_dim, hidden_dim),
            FiLMLayer_GRAM(hidden_dim, hidden_dim),
            FiLMLayer_GRAM(hidden_dim, hidden_dim),
        ])

        self.color_layer = nn.ModuleList(
            [FiLMLayer_GRAM(hidden_dim + 3, hidden_dim)])

        self.output_sigma = nn.ModuleList([
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
        ])

        self.output_color = nn.ModuleList([
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
        ])
        self.mlp_network.apply(freq_init(25))
        self.output_sigma.apply(freq_init(25))
        self.color_layer.apply(freq_init(25))
        self.output_color.apply(freq_init(25))
        self.mlp_network[0].apply(first_film_init)

    def forward(self, point_features, wp, ray_dirs, lod=None):

        num_dims = point_features.ndim
        assert num_dims in [3, 4, 5]
        if num_dims == 5:
            N, H, W, K, C = point_features.shape
            point_features = point_features.reshape(N, H * W * K, C)
        elif num_dims == 4:
            N, R, K, C = point_features.shape
            point_features = point_features.reshape(N, R * K, C)

        x = self.grid_warper(point_features)
        sigma = 0
        rgb = 0
        ray_dirs = ray_dirs.reshape(N, R, -1, ray_dirs.shape[-1])
        ray_dirs = ray_dirs[:, :, 0:1, :]
        ray_dirs = ray_dirs.expand(N, R, K, ray_dirs.shape[-1])
        ray_dirs = ray_dirs.reshape(N, R * K, ray_dirs.shape[-1])
        for idx, layer in enumerate(self.mlp_network):
            start = idx * 256
            end = (idx + 1) * 256
            x = layer(x, wp[:, start:end] * 15 + 30,
                      wp[:, start + 2304:end + 2304])
            if idx > 0:
                layer_sigma = self.output_sigma[idx - 1](x)
                if not idx == 7:
                    layer_rgb_feature = x
                else:
                    layer_rgb_feature = self.color_layer[0](
                        torch.cat([ray_dirs, x], dim=-1),
                        wp[:,
                           len(self.mlp_network) * 256:
                           (len(self.mlp_network) + 1) * 256] * 15 + 30,
                        wp[:,
                           len(self.mlp_network) * 256 +
                           2304:(len(self.mlp_network) + 1) * 256 + 2304])
                layer_rgb = self.output_color[idx - 1](layer_rgb_feature)

                sigma += layer_sigma
                rgb += layer_rgb

        sigma = sigma.reshape(N, R, K, sigma.shape[-1])
        rgb = rgb.reshape(N, R, K, rgb.shape[-1])

        results = {'density': sigma, 'color': rgb}

        return results


class FiLMLayer(nn.Module):

    def __init__(self, input_dim, output_dim, w_dim, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w_dim = w_dim

        self.layer = nn.Linear(input_dim, output_dim)
        self.style = nn.Linear(w_dim, output_dim * 2)

    def init_weights(self, first=False):
        # initial with 25 frequency
        if not first:
            self.layer.apply(freq_init(25))
        else:
            self.layer.apply(first_film_init)
        # kaiming initial && scale 1/4
        nn.init.kaiming_normal_(self.style.weight,
                                a=0.2,
                                mode='fan_in',
                                nonlinearity='leaky_relu')
        with torch.no_grad():
            self.style.weight *= 0.25

    def extra_repr(self):
        return (f'in_ch={self.input_dim}, '
                f'out_ch={self.output_dim}, '
                f'w_ch={self.w_dim}')

    def forward(self, x, wp):
        x = self.layer(x)
        style = self.style(wp)
        style_split = style.unsqueeze(1).chunk(2, dim=2)
        freq = style_split[0]
        # Scale for sin activation
        freq = freq * 15 + 30
        phase_shift = style_split[1]
        return torch.sin(freq * x + phase_shift)


class FiLMLayer_GRAM(nn.Module):

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        freq = freq.unsqueeze(1).expand_as(x)
        phase_shift = phase_shift.unsqueeze(1).expand_as(x)
        return torch.sin(freq * x + phase_shift)


class PixelNormLayer(nn.Module):
    """Implements pixel-wise feature vector normalization layer."""

    def __init__(self, dim, eps):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def extra_repr(self):
        return f'dim={self.dim}, epsilon={self.eps}'

    def forward(self, x):
        scale = (x.square().mean(dim=self.dim, keepdim=True) +
                 self.eps).rsqrt()
        return x * scale


class UniformBoxWarp(nn.Module):

    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2 / sidelength

    def forward(self, coordinates):
        return coordinates * self.scale_factor


def first_film_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def freq_init(freq):

    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq,
                                  np.sqrt(6 / num_input) / freq)

    return init


class EqualLinear(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        bias=True,
        bias_init=0,
        lr_mul=1,
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        out = F.linear(input,
                       self.weight * self.scale,
                       bias=self.bias * self.lr_mul)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.weight.shape[1]}, '
                '{self.weight.shape[0]})')


def geometry_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_output = m.weight.size(0)
            m.weight.normal_(0, np.sqrt(2 / num_output))
            nn.init.constant_(m.bias, 0)


def geometry_init_last_layer(radius):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                nn.init.constant_(m.weight, 10 * np.sqrt(np.pi / num_input))
                nn.init.constant_(m.bias, -radius)

    return init


class MPIPredictor(nn.Module):
    def __init__(self,
                 hidden_dim_sample=128,
                 layer_num_sample=3,
                 center=(0, 0, -1.5),
                 init_radius=0):
        super().__init__()
        self.hidden_dim = hidden_dim_sample
        self.layer_num = layer_num_sample

        self.network = [nn.Linear(3, self.hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(self.layer_num - 1):
            self.network += [
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True)
            ]

        self.network = nn.Sequential(*self.network)

        self.output_layer = nn.Linear(self.hidden_dim, 1)

        self.network.apply(geometry_init)
        self.output_layer.apply(geometry_init_last_layer(init_radius))
        self.center = torch.tensor(center)

        self.gridwarper = UniformBoxWarp(0.24)

    def forward(self, input):
        x = input
        x = self.gridwarper(x)
        x = x - self.center.to(x.device)
        x = self.network(x)
        s = self.output_layer(x)

        return s
