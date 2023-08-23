# python3.8
"""Contains the implementation of generator described in EpiGRAF.

Paper: https://arxiv.org/pdf/2206.10535.pdf

Official PyTorch implementation: https://github.com/universome/epigraf
"""

import math
import torch
import torch.nn as nn

from models.utils.epigraf_model_helper import FullyConnectedLayer
from models.utils.epigraf_model_helper import MappingNetwork
from models.utils.epigraf_model_helper import SynthesisBlock

from models.rendering import PointSampler
from models.rendering import PointRepresenter
from models.rendering import PointIntegrator

from models.rendering.utils import sample_importance
from models.rendering.utils import unify_attributes
from models.rendering.utils import compute_cam2world_matrix
from utils.misc import linear_schedule


class EpiGRAFGenerator(nn.Module):

    def __init__(
        self,
        z_dim,
        label_dim,
        w_dim,
        image_channels=3,
        label_gen_conditioning_zero=False,
        label_scale=1.0,
        # Settings for mapping network.
        mapping_layers=8,
        camera_cond=False,
        camera_raw_scalars=False,
        include_cam_input=True,
        # Settings for triplane generator.
        coordinate_scale=1.0,
        triplane_resolution=512,
        triplane_channels=32 * 3,
        use_fp32=True,
        num_fp16_res=4,
        channel_base=32768,
        channel_max=512,
        fused_modconv_default='inference_only',
        # Settings for mlp network.
        mlp_hidden_dim=64,
        mlp_output_dim=3,
        mlp_lr_mul=1,
        # Settings for point sampling.
        point_sampling_kwargs=None,
        # Settings for ray marching.
        ray_marching_kwargs=None,
        # Settings for rendering.
        resolution=512,  # Image resolution of final output image.
        rendering_resolution=64,  # Resolution of NeRF rendering.
        num_importance=48,  # Number of points for importance sampling.
        density_noise_std=0.0,  # Std of the gaussian noise added to densities.
        gpc_spoof_p=0.0,  # Probability of generator pose conditioning.
        camera_radius=1.0,  # Radius of camera orbit。
    ):

        super().__init__()
        self.z_dim = z_dim
        self.label_dim = label_dim
        self.w_dim = w_dim
        self.image_channels = image_channels
        self.label_gen_conditioning_zero = label_gen_conditioning_zero
        self.label_scale = label_scale
        self.resolution = resolution

        if point_sampling_kwargs is None:
            point_sampling_kwargs = {}
        if ray_marching_kwargs is None:
            ray_marching_kwargs = {}
        self.point_sampler = PointSampler(**point_sampling_kwargs)
        self.point_integrator = PointIntegrator(**ray_marching_kwargs)

        triplane_axes = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                      [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
                                      [[0, 1, 0], [0, 0, 1], [1, 0, 0]]],
                                     dtype=torch.float32)
        self.point_representer = PointRepresenter(
            representation_type='triplane',
            triplane_axes=triplane_axes,
            coordinate_scale=coordinate_scale)

        self.ref_representation_generator = SynthesisBlocksSequence(
            w_dim=w_dim,
            in_resolution=0,
            out_resolution=triplane_resolution,
            in_channels=0,
            out_channels=triplane_channels,
            architecture='skip',
            num_fp16_res=0 if use_fp32 else num_fp16_res,
            use_noise=True,
            channel_base=channel_base,
            channel_max=channel_max,
            fused_modconv_default=fused_modconv_default)

        num_ws=self.ref_representation_generator.num_ws
        self.mapping_network = MappingNetwork(
            z_dim,
            label_dim,
            w_dim,
            num_ws,
            num_layers=mapping_layers,
            camera_cond=camera_cond,
            camera_raw_scalars=camera_raw_scalars,
            include_cam_input=include_cam_input)

        mlp_input_dim = triplane_channels // 3
        assert mlp_output_dim == image_channels
        self.mlp = MLPNetwork(input_dim=mlp_input_dim,
                              hidden_dim=mlp_hidden_dim,
                              output_dim=mlp_output_dim,
                              lr_mul=mlp_lr_mul)

        # Set up some other arguments.
        self.rendering_resolution = rendering_resolution
        self.num_importance = num_importance
        self.density_noise_std = density_noise_std
        self.gpc_spoof_p = gpc_spoof_p
        self.camera_radius = camera_radius

    def progressive_update(self, cur_kimg):
        self.density_noise_std = linear_schedule(cur_kimg, 1.0, 0.0, 5000)
        self.gpc_spoof_p = linear_schedule(cur_kimg, 1.0, 0.5, 1000)

    def mapping(self,
                z,
                label,
                truncation_psi=1,
                truncation_cutoff=None,
                update_emas=False):
        if self.label_gen_conditioning_zero:
            label = torch.zeros_like(label)
        label = label * self.label_scale
        mapping_results = self.mapping_network(
            z,
            c=None,
            camera_angles=label,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            update_emas=update_emas)
        return mapping_results

    def synthesis(self, wp, camera_angles, patch_params):
        N = wp.shape[0]
        triplanes = self.ref_representation_generator(
            wp[:, :self.ref_representation_generator.num_ws])
        triplanes = triplanes.view(triplanes.shape[0], 3, -1,
                                   triplanes.shape[-2], triplanes.shape[-1])

        camera_angles[:, [1]] = torch.clamp(camera_angles[:, [1]],
                                            1e-5, math.pi - 1e-5)  # [N, 1]
        cam2world_matrix = compute_cam2world_matrix(
            camera_angles, self.camera_radius)  # [N, 4, 4]

        if self.training:
            image_size = self.rendering_resolution
            density_noise_std = self.density_noise_std
        else:
            image_size = self.resolution
            density_noise_std = 0

        point_sampling_result = self.point_sampler(
            batch_size=N,
            image_size=image_size,
            patch_params=patch_params,
            cam2world_matrix=cam2world_matrix)

        points = point_sampling_result['points_world']  # [N, H, W, K, 3]
        ray_dirs = point_sampling_result['rays_world']  # [N, H, W, 3]
        radii_coarse = point_sampling_result['radii']  # [N, H, W, K]
        ray_origins = point_sampling_result['cam2world_matrix'][:, :3,
                                                                -1]  # [N, 3]

        _, H, W, K, _ = points.shape
        R = H * W
        points_coarse = points.reshape(N, R, K, -1)  # [N, R, K, 3]
        points = points.reshape(N, -1, 3)  # [N, R * K, 3]
        ray_dirs = ray_dirs.reshape(N, R, 3)
        ray_origins = ray_origins.unsqueeze(1).repeat(1, R, 1)  # [N, R, 3]
        radii_coarse = radii_coarse.reshape(N, R, K, 1)

        point_features = self.point_representer(
            points,
            ref_representation=triplanes,
            align_corners=True)  # [N, R * K, C]

        color_density_result = self.mlp(point_features)

        densities_coarse = color_density_result['density']  # [N, R * K, 1]
        colors_coarse = color_density_result['color']  # [N, R * K, C]
        densities_coarse = densities_coarse.reshape(N, R, K, 1)
        colors_coarse = colors_coarse.reshape(N, R, K, colors_coarse.shape[-1])

        if self.num_importance > 0:
            # Do the integration along the coarse pass.
            rendering_result = self.point_integrator(
                colors_coarse,
                densities_coarse,
                radii_coarse,
                density_noise_std=density_noise_std)
            weights = rendering_result['weight'] + 1e-5

            # Importance sampling.
            radii_fine = sample_importance(radii_coarse,
                                           weights,
                                           self.num_importance,
                                           smooth_weights=True)
            points = ray_origins.unsqueeze(
                -2) + radii_fine * ray_dirs.unsqueeze(
                -2)  # [N, R, num_importance, 3]
            points_fine = points.reshape(N, R, self.num_importance, -1)
            points = points.reshape(N, -1, 3)  # [N, R * num_importance, 3]

            point_features = self.point_representer(
                points, ref_representation=triplanes)

            color_density_result = self.mlp(point_features)

            densities_fine = color_density_result['density']
            colors_fine = color_density_result['color']
            densities_fine = densities_fine.reshape(N, R, self.num_importance,
                                                    1)
            colors_fine = colors_fine.reshape(N, R, self.num_importance,
                                              colors_fine.shape[-1])

            # Gather coarse and fine results together.
            (all_radiis, all_colors, all_densities,
             all_points) = unify_attributes(radii_coarse,
                                            colors_coarse,
                                            densities_coarse,
                                            radii_fine,
                                            colors_fine,
                                            densities_fine,
                                            points1=points_coarse,
                                            points2=points_fine)

            # Do the integration along the fine pass.
            rendering_result = self.point_integrator(
                all_colors,
                all_densities,
                all_radiis,
                density_noise_std=density_noise_std)

        else:
            # Only do the integration along the coarse pass.
            rendering_result = self.point_integrator(colors_coarse,
                                                     densities_coarse,
                                                     radii_coarse)
            all_points = points_coarse  # [N, R, K, 3]

        rgb_samples = rendering_result['composite_color']
        radii_samples = rendering_result['composite_radial_dist']

        image = rgb_samples.permute(0, 2, 1).reshape(
            N, rgb_samples.shape[-1], H, W).contiguous()
        image_radii = radii_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Get image depth.
        all_points_camera = all_points - ray_origins.unsqueeze(-2)
        points_camera_z = all_points_camera[..., -1:].abs()  # [N, R, K, 1]
        point_weights = rendering_result['weight']  # [N, R, K, 1]
        image_depth = torch.sum(point_weights * points_camera_z,
                                dim=-2)  # [N, R, 1]
        image_depth = image_depth.permute(0, 2, 1).reshape(N, 1, H, W)

        return {
            'image': image,
            'image_depth': image_depth,
            'image_radii': image_radii
        }

    def forward(self,
                z,
                label,
                patch_params=None,
                label_swapped=None,
                style_mixing_prob=0,
                truncation_psi=1,
                truncation_cutoff=None,
                update_emas=False):
        wp = self.mapping(z,
                          label if label_swapped is None else label_swapped,
                          truncation_psi=truncation_psi,
                          truncation_cutoff=truncation_cutoff,
                          update_emas=update_emas)
        if style_mixing_prob > 0:
            cutoff = torch.empty([], dtype=torch.int64,
                                 device=wp.device).random_(1, wp.shape[1])
            cutoff = torch.where(
                torch.rand([], device=wp.device) < style_mixing_prob,
                cutoff, torch.full_like(cutoff, wp.shape[1]))
            wp[:, cutoff:] = self.mapping(torch.randn_like(z),
                                          label,
                                          update_emas=update_emas)[:, cutoff:]
        gen_output = self.synthesis(wp,
                                    camera_angles=label,
                                    patch_params=patch_params)

        return {'wp': wp, **gen_output}


class MLPNetwork(nn.Module):
    """Defines MLPNetwork of EpiGRAF."""

    def __init__(self, input_dim, hidden_dim, output_dim, lr_mul):
        super().__init__()

        self.net = nn.Sequential(
            FullyConnectedLayer(input_dim,
                                hidden_dim,
                                lr_multiplier=lr_mul,
                                activation='lrelu'),
            FullyConnectedLayer(hidden_dim,
                                1 + output_dim,
                                lr_multiplier=lr_mul,
                                activation='linear'))

    def forward(self, point_features):
        N, M, C = point_features.shape
        point_features = point_features.view(N * M, C)

        y = self.net(point_features)
        y = y.view(N, M, -1)

        color = y[..., :-1]
        density = y[..., -1:]

        return {'color': color, 'density': density}


class SynthesisBlocksSequence(torch.nn.Module):

    def __init__(
        self,
        w_dim,
        in_resolution,
        out_resolution,
        in_channels,
        out_channels,
        channel_base=32768,
        channel_max=512,
        num_fp16_res=4,
        **block_kwargs,
    ):
        assert in_resolution == 0 or (in_resolution >= 4 and
                                      math.log2(in_resolution).is_integer())
        assert out_resolution >= 4 and math.log2(out_resolution).is_integer()
        assert in_resolution < out_resolution

        super().__init__()

        self.w_dim = w_dim
        self.out_resolution = out_resolution  # 512
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_fp16_res = num_fp16_res

        in_resolution_log2 = 2 if in_resolution == 0 else (
            int(math.log2(in_resolution)) + 1)
        out_resolution_log2 = int(math.log2(out_resolution))
        self.block_resolutions = [
            2**i for i in range(in_resolution_log2, out_resolution_log2 + 1)
        ]  # [4, 8, 16, 32, 64, 128, 256, 512]
        out_channels_dict = {
            res: min(channel_base // res, channel_max)
            for res in self.block_resolutions
        }
        fp16_resolution = max(2 ** (out_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for block_idx, res in enumerate(self.block_resolutions):
            cur_in_channels = out_channels_dict[
                res // 2] if block_idx > 0 else in_channels
            cur_out_channels = out_channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.out_resolution)
            block = SynthesisBlock(cur_in_channels,
                                   cur_out_channels,
                                   w_dim=w_dim,
                                   resolution=res,
                                   img_channels=self.out_channels,
                                   is_last=is_last,
                                   use_fp16=use_fp16,
                                   **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, x: torch.Tensor = None, **block_kwargs):
        block_ws = []
        ws = ws.to(torch.float32)
        w_idx = 0
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            block_ws.append(
                ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
            w_idx += block.num_conv

        img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img
