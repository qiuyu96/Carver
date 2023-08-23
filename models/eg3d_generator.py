# python3.8
"""Contains the implementation of generator described in EG3D.

Paper: https://arxiv.org/pdf/2112.07945.pdf

Official PyTorch implementation: https://github.com/NVlabs/eg3d
"""

import torch
import torch.nn as nn

from models.utils.eg3d_model_helper import Generator as StyleGAN2Backbone
from models.utils.eg3d_model_helper import FullyConnectedLayer
from models.utils.eg3d_superres import SuperresolutionHybrid2X
from models.utils.eg3d_superres import SuperresolutionHybrid4X
from models.utils.eg3d_superres import SuperresolutionHybrid8XDC

from models.rendering import PointSampler
from models.rendering import PointRepresenter
from models.rendering import PointIntegrator

from models.rendering.utils import sample_importance
from models.rendering.utils import unify_attributes


class EG3DGenerator(nn.Module):

    def __init__(
        self,
        # Settings for mapping network.
        z_dim,
        label_dim,
        w_dim,
        mapping_layers,
        image_channels=3,
        label_gen_conditioning_zero=False,
        label_scale=1.0,
        # Settings for triplane.
        triplane_resolution=256,
        triplane_channels=32 * 3,
        num_fp16_res=4,
        conv_clamp=256,
        coordinate_scale=1.0,
        # Settings for post CNN.
        sr_num_fp16_res=0,
        sr_channel_base=2**15,
        sr_channel_max=512,
        sr_antialias=True,
        sr_fused_modconv_default='inference_only',
        sr_noise_mode='none',
        # Settings for mlp network.
        mlp_hidden_dim=64,
        mlp_output_dim=32,
        mlp_lr_mul=1,
        # Settings for point sampling.
        point_sampling_kwargs=None,
        # Settings for ray marching.
        ray_marching_kwargs=None,
        # Settings for rendering.
        resolution=512,  # Image resolution of final output image.
        rendering_resolution=64,  # Resolution of NeRF rendering.
        num_importance=48,  # Number of points for importance sampling.
        # Settings for rendering options of evaluation.
        avg_camera_radius=2.7,
        avg_camera_pivot=[0, 0, 0.2],
        # Whether to use random pose from the specified distribution.
        random_pose=False,
        # Whether to use raw triplane axes as the official code.
        use_raw_triplane_axes=False):

        super().__init__()
        self.z_dim = z_dim
        self.label_dim = label_dim
        self.w_dim = w_dim
        self.image_channels = image_channels
        self.label_gen_conditioning_zero = label_gen_conditioning_zero
        self.label_scale = label_scale
        self.resolution = resolution

        # Set up the rendering related module.
        if point_sampling_kwargs is None:
            point_sampling_kwargs = {}
        if ray_marching_kwargs is None:
            ray_marching_kwargs = {}
        self.point_sampler = PointSampler(**point_sampling_kwargs)
        self.point_integrator = PointIntegrator(**ray_marching_kwargs)

        if use_raw_triplane_axes:
            # The axes of the triplane in this case are actually incorrect
            # as stated in https://github.com/NVlabs/eg3d/issues/67
            triplane_axes = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                          [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
                                          [[0, 0, 1], [1, 0, 0], [0, 1, 0]]],
                                         dtype=torch.float32)
        else:
            triplane_axes = None

        self.point_representer = PointRepresenter(
            representation_type='triplane',
            triplane_axes=triplane_axes,
            coordinate_scale=coordinate_scale)

        # Set up the reference representation generator.
        self.backbone = StyleGAN2Backbone(
            z_dim,
            label_dim,
            w_dim,
            img_resolution=triplane_resolution,
            img_channels=triplane_channels,
            num_fp16_res=num_fp16_res,
            conv_clamp=conv_clamp,
            mapping_kwargs=dict(num_layers=mapping_layers))

        # Set up the post CNN.
        self.sr_noise_mode = sr_noise_mode
        sr_kwargs = {
            'channels': mlp_output_dim,
            'img_resolution': resolution,
            'sr_num_fp16_res': sr_num_fp16_res,
            'sr_antialias': sr_antialias,
            'channel_base': sr_channel_base,
            'channel_max': sr_channel_max,
            'fused_modconv_default': sr_fused_modconv_default
        }
        if resolution == 128:
            self.post_cnn = SuperresolutionHybrid2X(**sr_kwargs)
        elif resolution == 256:
            self.post_cnn = SuperresolutionHybrid4X(**sr_kwargs)
        elif resolution == 512:
            self.post_cnn = SuperresolutionHybrid8XDC(**sr_kwargs)
        else:
            raise TypeError(f'Unsupported image resolution: {resolution}!')

        # Set up the MLP network.
        mlp_input_dim = triplane_channels // 3
        self.mlp = MLPNetwork(input_dim=mlp_input_dim,
                              hidden_dim=mlp_hidden_dim,
                              output_dim=mlp_output_dim,
                              lr_mul=mlp_lr_mul)

        # Set up some other arguments.
        self.register_buffer('rendering_resolution',
                             torch.tensor(rendering_resolution))
        self.num_importance = num_importance
        self.random_pose = random_pose
        self.eval_rendering_options = {
            'avg_camera_radius': avg_camera_radius,
            'avg_camera_pivot': avg_camera_pivot
        }

    def mapping(self,
                z,
                label,
                truncation_psi=1,
                truncation_cutoff=None,
                update_emas=False):
        if self.label_gen_conditioning_zero:
            label = torch.zeros_like(label)
        label = label * self.label_scale
        results = self.backbone.mapping(z,
                                        label,
                                        truncation_psi=truncation_psi,
                                        truncation_cutoff=truncation_cutoff,
                                        update_emas=update_emas)
        return results

    def synthesis(self,
                  wp,
                  label,
                  cam2world_matrix=None,
                  update_emas=False,
                  **synthesis_kwargs):
        N = wp.shape[0]

        if self.random_pose:
            cam2world_matrix = cam2world_matrix
            point_sampling_kwargs = {}
        else:
            cam2world_matrix = label[:, :16].view(-1, 4, 4)
            image_boundary_value = 0.5 * (1 - 1 / self.rendering_resolution)
            point_sampling_kwargs = {
                'image_boundary_value': image_boundary_value
            }

        triplanes = self.backbone.synthesis(wp,
                                            update_emas=update_emas,
                                            **synthesis_kwargs)
        triplanes = triplanes.view(triplanes.shape[0], 3, -1,
                                   triplanes.shape[-2], triplanes.shape[-1])

        point_sampling_result = self.point_sampler(
            batch_size=N,
            image_size=int(self.rendering_resolution),
            cam2world_matrix=cam2world_matrix,
            **point_sampling_kwargs)

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
            points, ref_representation=triplanes)  # [N, R * K, C]

        color_density_result = self.mlp(point_features)

        densities_coarse = color_density_result['density']  # [N, R * K, 1]
        colors_coarse = color_density_result['color']  # [N, R * K, C]
        densities_coarse = densities_coarse.reshape(N, R, K, 1)
        colors_coarse = colors_coarse.reshape(N, R, K, colors_coarse.shape[-1])

        if self.num_importance > 0:
            # Do the integration along the coarse pass.
            rendering_result = self.point_integrator(colors_coarse,
                                                     densities_coarse,
                                                     radii_coarse)
            weights = rendering_result['weight']

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
            rendering_result = self.point_integrator(all_colors,
                                                     all_densities,
                                                     all_radiis)

        else:
            # Only do the integration along the coarse pass.
            rendering_result = self.point_integrator(colors_coarse,
                                                     densities_coarse,
                                                     radii_coarse)
            all_points = points_coarse  # [N, R, K, 3]

        feature_samples = rendering_result['composite_color']
        radii_samples = rendering_result['composite_radial_dist']

        # Reshape to keep consistent with 'raw' NeRF-rendered image.
        feature_image = feature_samples.permute(0, 2, 1).reshape(
            N, feature_samples.shape[-1], H, W).contiguous()
        image_radii = radii_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Get image depth.
        all_points_camera = all_points - ray_origins.unsqueeze(-2)
        points_camera_z = all_points_camera[..., -1:].abs()  # [N, R, K, 1]
        points_camera_z = (points_camera_z[:, :, :-1, :] +
                           points_camera_z[:, :, 1:, :]) / 2  # [N, R, K-1, 1]
        point_weights = rendering_result['weight']  # [N, R, K-1, 1]
        image_depth = torch.sum(point_weights * points_camera_z,
                                dim=-2)  # [N, R, 1]
        image_depth = image_depth.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run the post CNN to get final image.
        # Here, the post CNN is a super-resolution network.
        rgb_image = feature_image[:, :3]
        sr_kwargs = {
            k: synthesis_kwargs[k]
            for k in synthesis_kwargs.keys() if k != 'noise_mode'
        }
        sr_image = self.post_cnn(rgb_image,
                                 feature_image,
                                 wp,
                                 noise_mode=self.sr_noise_mode,
                                 **sr_kwargs)

        results = {
            'image': sr_image,
            'image_raw': rgb_image,
            'image_depth': image_depth,
            'image_radii': image_radii
        }

        if self.random_pose:
            results['label'] = torch.Tensor(wp.shape[0],
                                            self.label_dim).to(wp.device)

        return results

    def forward(self,
                z,
                label,
                label_swapped=None,
                style_mixing_prob=0,
                truncation_psi=1,
                truncation_cutoff=None,
                update_emas=False,
                density_reg=False,
                coordinates=None,
                **synthesis_kwargs):
        wp = self.mapping(z,
                          label if label_swapped is None else label_swapped,
                          truncation_psi=truncation_psi,
                          truncation_cutoff=truncation_cutoff,
                          update_emas=update_emas)
        if style_mixing_prob > 0:
            cutoff = torch.empty([], dtype=torch.int64,
                                 device=wp.device).random_(1, wp.shape[1])
            cutoff = torch.where(
                torch.rand([], device=wp.device) < style_mixing_prob, cutoff,
                torch.full_like(cutoff, wp.shape[1]))
            wp[:, cutoff:] = self.mapping(torch.randn_like(z),
                                          label,
                                          update_emas=update_emas)[:, cutoff:]
        if density_reg:
            # Only for density regularization in training process.
            assert coordinates is not None
            sample_density = self.sample_mixed(coordinates,
                                               wp,
                                               update_emas=False)['density']

            results = {'wp': wp, 'sample_density': sample_density}

        else:
            gen_output = self.synthesis(wp,
                                        label,
                                        update_emas=update_emas,
                                        **synthesis_kwargs)

            results = {'wp': wp, **gen_output}

        return results

    def sample(self,
               coordinates,
               z,
               label,
               truncation_psi=1,
               truncation_cutoff=None,
               update_emas=False,
               **synthesis_kwargs):
        """Samples densities and colors.

        By feeding the coordinates and latent codes into the trained network,
        one can sample densities and colors. This function is primarily used
        for evaluation purposes.
        """
        wp = self.mapping(z,
                          label,
                          truncation_psi=truncation_psi,
                          truncation_cutoff=truncation_cutoff,
                          update_emas=update_emas)

        triplanes = self.backbone.synthesis(wp,
                                            update_emas=update_emas,
                                            **synthesis_kwargs)
        triplanes = triplanes.view(triplanes.shape[0], 3, -1,
                                   triplanes.shape[-2], triplanes.shape[-1])

        point_features = self.point_representer(
            coordinates, ref_representation=triplanes)
        color_density_result = self.mlp(point_features)

        return color_density_result

    def sample_mixed(self,
                     coordinates,
                     wp,
                     truncation_psi=1,
                     truncation_cutoff=None,
                     update_emas=False,
                     **synthesis_kwargs):
        triplanes = self.backbone.synthesis(wp,
                                            update_emas=update_emas,
                                            **synthesis_kwargs)
        triplanes = triplanes.view(triplanes.shape[0], 3, -1,
                                   triplanes.shape[-2], triplanes.shape[-1])

        point_features = self.point_representer(
            coordinates, ref_representation=triplanes)
        color_density_result = self.mlp(point_features)

        return color_density_result


class MLPNetwork(nn.Module):
    """Defines fully-connected layer head in EG3D."""

    def __init__(self, input_dim, hidden_dim, output_dim, lr_mul):
        super().__init__()

        self.net = nn.Sequential(
            FullyConnectedLayer(input_dim, hidden_dim, lr_multiplier=lr_mul),
            nn.Softplus(),
            FullyConnectedLayer(hidden_dim,
                                1 + output_dim,
                                lr_multiplier=lr_mul))

    def forward(self, point_features):
        N, M, C = point_features.shape
        point_features = point_features.view(N * M, C)

        y = self.net(point_features)
        y = y.view(N, M, -1)

        color = torch.sigmoid(y[..., 1:]) * (1 + 2 * 0.001) - 0.001
        density = y[..., 0:1]

        return {'color': color, 'density': density}
