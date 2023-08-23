# python3.8
"""Contains the implementation of 3D-aware generator for ablation."""

import math
from einops import rearrange

import torch
import torch.nn as nn
import torch.autograd as autograd

from models.utils.eg3d_model_helper import Generator as StyleGAN2Backbone
from models.utils.eg3d_model_helper import FullyConnectedLayer
from models.utils.eg3d_model_helper import MappingNetwork
from models.utils.eg3d_superres import SuperresolutionHybrid2X
from models.utils.eg3d_superres import SuperresolutionHybrid4X
from models.utils.eg3d_superres import SuperresolutionHybrid8XDC

from models.rendering import PointSampler
from models.rendering import PointRepresenter
from models.rendering import PointIntegrator
from models.volumegan_generator import FeatureVolume
from models.rendering.utils import PositionEncoder
from models.rendering.utils import sample_importance
from models.rendering.utils import unify_attributes
from models.rendering.utils import dividable

from models.pigan_generator import FiLMLayer
from models.pigan_generator import freq_init

from models.stylenerf_generator import Style2Layer
from models.stylenerf_generator import ToRGBLayer


class Ablation3DGenerator(nn.Module):

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
        # Settings for feature volume generator.
        fv_feat_res=32,
        fv_init_res=4,
        fv_base_channels=256,
        fv_output_channels=32,
        fv_w_dim=512,
        # Settings for points encoder.
        pe_input_dim=3,
        pe_num_freqs=10,
        factor=math.pi,
        include_input=False,
        # Settings for post CNN.
        use_upsampler=True,
        sr_num_fp16_res=0,
        sr_channel_base=2**15,
        sr_channel_max=512,
        sr_antialias=True,
        sr_fused_modconv_default='inference_only',
        sr_noise_mode='none',
        # Settings for mlp network.
        mlp_type='eg3d',
        mlp_depth=2,
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
        # Whether use random pose from the specified distribution.
        random_pose=False,
        # Settings for reference mode.
        ref_mode='triplane',
        bound=None,
        use_raw_triplane_axes=False,
        use_positional_encoding=False,
        use_sdf=False,
        return_eikonal=False):

        super().__init__()
        self.z_dim = z_dim
        self.label_dim = label_dim
        self.w_dim = w_dim
        self.image_channels = image_channels
        self.label_gen_conditioning_zero = label_gen_conditioning_zero
        self.label_scale = label_scale
        self.resolution = resolution
        self.ref_mode = ref_mode
        self.use_positional_encoding = use_positional_encoding
        self.mlp_type = mlp_type
        self.use_upsampler = use_upsampler

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
            representation_type=ref_mode,
            triplane_axes=triplane_axes,
            coordinate_scale=coordinate_scale)

        mlp_input_dim = 0
        num_ws = 0

        # Set up the position encoder.
        if use_positional_encoding:
            self.position_encoder = PositionEncoder(
                input_dim=pe_input_dim,
                max_freq_log2=pe_num_freqs - 1,
                num_freqs=pe_num_freqs,
                factor=factor,
                include_input=include_input)
            mlp_input_dim += self.position_encoder.get_out_dim()

        # Set up the reference representation generator.
        if self.ref_mode == 'coordinate':
            if not self.use_positional_encoding:
                mlp_input_dim += 3
        elif self.ref_mode == 'triplane':
            self.triplane_generator = StyleGAN2Backbone(
                z_dim,
                label_dim,
                w_dim,
                img_resolution=triplane_resolution,
                img_channels=triplane_channels,
                num_fp16_res=num_fp16_res,
                conv_clamp=conv_clamp,
                mapping_kwargs=dict(num_layers=mapping_layers))
            num_ws += (self.triplane_generator.synthesis.num_ws)
            mlp_input_dim += triplane_channels // 3
        elif self.ref_mode == 'volume':
            self.fv_generator = FeatureVolume(
                feat_res=fv_feat_res,
                init_res=fv_init_res,
                base_channels=fv_base_channels,
                output_channels=fv_output_channels,
                w_dim=fv_w_dim)
            num_ws += 2
            mlp_input_dim += fv_output_channels
        elif self.ref_mode == 'hybrid':
            self.triplane_generator = StyleGAN2Backbone(
                z_dim,
                label_dim,
                w_dim,
                img_resolution=triplane_resolution,
                img_channels=triplane_channels,
                num_fp16_res=num_fp16_res,
                conv_clamp=conv_clamp)
            self.fv_generator = FeatureVolume(
                feat_res=fv_feat_res,
                init_res=fv_init_res,
                base_channels=fv_base_channels,
                output_channels=fv_output_channels,
                w_dim=fv_w_dim)
            num_ws += (self.triplane_generator.synthesis.num_ws)
            mlp_input_dim += (triplane_channels // 3 + fv_output_channels)
        else:
            raise TypeError(f'Unsupported ref mode: {self.ref_mode}!')

        if mlp_type == 'pigan':
            num_ws += (mlp_depth + 1)
        elif mlp_type == 'stylenerf':
            num_ws += (mlp_depth + 3)

        # Set up the mapping network.
        self.mapping_network = MappingNetwork(
            z_dim=z_dim,
            c_dim=label_dim,
            w_dim=w_dim,
            num_ws=num_ws,
            num_layers=mapping_layers)

        # Set up the post CNN (Upsampler).
        if self.use_upsampler:
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
        if mlp_type == 'eg3d':
            self.mlp = EG3DMLPNetwork(input_dim=mlp_input_dim,
                                      hidden_dim=mlp_hidden_dim,
                                      output_dim=mlp_output_dim,
                                      depth=mlp_depth,
                                      lr_mul=mlp_lr_mul)
        elif mlp_type == 'pigan':
            self.mlp = PiGANMLPNetwork(w_dim=w_dim,
                                       input_dim=mlp_input_dim,
                                       hidden_dim=mlp_hidden_dim,
                                       output_dim=mlp_output_dim,
                                       depth=mlp_depth)
            self.mlp.init_weights()
        elif mlp_type == 'stylenerf':
            self.mlp = StyleNeRFMLPNetwork(w_dim=w_dim,
                                           input_dim=mlp_input_dim,
                                           hidden_dim=mlp_hidden_dim,
                                           output_dim=mlp_output_dim,
                                           depth=mlp_depth)
        else:
            raise TypeError(f'Unspported mlp type: {mlp_type}!')

        if use_sdf:
            self.sigmoid_beta = nn.Parameter(0.1 * torch.ones(1).float())

        # Set up some other arguments.
        self.register_buffer('rendering_resolution',
                             torch.tensor(rendering_resolution))
        self.num_importance = num_importance
        self.random_pose = random_pose
        self.eval_rendering_options = {
            'avg_camera_radius': avg_camera_radius,
            'avg_camera_pivot': avg_camera_pivot
        }
        self.use_sdf = use_sdf
        self.return_eikonal = (use_sdf and return_eikonal)

    def mapping(self,
                z,
                label,
                truncation_psi=1,
                truncation_cutoff=None,
                update_emas=False):
        if self.label_gen_conditioning_zero:
            label = torch.zeros_like(label)
        label = label * self.label_scale
        results = self.mapping_network(z,
                                       label,
                                       truncation_psi=truncation_psi,
                                       truncation_cutoff=truncation_cutoff,
                                       update_emas=update_emas)
        return results

    def gen_ref_representation(self, wp, update_emas=False, **synthesis_kwargs):
        if self.ref_mode == 'coordinate':
            ref_representation = None
        elif self.ref_mode == 'triplane':
            triplane = self.triplane_generator.synthesis(
                wp[:, :self.triplane_generator.synthesis.num_ws],
                update_emas=update_emas,
                **synthesis_kwargs)
            triplane = triplane.view(len(triplane), 3, -1, triplane.shape[-2],
                                     triplane.shape[-1])
            ref_representation = triplane
        elif self.ref_mode == 'volume':
            feature_volume = self.fv_generator(wp)
            ref_representation = feature_volume
        elif self.ref_mode == 'hybrid':
            triplane = self.triplane_generator.synthesis(
                wp[:, :self.triplane_generator.synthesis.num_ws],
                update_emas=update_emas,
                **synthesis_kwargs)
            triplane = triplane.view(len(triplane), 3, -1, triplane.shape[-2],
                                     triplane.shape[-1])
            feature_volume = self.fv_generator(wp) # [N, C, D, H, W]
            ref_representation = (triplane, feature_volume)
        else:
            raise TypeError(f'Unsupported reference mode: {self.ref_mode}!')

        return ref_representation

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

        ref_representation = self.gen_ref_representation(
            wp, update_emas, **synthesis_kwargs)

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

        ray_dirs_expand = ray_dirs.unsqueeze(2).expand(
            -1, -1, K, -1)  # [N, R, K, 3]
        ray_dirs_expand = ray_dirs_expand.reshape(N, -1, 3)  # [N, R * K, 3]

        # Get point features.
        point_features = self.point_representer(
            points, ref_representation=ref_representation)  # [N, R * K, C]
        if self.use_positional_encoding:
            point_encodings = self.position_encoder(points)
            if self.ref_mode == 'coordinate':
                point_features = point_encodings
            else:
                point_features = torch.cat([point_encodings, point_features],
                                           dim=-1)
        if self.mlp_type == 'stylenerf':
            point_features = point_features.reshape(N, R, K, -1)

        color_density_result = self.mlp(point_features,
                                        latents=wp,
                                        ray_dirs=ray_dirs_expand)

        if self.use_sdf:
            sdf_coarse = color_density_result['density']  # [N, R * K, 1]
            densities_coarse = torch.sigmoid(
                -sdf_coarse / self.sigmoid_beta) / self.sigmoid_beta
        else:
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

            if self.return_eikonal:
                points.requires_grad = True

            # Get point features.
            point_features = self.point_representer(
                points, ref_representation=ref_representation)  # [N, R * K, C]
            if self.use_positional_encoding:
                point_encodings = self.position_encoder(points)
                if self.ref_mode == 'coordinate':
                    point_features = point_encodings
                else:
                    point_features = torch.cat(
                        [point_encodings, point_features], dim=-1)
            if self.mlp_type == 'stylenerf':
                point_features = point_features.reshape(N, R, K, -1)

            color_density_result = self.mlp(point_features,
                                            latents=wp,
                                            ray_dirs=ray_dirs_expand)

            if self.use_sdf:
                sdf_fine = color_density_result['density']  # [N, R * K, 1]
                densities_fine = torch.sigmoid(
                    -sdf_coarse / self.sigmoid_beta) / self.sigmoid_beta
                eikonal_term = None
                if self.return_eikonal and self.training:
                    eikonal_term = autograd.grad(
                        outputs=sdf_fine,
                        inputs=points,
                        grad_outputs=torch.ones_like(sdf_fine),
                        create_graph=True)[0]
            else:
                densities_fine = color_density_result[
                    'density']  # [N, R * K, 1]

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
        if self.use_upsampler:
            sr_kwargs = {
                k: synthesis_kwargs[k]
                for k in synthesis_kwargs.keys() if k != 'noise_mode'
            }
            sr_image = self.post_cnn(rgb_image,
                                    feature_image,
                                    wp,
                                    noise_mode=self.sr_noise_mode,
                                    **sr_kwargs)
        else:
            sr_image = rgb_image

        results = {
            'image': sr_image,
            'image_raw': rgb_image,
            'image_depth': image_depth,
            'image_radii': image_radii
        }

        if self.random_pose:
            results['label'] = torch.Tensor(wp.shape[0],
                                            self.label_dim).to(wp.device)
        if self.use_sdf:
            results['sdf'] = sdf_fine
            results['eikonal_term'] = eikonal_term

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

        ref_representation = self.gen_ref_representation(
            wp, update_emas, **synthesis_kwargs)

        # Get point features.
        point_features = self.point_representer(
            coordinates, ref_representation=ref_representation)  # [N, R * K, C]
        if self.use_positional_encoding:
            point_encodings = self.position_encoder(coordinates)
            if self.ref_mode == 'coordinate':
                point_features = point_encodings
            else:
                point_features = torch.cat([point_encodings, point_features],
                                           dim=-1)
        if self.mlp_type == 'stylenerf' and point_features.ndim == 3:
            point_features = point_features.unsqueeze(2)  # [N, M, 1, 3]

        color_density_result = self.mlp(point_features, latents=wp)

        if self.use_sdf:
            sdf = color_density_result['density']  # [N, R * K, 1]
            density = torch.sigmoid(
                -sdf / self.sigmoid_beta) / self.sigmoid_beta
            color_density_result['density'] = density

        return color_density_result

    def sample_mixed(self,
                     coordinates,
                     wp,
                     truncation_psi=1,
                     truncation_cutoff=None,
                     update_emas=False,
                     **synthesis_kwargs):
        ref_representation = self.gen_ref_representation(
            wp, update_emas, **synthesis_kwargs)

        # Get point features.
        point_features = self.point_representer(
            coordinates, ref_representation=ref_representation)  # [N, R * K, C]
        if self.use_positional_encoding:
            point_encodings = self.position_encoder(coordinates)
            if self.ref_mode == 'coordinate':
                point_features = point_encodings
            else:
                point_features = torch.cat([point_encodings, point_features],
                                           dim=-1)
        if self.mlp_type == 'stylenerf' and point_features.ndim == 3:
            point_features = point_features.unsqueeze(2)  # [N, M, 1, 3]

        color_density_result = self.mlp(point_features, latents=wp)

        return color_density_result


class EG3DMLPNetwork(nn.Module):
    """Defines fully-connected layer head in EG3D."""

    def __init__(self, input_dim, hidden_dim, output_dim, depth=2, lr_mul=1.0):
        super().__init__()

        assert depth >= 2
        if depth == 2:
            self.net = nn.Sequential(
                FullyConnectedLayer(input_dim,
                                    hidden_dim,
                                    lr_multiplier=lr_mul),
                nn.Softplus(),
                FullyConnectedLayer(hidden_dim,
                                    1 + output_dim,
                                    lr_multiplier=lr_mul))
        else:
            net = []
            net.append(
                FullyConnectedLayer(input_dim,
                                    hidden_dim,
                                    lr_multiplier=lr_mul))
            net.append(nn.Softplus())
            for _ in range(depth - 2):
                net.append(
                    FullyConnectedLayer(hidden_dim,
                                        hidden_dim,
                                        lr_multiplier=lr_mul))
                net.append(nn.Softplus())
            net.append(
                FullyConnectedLayer(hidden_dim,
                                    1 + output_dim,
                                    lr_multiplier=lr_mul))
            self.net = nn.Sequential(*net)

    def forward(self, point_features, latents=None, ray_dirs=None):
        N, M, C = point_features.shape
        point_features = point_features.reshape(N * M, C).contiguous()

        y = self.net(point_features)
        y = y.reshape(N, M, -1).contiguous()

        color = torch.sigmoid(y[..., 1:]) * (1 + 2 * 0.001) - 0.001
        density = y[..., 0:1]

        return {'color': color, 'density': density}


class PiGANMLPNetwork(nn.Module):
    """Defines MLP Network in Pi-GAN."""
    def __init__(self,
                 w_dim,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 depth):
        super().__init__()

        self.input_dim = input_dim
        self.w_dim = w_dim
        self.output_dim = output_dim

        network = []
        for i in range(depth - 1):
            input_dim = input_dim if i == 0 else hidden_dim
            film = FiLMLayer(input_dim, hidden_dim, w_dim)
            network.append(film)
        film = FiLMLayer(hidden_dim, output_dim, w_dim)
        network.append(film)
        self.mlp_network = nn.Sequential(*network)

        self.density_head = nn.Linear(output_dim, 1)
        self.color_film = FiLMLayer(output_dim + 3, output_dim, w_dim)
        self.color_head = nn.Linear(output_dim, output_dim)

    def init_weights(self,):
        self.density_head.apply(freq_init(25))
        self.color_head.apply(freq_init(25))

        for module in self.modules():
            if isinstance(module, FiLMLayer):
                module.init_weights()

        self.mlp_network[0].init_weights(first=True)

    def forward(self,
                points,
                latents=None,
                ray_dirs=None):
        assert points.ndim == 3

        x = points

        for idx, layer in enumerate(self.mlp_network):
            x = layer(x, latents[:, idx])

        density = self.density_head(x)

        if ray_dirs is None:
            ray_dirs = torch.zeros_like(x[..., :3])
        x = torch.cat([x, ray_dirs], dim=-1)
        color = self.color_film(x, latents[:, len(self.mlp_network)])
        color = self.color_head(color).sigmoid()

        results = {'density': density, 'color': color}

        return results


class StyleNeRFMLPNetwork(nn.Module):
    """Defines class of MLP Network in StyleNeRF.

    Basically, this module consists of several `Style2Layer`s where convolutions
    with 1x1 kernel are involved. Note that this module is not strictly
    equivalent to MLP. Since 1x1 convolution is equal to fully-connected layer,
    we name this module `MLPNetwork`. Besides, our `MLPNetwork` takes in
    sampled points, view directions, latent codes as input, and outputs features
    for the following computation of `density` and `color` / `feature`.
    """

    def __init__(
        self,
        w_dim=512,
        input_dim=3,
        output_dim=256,
        hidden_dim=128,
        depth=8,
        magnitude_ema_beta=-1,
        activation='lrelu',
        use_skip=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.w_dim = w_dim
        self.activation = activation
        self.depth = depth
        self.magnitude_ema_beta = magnitude_ema_beta
        self.use_skip = use_skip

        self.fc_in = Style2Layer(self.input_dim,
                                 self.hidden_dim,
                                 w_dim,
                                 activation=self.activation)

        self.skip_layer = self.depth // 2 - 1 if self.use_skip else None
        if self.depth > 1:
            self.blocks = nn.ModuleList([
                Style2Layer(self.hidden_dim if i != self.skip_layer else
                            self.hidden_dim + self.input_dim,
                            self.hidden_dim,
                            w_dim,
                            activation=self.activation,
                            magnitude_ema_beta=self.magnitude_ema_beta)
                for i in range(self.depth - 1)
            ])

        self.density_head = ToRGBLayer(self.hidden_dim,
                                       1,
                                       w_dim,
                                       kernel_size=1)
        self.color_head = ToRGBLayer(self.hidden_dim,
                                     output_dim,
                                     w_dim,
                                     kernel_size=1)

    def forward(self,
                points,
                latents=None,
                ray_dirs=None):

        _, R, K, _ = points.shape
        input_p = points
        wp = latents

        height, width = dividable(R)

        assert input_p.shape[1] == height * width
        input_p = rearrange(input_p,
                            'N (H W) K C -> (N K) C H W',
                            H=height,
                            W=width)

        if height == width == 1:
            input_p = input_p.squeeze(-1).squeeze(-1)

        out = self.fc_in(input_p, wp[:, 0] if wp is not None else None)
        if self.depth > 1:
            for idx, layer in enumerate(self.blocks):
                wp_i = wp[:, idx + 1] if wp is not None else None
                if (self.skip_layer is not None) and (idx == self.skip_layer):
                    out = torch.cat([out, input_p], 1)
                out = layer(out, wp_i, up=1)

        w_idx = self.depth
        wp_i = wp[:, w_idx] if wp is not None else None
        density = self.density_head(out, wp_i)
        wp_i = wp[:, -1] if wp is not None else None
        color = self.color_head(out, wp_i)

        density = rearrange(density, '(N K) C H W -> N (H W K) C', K=K)
        color = rearrange(color, '(N K) C H W -> N (H W K) C', K=K)

        results = {'density': density, 'color': color}

        return results
