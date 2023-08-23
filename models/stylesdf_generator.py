# python3.8
"""Contains the implementation of generator described in StyleSDF.

Paper: https://arxiv.org/pdf/2112.11427.pdf

Official PyTorch implementation: https://github.com/royorel/StyleSDF
"""

import math
import numpy as np
import random
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torch.autograd as autograd
import torch.distributed as dist

from models.rendering import PointSampler
from models.rendering import PointRepresenter
from models.rendering import PointIntegrator

from .utils.stylesdf_model_helper import MappingLinear
from .utils.stylesdf_model_helper import EqualLinear_sdf
from .utils.stylesdf_model_helper import StyConv
from .utils.stylesdf_model_helper import ToRGB
from .utils.stylesdf_model_helper import PixelNorm


class StyleSDFGenerator(nn.Module):
    """Defines the generator network in StyleSDF."""

    def __init__(
            self,
            # Settings for mapping network.
            z_dim=256,
            w_dim=256,
            normalize_z=False,
            mapping_layers=3,
            # Settings for conditional generation.
            label_dim=0,
            embedding_dim=512,
            normalize_embedding=True,
            normalize_embedding_latent=False,
            # Settings for synthesis network.
            synthesis_input_dim=3,
            synthesis_output_dim=256,
            synthesis_layers=8,
            grid_scale=0.24,
            eps=1e-8,
            # Settings for rendering related module.
            image_resolution=64,
            render_resolution=64,
            num_importance=12,
            freeze_renderer=True,
            full_pipeline=True,
            smooth_weights=False,
            point_sampling_kwargs=None,
            ray_marching_kwargs=None,
            sphere_init_path=None):

        """Initializes with basic settings."""
        super().__init__()

        self.image_channels=3
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.normalize_z = normalize_z
        self.mapping_layers = mapping_layers

        self.latent_dim = (z_dim,)
        self.label_dim = label_dim
        self.embedding_dim = embedding_dim
        self.normalize_embedding = normalize_embedding
        self.normalize_embedding_latent = normalize_embedding_latent

        self.num_layers = synthesis_layers
        self.eps = eps

        self.image_resolution = image_resolution
        self.render_resolution = render_resolution

        self.train_renderer = not freeze_renderer
        self.full_pipeline = full_pipeline

        self.sigmoid_beta = nn.Parameter(0.1 * torch.ones(1).float())

        # Mapping Network to tranform latent codes from Z-Space into W-Space.
        layers = []
        for _ in range(mapping_layers):
            layers.append(
                MappingLinear(self.z_dim, self.z_dim, activation="fused_lrelu")
            )
        self.render_mapping = nn.Sequential(*layers)

        # Set up the rendering related module.
        if point_sampling_kwargs is None:
            point_sampling_kwargs = {}
        if ray_marching_kwargs is None:
            ray_marching_kwargs = {}
        self.point_sampler = PointSampler(**point_sampling_kwargs)
        self.point_integrator = PointIntegrator(**ray_marching_kwargs)

        self.point_representer = PointRepresenter(
            representation_type='coordinate')

        # Set up the MLP network.
        self.mlp = SirenGenerator(resolution=self.render_resolution,
                                  W=self.w_dim,
                                  z_dim=self.z_dim,
                                  num_layers=self.num_layers,
                                  input_ch=self.image_channels,
                                  input_ch_views=self.image_channels,
                                  output_features=self.full_pipeline)
        if self.full_pipeline:
            self.post_decoder = PostNeuralRendererNetwork(
                image_resolution=self.image_resolution,
                style_dim=self.w_dim,
                lr_mapping=0.01,
                render_resolution=self.render_resolution,
                channel_multiplier=2,
                feature_encoder_in_channels=256,
                project_noise=False,
                blur_kernel=[1, 3, 3, 1])


        # Some other rendering related arguments.
        self.num_importance = num_importance
        self.smooth_weights = smooth_weights

        # Initialize weights.
        self.init_weights(sphere_init_path=sphere_init_path)

    def init_weights(self, sphere_init_path):
        print(f'Loading the sphere init from: {sphere_init_path}')
        if os.path.exists(sphere_init_path):
            # Download pre-trained weights.
            if dist.is_initialized() and dist.get_rank() != 0:
                dist.barrier()  # Download by chief.
            state_dict = torch.load(sphere_init_path, map_location='cpu')
            if dist.is_initialized() and dist.get_rank() == 0:
                dist.barrier()  # Wait for other replicas.
            render_mapping_state_dict = OrderedDict()
            mlp_state_dict = OrderedDict()
            sigmoid_beta_state_dict = OrderedDict()
            for name, v in state_dict['g'].items():
                if 'style' in name:
                    name = name.replace("style.", "")
                    render_mapping_state_dict[name] = v
                elif 'renderer.network' in name:
                    name = name.replace("renderer.network.", "")
                    mlp_state_dict[name] = v
                else:
                    sigmoid_beta_state_dict= v
            self.render_mapping.load_state_dict(render_mapping_state_dict,
                                                strict=True)
            self.mlp.load_state_dict(mlp_state_dict,
                                     strict=False)
            self.sigmoid_beta=nn.Parameter(sigmoid_beta_state_dict)
            del state_dict,\
                render_mapping_state_dict,\
                mlp_state_dict,\
                sigmoid_beta_state_dict
        else:
            for module in self.render_mapping.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight,
                                            a=0.2,
                                            mode='fan_in',
                                            nonlinearity='leaky_relu')

    def forward(self,
                z,
                label=None,
                style_mixing_prob=None,
                enable_amp=False,
                return_eikonal=False,
                return_latents=False,
                noise_std=None):

        with torch.set_grad_enabled(self.train_renderer):
            with autocast(enabled=enable_amp):
                N = z.shape[0]
                w = self.render_mapping(z)
                if self.full_pipeline:
                    # Truncation.
                    wp = [w]
                    if self.training and style_mixing_prob is not None:
                        if np.random.uniform() < style_mixing_prob:
                            new_z = torch.randn_like(z)
                            new_w = self.render_mapping(new_z)
                            wp.append(new_w)

                point_sampling_result = self.point_sampler(
                    batch_size=N,
                    image_size=int(self.render_resolution))

                points = point_sampling_result[
                    'points_world']  # [N, H, W, K, 3]
                ray_dirs = point_sampling_result['rays_world']  # [N, H, W, 3]
                radii_coarse = point_sampling_result['radii']  # [N, H, W, K]


                camera_polar = point_sampling_result['camera_polar'].unsqueeze(
                    -1)  # [N, 1]
                camera_azimuthal = point_sampling_result[
                    'camera_azimuthal'].unsqueeze(-1)  # [N, 1]

                _, H, W, K, _ = points.shape
                R = H * W
                points = points.reshape(N, -1, 3)  # [N, R * K, 3]
                ray_dirs = ray_dirs.reshape(N, R, 3)

                radii_coarse = radii_coarse.reshape(N, R, K, 1)

                ray_dirs_expand = ray_dirs.unsqueeze(2).expand(
                    -1, -1, K, -1)  # [N, R, K, 3]
                ray_dirs_expand = ray_dirs_expand.reshape(N, H, W, K,
                                                          3)  # [N, R * K, 3]

                points = points.reshape(N, H, W, K, 3)
                if return_eikonal: points.requires_grad = True

                color_sdf_result = self.mlp(
                    points,  # [N, R * K, 3]
                    latents=w,  # [N, 256]
                    ray_dirs=ray_dirs_expand)  # [N, R * K, 3]

                # sdf_coarse = color_sdf_result['sdf']  # [N, R * K, 1]
                # colors_coarse = color_sdf_result['color']  # [N, R * K, C]
                if not self.full_pipeline:
                    colors_coarse, sdf_coarse = torch.split(color_sdf_result,
                                                            [3, 1],
                                                            dim=-1)
                else:
                    colors_coarse, sdf_coarse, features_lowres = torch.split(
                        color_sdf_result, [3, 1, 256], dim=-1)
                    features_lowres = features_lowres.reshape(
                        z.shape[0], self.render_resolution,
                        self.render_resolution, K, -1)

                if return_eikonal:
                    eikonal_term = self.get_eikonal_term(points, sdf_coarse)
                else:
                    eikonal_term = None

                colors_coarse = colors_coarse.reshape(N, R, K,
                                                      colors_coarse.shape[-1])

                densities_coarse = torch.sigmoid(
                    -sdf_coarse / self.sigmoid_beta) / self.sigmoid_beta

                densities_coarse = densities_coarse.reshape(N, R, K, 1)

                camera = torch.cat([camera_polar, camera_azimuthal], -1)
                rendering_result = self.point_integrator(
                    colors_coarse, densities_coarse, radii_coarse)
                features_lowres_weight = rendering_result['weight'].reshape(
                    z.shape[0], self.render_resolution, self.render_resolution,
                    K, -1)
                image = rendering_result['composite_color'].reshape(
                    z.shape[0], self.render_resolution, self.render_resolution,
                    -1)  # [N, H, W, 3]
                image = image.permute(0, 3, 1, 2)  # [N, 3, H, W]


        if self.full_pipeline:
            features_lowres = torch.sum(features_lowres *
                                        features_lowres_weight, -2)\
                .permute(0, 3, 1, 2).contiguous()

            post_render_imgs, out_latent = self.post_decoder(features_lowres,
                                                             wp,
                                                             return_latents)
        else:
            out_latent = None
            post_render_imgs = image

        return {
            'label': label,
            'image': post_render_imgs,
            'image_low': image,
            'camera': camera,
            'latent': z,
            'points': points,
            'sdf': sdf_coarse,
            'eikonal_term': eikonal_term,
            'out_latent': out_latent
        }

    def get_eikonal_term(self, pts, sdf):
        eikonal_term = autograd.grad(outputs=sdf,
                                     inputs=pts,
                                     grad_outputs=torch.ones_like(sdf),
                                     create_graph=True)[0]

        return eikonal_term

class PostNeuralRendererNetwork(nn.Module):
    def __init__(self,
                 image_resolution=1024,
                 style_dim=256,
                 lr_mapping=0.01,
                 render_resolution=64,
                 channel_multiplier=2,
                 feature_encoder_in_channels=256,
                 project_noise=False,
                 blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        # decoder mapping network
        self.image_resolution = image_resolution
        self.style_dim = style_dim * 2

        layers = [
            PixelNorm(),
            EqualLinear_sdf(self.style_dim // 2,
                            self.style_dim,
                            lr_mul=lr_mapping,
                            activation="fused_lrelu")
        ]

        for i in range(4):
            layers.append(
                EqualLinear_sdf(self.style_dim,
                                self.style_dim,
                                lr_mul=lr_mapping,
                                activation="fused_lrelu"))

        self.sty_decoder = nn.Sequential(*layers)

        # decoder network
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        decoder_in_size = render_resolution

        # image decoder
        self.log_size = int(math.log(self.image_resolution, 2))
        self.log_in_size = int(math.log(decoder_in_size, 2))

        self.conv1 = StyConv(feature_encoder_in_channels,
                             self.channels[decoder_in_size],
                             3,
                             self.style_dim,
                             blur_kernel=blur_kernel,
                             project_noise=project_noise)

        self.to_rgb1 = ToRGB(self.channels[decoder_in_size],
                             self.style_dim,
                             upsample=False)

        self.num_layers = (self.log_size - self.log_in_size) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[decoder_in_size]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 2 * self.log_in_size + 1) // 2
            shape = [1, 1, 2 ** (res), 2 ** (res)]
            self.noises.register_buffer(f"noise_{layer_idx}",
                                        torch.randn(*shape))

        for i in range(self.log_in_size+1, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyConv(in_channel,
                        out_channel,
                        3,
                        self.style_dim,
                        upsample=True,
                        blur_kernel=blur_kernel,
                        project_noise=project_noise))

            self.convs.append(
                StyConv(out_channel,
                        out_channel,
                        3,
                        self.style_dim,
                        blur_kernel=blur_kernel,
                        project_noise=project_noise))

            self.to_rgbs.append(ToRGB(out_channel, self.style_dim))

            in_channel = out_channel

        self.n_latent = (self.log_size - self.log_in_size) * 2 + 2

    def mean_latent(self, renderer_latent):
        latent = self.sty_decoder(renderer_latent).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.sty_decoder(input)

    def styles_and_noise_forward(self,
                                 styles,
                                 noise,
                                 inject_index=None,
                                 truncation=1,
                                 truncation_latent=None,
                                 input_is_latent=False,
                                 randomize_noise=True):
        if not input_is_latent:
            styles = [self.sty_decoder(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}")
                    for i in range(self.num_layers)
                ]

        if (truncation < 1):
            style_t = []

            for style in styles:
                style_t.append(truncation_latent[1] + truncation *
                               (style - truncation_latent[1]))

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(
                1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        return latent, noise

    def forward(self,
                features,
                latent,
                return_latents=False,
                rgbd_in=None,
                transform=None,
                inject_index=None,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                noise=None,
                randomize_noise=True,
                mesh_path=None):
        latent, noise = self.styles_and_noise_forward(latent, noise,
                                                      inject_index, truncation,
                                                      truncation_latent,
                                                      input_is_latent,
                                                      randomize_noise)

        out = self.conv1(features,
                         latent[:, 0],
                         noise=noise[0],
                         transform=transform,
                         mesh_path=mesh_path)

        skip = self.to_rgb1(out, latent[:, 1], skip=rgbd_in)
        skips = [skip]
        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2],
                self.to_rgbs):
            out = conv1(out,
                        latent[:, i],
                        noise=noise1,
                        transform=transform,
                        mesh_path=mesh_path)
            out = conv2(out,
                        latent[:, i + 1],
                        noise=noise2,
                        transform=transform,
                        mesh_path=mesh_path)
            skip_now = to_rgb(out, latent[:, i + 2], skip=skips[-1])
            skips.append(skip_now)

            i += 2

        out_latent = latent if return_latents else None
        image = skips[-1]

        return image, out_latent


class SirenGenerator(nn.Module):
    def __init__(self,
                 resolution=64,
                 W=256,
                 z_dim=256,
                 input_ch=3,
                 input_ch_views=3,
                 num_layers=8,
                 output_features=True):
        super(SirenGenerator, self).__init__()
        self.resolution = resolution
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.z_dim = z_dim
        self.output_features = output_features

        self.register_buffer('lod', torch.zeros(()))
        self.pts_linears = nn.ModuleList(
            [FiLMSiren(3, W, z_dim=z_dim, is_first=True)] + \
            [FiLMSiren(W, W, z_dim=z_dim) for i in range(num_layers-1)])

        self.views_linears = FiLMSiren(input_ch_views + W,
                                       W,
                                       z_dim=z_dim)
        self.rgb_linear = LinearLayer(W, 3, freq_init=True)
        self.sigma_linear = LinearLayer(W, 1, freq_init=True)

    def forward(self, points, latents, ray_dirs):
        input_pts, input_views = points, ray_dirs
        mlp_out = input_pts.contiguous()
        for i in range(len(self.pts_linears)):
            mlp_out = self.pts_linears[i](mlp_out, latents)


        sdf = self.sigma_linear(mlp_out)

        mlp_out = torch.cat([mlp_out, input_views], -1)
        out_features = self.views_linears(mlp_out, latents)
        rgb = self.rgb_linear(out_features)

        outputs = torch.cat([rgb, sdf], -1)
        if self.output_features:
            outputs = torch.cat([outputs, out_features], -1)

        return outputs


class MappingNetwork(nn.Module):
    """Implements the latent space mapping module.

    Basically, this module executes several dense layers in sequence, and the
    label embedding if needed.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 num_outputs,
                 repeat_output,
                 normalize_input,
                 num_layers,
                 hidden_dim,
                 label_dim,
                 embedding_dim,
                 normalize_embedding,
                 normalize_embedding_latent,
                 eps,
                 label_concat,
                 lr=None):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_outputs = num_outputs
        self.repeat_output = repeat_output
        self.normalize_input = normalize_input
        self.num_layers = num_layers

        self.label_dim = label_dim
        self.embedding_dim = embedding_dim
        self.normalize_embedding = normalize_embedding
        self.normalize_embedding_latent = normalize_embedding_latent
        self.eps = eps
        self.label_concat = label_concat

        self.norm = PixelNormLayer(dim=1, eps=eps)

        if num_outputs is not None and not repeat_output:
            output_dim = output_dim * num_outputs

        if self.label_dim > 0:
            if self.label_concat:
                input_dim = input_dim + embedding_dim
                self.embedding = EqualLinear(label_dim,
                                             embedding_dim,
                                             bias=True,
                                             bias_init=0,
                                             lr_mul=1)
            else:
                self.embedding = EqualLinear(label_dim,
                                             output_dim,
                                             bias=True,
                                             bias_init=0,
                                             lr_mul=1)

        network = []
        for i in range(num_layers):
            in_channels = (input_dim if i == 0 else hidden_dim)
            out_channels = (output_dim if i == (num_layers - 1) else hidden_dim)
            network.append(nn.Linear(in_channels, out_channels))
            network.append(nn.LeakyReLU(0.2, inplace=True))
        self.network = nn.Sequential(*network)

    def init_weights(self):
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight,
                                        a=0.2,
                                        mode='fan_in',
                                        nonlinearity='leaky_relu')

    def forward(self, z, label=None):
        if z.ndim != 2 or z.shape[1] != self.input_dim:
            raise ValueError(f'Input latent code should be with shape '
                             f'[batch_size, input_dim], where '
                             f'`input_dim` equals to {self.input_dim}!\n'
                             f'But `{z.shape}` is received!')
        if self.normalize_input:
            z = self.norm(z)
        if self.label_dim > 0:
            if label is None:
                raise ValueError(f'Model requires an additional label '
                                 f'(with dimension {self.label_dim}) as input, '
                                 f'but no label is received!')
            if label.ndim != 2 or label.shape != (z.shape[0], self.label_dim):
                raise ValueError(f'Input label should be with shape '
                                 f'[batch_size, label_dim], where '
                                 f'`batch_size` equals to that of '
                                 f'latent codes ({z.shape[0]}) and '
                                 f'`label_dim` equals to {self.label_dim}!\n'
                                 f'But `{label.shape}` is received!')
            label = label.to(dtype=torch.float32)

            embedding = self.embedding(label)
            if self.normalize_embedding and self.label_concat:
                embedding = self.norm(embedding)
            if self.label_concat:
                w = torch.cat((z, embedding), dim=1)
            else:
                w = z
        else:
            w = z

        if (self.label_dim > 0 and self.normalize_embedding_latent
            and self.label_concat):
            w = self.norm(w)

        for layer in self.network:
            w = layer(w)

        if self.label_dim > 0 and (not self.label_concat):
            w = w * embedding

        wp = None
        if self.num_outputs is not None:
            if self.repeat_output:
                wp = w.unsqueeze(1).repeat((1, self.num_outputs, 1))
            else:
                wp = w.reshape(-1, self.num_outputs, self.output_dim)

        results = {
            'z': z,
            'label': label,
            'w': w,
            'wp': wp,
        }
        if self.label_dim > 0:
            results['embedding'] = embedding
        return results


class FiLMSiren(nn.Module):

    def __init__(self, in_channel, out_channel, z_dim, is_first=False):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        if is_first:
            self.weight = nn.Parameter(
                torch.empty(out_channel, in_channel).uniform_(-1 / 3, 1 / 3))
        else:
            self.weight = nn.Parameter(
                torch.empty(out_channel,
                            in_channel).uniform_(-np.sqrt(6 / in_channel) / 25,
                                                 np.sqrt(6 / in_channel) / 25))

        self.bias = nn.Parameter(
            nn.Parameter(
                nn.init.uniform_(torch.empty(out_channel),
                                 a=-np.sqrt(1 / in_channel),
                                 b=np.sqrt(1 / in_channel))))
        self.activation = torch.sin

        self.gamma = LinearLayer(z_dim, out_channel, bias_init=30, std_init=15)
        self.beta = LinearLayer(z_dim, out_channel, bias_init=0, std_init=0.25)

    def forward(self, input, style):
        batch, features = style.shape
        out = F.linear(input, self.weight, bias=self.bias)
        gamma = self.gamma(style).view(batch, 1, 1, 1, features)
        beta = self.beta(style).view(batch, 1, 1, 1, features)

        out = self.activation(gamma * out + beta)

        return out


class LinearLayer(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 bias=True,
                 bias_init=0,
                 std_init=1,
                 freq_init=False,
                 is_first=False):
        super().__init__()
        if is_first:
            self.weight = nn.Parameter(
                torch.empty(out_dim, in_dim).uniform_(-1 / in_dim, 1 / in_dim))
        elif freq_init:
            self.weight = nn.Parameter(
                torch.empty(out_dim,
                            in_dim).uniform_(-np.sqrt(6 / in_dim) / 25,
                                             np.sqrt(6 / in_dim) / 25))
        else:
            self.weight = nn.Parameter(
                0.25 * nn.init.kaiming_normal_(torch.randn(out_dim, in_dim),
                                               a=0.2,
                                               mode='fan_in',
                                               nonlinearity='leaky_relu'))

        self.bias = nn.Parameter(
            nn.init.uniform_(torch.empty(out_dim),
                             a=-np.sqrt(1 / in_dim),
                             b=np.sqrt(1 / in_dim)))

        self.bias_init = bias_init
        self.std_init = std_init

    def forward(self, input):
        out = self.std_init * F.linear(input, self.weight,
                                       bias=self.bias) + self.bias_init

        return out




class PixelNormLayer(nn.Module):
    """Implements pixel-wise feature vector normalization layer."""

    def __init__(self, dim, eps):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def extra_repr(self):
        return f'dim={self.dim}, epsilon={self.eps}'

    def forward(self, x):
        scale = (x.square().mean(dim=self.dim, keepdim=True) + self.eps).rsqrt()
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
            m.weight.uniform_(-1/num_input, 1/num_input)

def freq_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6/num_input)/freq,
                                  np.sqrt(6/num_input)/freq)
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
                f'{self.weight.shape[0]})')
