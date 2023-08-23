# python3.8
"""Contains the implementation of generator described in VolumeGAN.

Paper: https://arxiv.org/pdf/2112.10759.pdf

Official PyTorch implementation: https://github.com/genforce/volumegan
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stylegan2_generator import MappingNetwork
from models.stylegan2_generator import ModulateConvLayer
from models.stylegan2_generator import ConvLayer
from models.stylegan2_generator import DenseLayer
from third_party.stylegan2_official_ops import upfirdn2d

from models.rendering import PointSampler
from models.rendering import PointRepresenter
from models.rendering import PointIntegrator
from models.rendering.utils import PositionEncoder

from einops import rearrange
from models.utils.ops import all_gather
from models.rendering.utils import sample_importance
from models.rendering.utils import unify_attributes


class VolumeGANGenerator(nn.Module):
    """Defines the generator network in VoumeGAN."""

    def __init__(
            self,
            # Settings for mapping network.
            z_dim=512,
            w_dim=512,
            repeat_w=True,
            normalize_z=True,
            mapping_layers=8,
            mapping_fmaps=512,
            mapping_use_wscale=True,
            mapping_wscale_gain=1.0,
            mapping_lr_mul=0.01,
            # Settings for conditional generation.
            label_dim=0,
            embedding_dim=512,
            embedding_bias=True,
            embedding_use_wscale=True,
            embedding_wscale_gain=1.0,
            embedding_lr_mul=1.0,
            normalize_embedding=True,
            normalize_embedding_latent=False,
            # Settings for post CNN.
            image_channels=3,
            final_tanh=False,
            demodulate=True,
            use_wscale=True,
            wscale_gain=1.0,
            lr_mul=1.0,
            noise_type='spatial',
            fmaps_base=32 << 10,
            fmaps_max=512,
            filter_kernel=(1, 3, 3, 1),
            conv_clamp=None,
            eps=1e-8,
            rgb_init_res_out=True,
            # Settings for feature volume.
            fv_feat_res=32,
            fv_init_res=4,
            fv_base_channels=256,
            fv_output_channels=32,
            fv_w_dim=512,
            # Settings for position encoder.
            pe_input_dim=3,
            pe_num_freqs=10,
            # Settings for MLP network.
            mlp_num_layers=4,
            mlp_hidden_dim=256,
            mlp_activation_type='lrelu',
            mlp_out_dim=512,
            # Settings for rendering.
            nerf_res=32,
            resolution=64,
            num_importance=12,
            point_sampling_kwargs=None,
            ray_marching_kwargs=None):

        super().__init__()

        self.z_dim = z_dim
        self.w_dim = w_dim
        self.repeat_w = repeat_w
        self.normalize_z = normalize_z
        self.mapping_layers = mapping_layers
        self.mapping_fmaps = mapping_fmaps
        self.mapping_use_wscale = mapping_use_wscale
        self.mapping_wscale_gain = mapping_wscale_gain
        self.mapping_lr_mul = mapping_lr_mul

        self.latent_dim = (z_dim,)
        self.label_size = label_dim
        self.label_dim = label_dim
        self.embedding_dim = embedding_dim
        self.embedding_bias = embedding_bias
        self.embedding_use_wscale = embedding_use_wscale
        self.embedding_wscale_gain = embedding_wscale_gain
        self.embedding_lr_mul = embedding_lr_mul
        self.normalize_embedding = normalize_embedding
        self.normalize_embedding_latent = normalize_embedding_latent

        self.resolution = resolution
        self.nerf_res = nerf_res
        self.image_channels = image_channels
        self.final_tanh = final_tanh
        self.demodulate = demodulate
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.noise_type = noise_type.lower()
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max
        self.filter_kernel = filter_kernel
        self.conv_clamp = conv_clamp
        self.eps = eps

        self.num_nerf_layers = mlp_num_layers
        self.num_cnn_layers = int(np.log2(resolution // nerf_res * 2)) * 2
        self.num_layers = self.num_nerf_layers + self.num_cnn_layers

        # Set up `w_avg` for truncation trick.
        if self.repeat_w:
            self.register_buffer('w_avg', torch.zeros(w_dim))
        else:
            self.register_buffer('w_avg', torch.zeros(self.num_layers * w_dim))

        # Set up the mapping network.
        self.mapping = MappingNetwork(
            input_dim=z_dim,
            output_dim=w_dim,
            num_outputs=self.num_layers,
            repeat_output=repeat_w,
            normalize_input=normalize_z,
            num_layers=mapping_layers,
            hidden_dim=mapping_fmaps,
            use_wscale=mapping_use_wscale,
            wscale_gain=mapping_wscale_gain,
            lr_mul=mapping_lr_mul,
            label_dim=label_dim,
            embedding_dim=embedding_dim,
            embedding_bias=embedding_bias,
            embedding_use_wscale=embedding_use_wscale,
            embedding_wscale_gain=embedding_wscale_gain,
            embedding_lr_mul=embedding_lr_mul,
            normalize_embedding=normalize_embedding,
            normalize_embedding_latent=normalize_embedding_latent,
            eps=eps)

        # Set up the rendering related module.
        if point_sampling_kwargs is None:
            point_sampling_kwargs = {}
        if ray_marching_kwargs is None:
            ray_marching_kwargs = {}
        self.point_sampler = PointSampler(**point_sampling_kwargs)
        self.point_integrator = PointIntegrator(**ray_marching_kwargs)

        volume_bound = torch.tensor(
            [[-0.1886, -0.1671, -0.1956], [0.1887, 0.1692, 0.1872]],
            dtype=torch.float32).unsqueeze(0)
        self.point_representer = PointRepresenter(
            representation_type='volume', bound=volume_bound)

        # Set up the reference representation generator.
        self.ref_representation_generator = FeatureVolume(
            feat_res=fv_feat_res,
            init_res=fv_init_res,
            base_channels=fv_base_channels,
            output_channels=fv_output_channels,
            w_dim=fv_w_dim)

        # Set up the position encoder.
        self.position_encoder = PositionEncoder(input_dim=pe_input_dim,
                                                max_freq_log2=pe_num_freqs - 1,
                                                num_freqs=pe_num_freqs)

        # Set up the mlp.
        self.mlp = MLPNetwork(input_dim=self.position_encoder.out_dim +
                              fv_output_channels,
                              out_dim=mlp_out_dim,
                              num_layers=mlp_num_layers,
                              hidden_dim=mlp_hidden_dim,
                              activation_type=mlp_activation_type)

        # Set up the post neural renderer.
        self.post_cnn = PostNeuralRendererNetwork(
            resolution=resolution,
            init_res=nerf_res,
            w_dim=w_dim,
            image_channels=image_channels,
            final_tanh=final_tanh,
            demodulate=demodulate,
            use_wscale=use_wscale,
            wscale_gain=wscale_gain,
            lr_mul=lr_mul,
            noise_type=noise_type,
            fmaps_base=fmaps_base,
            filter_kernel=filter_kernel,
            fmaps_max=fmaps_max,
            conv_clamp=conv_clamp,
            eps=eps,
            rgb_init_res_out=rgb_init_res_out)

        # Some other rendering related arguments.
        self.resolution = resolution
        self.num_importance = num_importance

    def forward(
            self,
            z,
            label=None,
            lod=None,
            w_moving_decay=None,
            sync_w_avg=False,
            style_mixing_prob=None,
            trunc_psi=None,
            trunc_layers=None,
            noise_mode='const',
            fused_modulate=False,
            impl='cuda',
            fp16_res=None
    ):
        N = z.shape[0]

        mapping_results = self.mapping(z, label, impl=impl)
        w = mapping_results['w']
        lod = self.post_cnn.lod.item() if lod is None else lod

        if self.training and w_moving_decay is not None:
            if sync_w_avg:
                batch_w_avg = all_gather(w.detach()).mean(dim=0)
            else:
                batch_w_avg = w.detach().mean(dim=0)
            self.w_avg.copy_(batch_w_avg.lerp(self.w_avg, w_moving_decay))

        wp = mapping_results['wp']

        if self.training and style_mixing_prob is not None:
            if np.random.uniform() < style_mixing_prob:
                new_z = torch.randn_like(z)
                new_wp = self.mapping(new_z, label, impl=impl)['wp']
                current_layers = self.num_layers
                if current_layers > self.num_nerf_layers:
                    mixing_cutoff = np.random.randint(self.num_nerf_layers,
                                                      current_layers)
                    wp[:, mixing_cutoff:] = new_wp[:, mixing_cutoff:]

        if not self.training:
            trunc_psi = 1.0 if trunc_psi is None else trunc_psi
            trunc_layers = 0 if trunc_layers is None else trunc_layers
            if trunc_psi < 1.0 and trunc_layers > 0:
                w_avg = self.w_avg.reshape(1, -1, self.w_dim)[:, :trunc_layers]
                wp[:, :trunc_layers] = w_avg.lerp(
                    wp[:, :trunc_layers], trunc_psi)

        nerf_wp = wp[:, :self.num_nerf_layers]
        cnn_wp = wp[:, self.num_nerf_layers:]

        feature_volume = self.ref_representation_generator(nerf_wp)

        point_sampling_result = self.point_sampler(
            batch_size=N,
            image_size=int(self.nerf_res))

        points = point_sampling_result['points_world']  # [N, H, W, K, 3]
        ray_dirs = point_sampling_result['rays_world']  # [N, H, W, 3]
        radii_coarse = point_sampling_result['radii']  # [N, H, W, K]
        ray_origins = point_sampling_result[
            'cam2world_matrix'][:, :3, -1]  # [N, 3]

        _, H, W, K, _ = points.shape
        R = H * W
        points = points.reshape(N, -1, 3)  # [N, R * K, 3]
        ray_dirs = ray_dirs.reshape(N, R, 3)
        ray_origins = ray_origins.unsqueeze(1).repeat(1, R, 1)  # [N, R, 3]
        radii_coarse = radii_coarse.reshape(N, R, K, 1)

        point_features = self.point_representer(
            points, ref_representation=feature_volume)  # [N, R * K, C1]
        points_encoding = self.position_encoder(points)  # [N, R * K, C2]
        color_density_result = self.mlp(point_features,
                                        nerf_wp,
                                        points_encoding)

        densities_coarse = color_density_result['density']  # [N, R * K, 1]
        colors_coarse = color_density_result['color']  # [N, R * K, C3]
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
                                           smooth_weights=False)
            points = ray_origins.unsqueeze(
                2) + radii_fine * ray_dirs.unsqueeze(
                2)  # [N, R, K', 3], where K' = num_importance
            points = points.reshape(N, -1, 3)  # [N, R * K', 3]

            point_features = self.point_representer(
                points, ref_representation=feature_volume)  # [N, R * K', C1]
            points_encoding = self.position_encoder(points)  # [N, R * K', C2]
            color_density_result = self.mlp(point_features,
                                            nerf_wp,
                                            points_encoding)

            densities_fine = color_density_result['density']
            colors_fine = color_density_result['color']
            densities_fine = densities_fine.reshape(
                N, R, self.num_importance, 1)
            colors_fine = colors_fine.reshape(N, R, self.num_importance,
                                              colors_fine.shape[-1])

            # Gather coarse and fine results together.
            all_radiis, all_colors, all_densities = unify_attributes(
                radii_coarse, colors_coarse, densities_coarse, radii_fine,
                colors_fine, densities_fine)

            # Do the integration along the fine pass.
            rendering_result = self.point_integrator(all_colors,
                                                    all_densities,
                                                    all_radiis)

        else:
            # Only do the integration along the coarse pass.
            rendering_result = self.point_integrator(colors_coarse,
                                                    densities_coarse,
                                                    radii_coarse)

        feature2d = rendering_result['composite_color']         # [N, R, C]
        feature2d = feature2d.reshape(feature2d.shape[0],
                                      self.nerf_res,
                                      self.nerf_res,
                                      -1).permute(0, 3, 1, 2)

        final_results = self.post_cnn(feature2d,
                                      cnn_wp,
                                      lod=None,
                                      noise_mode=noise_mode,
                                      fused_modulate=fused_modulate,
                                      impl=impl,
                                      fp16_res=fp16_res)

        return {**mapping_results, **final_results}


class FeatureVolume(nn.Module):
    """Defines feature volume in VolumeGAN."""

    def __init__(self,
                 feat_res=32,
                 init_res=4,
                 base_channels=256,
                 output_channels=32,
                 w_dim=512,
                 **kwargs):
        super().__init__()
        self.num_stages = int(np.log2(feat_res // init_res)) + 1

        self.const = nn.Parameter(
            torch.ones(1, base_channels, init_res, init_res, init_res))
        inplanes = base_channels
        outplanes = base_channels

        self.stage_channels = []
        for i in range(self.num_stages):
            conv = nn.Conv3d(inplanes,
                             outplanes,
                             kernel_size=(3, 3, 3),
                             padding=(1, 1, 1))
            self.stage_channels.append(outplanes)
            self.add_module(f'layer{i}', conv)
            instance_norm = InstanceNormLayer(num_features=outplanes,
                                              affine=False)

            self.add_module(f'instance_norm{i}', instance_norm)
            inplanes = outplanes
            outplanes = max(outplanes // 2, output_channels)
            if i == self.num_stages - 1:
                outplanes = output_channels

        self.mapping_network = nn.Linear(w_dim, sum(self.stage_channels) * 2)
        self.mapping_network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.mapping_network.weight *= 0.25
        self.upsample = UpsamplingLayer()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, w):
        if w.ndim == 3:
            _w = w[:, 0]
        else:
            _w = w
        scale_shifts = self.mapping_network(_w)
        scales = scale_shifts[..., :scale_shifts.shape[-1] // 2]
        shifts = scale_shifts[..., scale_shifts.shape[-1] // 2:]

        x = self.const.repeat(w.shape[0], 1, 1, 1, 1)
        for idx in range(self.num_stages):
            if idx != 0:
                x = self.upsample(x)
            conv_layer = self.__getattr__(f'layer{idx}')
            x = conv_layer(x)
            instance_norm = self.__getattr__(f'instance_norm{idx}')
            scale = scales[:,
                           sum(self.stage_channels[:idx]
                               ):sum(self.stage_channels[:idx + 1])]
            shift = shifts[:,
                           sum(self.stage_channels[:idx]
                               ):sum(self.stage_channels[:idx + 1])]
            scale = scale.view(scale.shape + (1, 1, 1))
            shift = shift.view(shift.shape + (1, 1, 1))
            x = instance_norm(x, weight=scale, bias=shift)
            x = self.lrelu(x)

        return x


class MLPNetwork(nn.Module):
    """Defines class of MLP Network described in VolumeGAN.

    Basically, this class takes in latent codes and point coodinates as input,
    and outputs features of each point, which is followed by two fully-connected
    layer heads.
    """

    def __init__(self,
                 input_dim,
                 out_dim=512,
                 num_layers=4,
                 hidden_dim=256,
                 activation_type='lrelu'):
        super().__init__()
        self.mlp = self.build_mlp(input_dim=input_dim,
                                     num_layers=num_layers,
                                     hidden_dim=hidden_dim,
                                     activation_type=activation_type)
        self.density_head = DenseLayer(in_channels=hidden_dim,
                                        out_channels=1,
                                        add_bias=True,
                                        init_bias=0.0,
                                        use_wscale=True,
                                        wscale_gain=1,
                                        lr_mul=1,
                                        activation_type='linear')
        self.color_head = DenseLayer(in_channels=hidden_dim,
                                      out_channels=out_dim,
                                      add_bias=True,
                                      init_bias=0.0,
                                      use_wscale=True,
                                      wscale_gain=1,
                                      lr_mul=1,
                                      activation_type='linear')

    def build_mlp(self, input_dim, num_layers, hidden_dim, activation_type):
        """Implements function to build the `MLP`.

        Note that here the `MLP` network is consists of a series of
        `ModulateConvLayer` with `kernel_size=1` to simulate fully-connected
        layer. Typically, the input's shape of convolutional layers is
        `[N, C, H, W]`. And the input's shape is `[N, C, R*K, 1]` here, which
        aims to keep consistent with `MLP`.
        """
        default_conv_cfg = dict(resolution=32,
                                w_dim=512,
                                kernel_size=1,
                                add_bias=True,
                                scale_factor=1,
                                filter_kernel=None,
                                demodulate=True,
                                use_wscale=True,
                                wscale_gain=1,
                                lr_mul=1,
                                noise_type='none',
                                conv_clamp=None,
                                eps=1e-8)
        mlp_list = nn.ModuleList()
        in_ch = input_dim
        out_ch = hidden_dim
        for _ in range(num_layers):
            mlp = ModulateConvLayer(in_channels=in_ch,
                                    out_channels=out_ch,
                                    activation_type=activation_type,
                                    **default_conv_cfg)
            mlp_list.append(mlp)
            in_ch = out_ch
            out_ch = hidden_dim

        return mlp_list

    def forward(self,
                point_features,
                wp,
                points_encoding=None,
                fused_modulate=False,
                impl='cuda'):
        point_features = point_features.permute(0, 2, 1)  # [N, C, R * K]
        point_features = point_features.unsqueeze(-1)  # [N, C, R * K, 1]
        points_encoding = rearrange(points_encoding,
                                    'N R_K C -> N C R_K 1').contiguous()
        x = torch.cat([point_features, points_encoding], dim=1)

        for idx, mlp in enumerate(self.mlp):
            if wp.ndim == 3:
                _w = wp[:, idx]
            else:
                _w = wp
            x, _ = mlp(x, _w, fused_modulate=fused_modulate, impl=impl)

        post_point_features = rearrange(
            x, 'N C (R_K) 1 -> (N R_K) C').contiguous()

        density = self.density_head(post_point_features)
        color = self.color_head(post_point_features)

        results = {'density': density, 'color': color}

        return results


class PostNeuralRendererNetwork(nn.Module):
    """Implements the neural renderer in VolumeGAN to render high-resolution
    images.

    Basically, this network executes several convolutional layers in sequence.
    """

    def __init__(
        self,
        resolution,
        init_res,
        w_dim,
        image_channels,
        final_tanh,
        demodulate,
        use_wscale,
        wscale_gain,
        lr_mul,
        noise_type,
        fmaps_base,
        fmaps_max,
        filter_kernel,
        conv_clamp,
        eps,
        rgb_init_res_out=False,
    ):
        super().__init__()

        self.init_res = init_res
        self.init_res_log2 = int(np.log2(init_res))
        self.resolution = resolution
        self.final_res_log2 = int(np.log2(resolution))
        self.w_dim = w_dim
        self.image_channels = image_channels
        self.final_tanh = final_tanh
        self.demodulate = demodulate
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.noise_type = noise_type.lower()
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max
        self.filter_kernel = filter_kernel
        self.conv_clamp = conv_clamp
        self.eps = eps
        self.rgb_init_res_out = rgb_init_res_out

        self.num_layers = (self.final_res_log2 - self.init_res_log2 + 1) * 2

        self.register_buffer('lod', torch.zeros(()))

        for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
            res = 2**res_log2
            in_channels = self.get_nf(res // 2)
            out_channels = self.get_nf(res)
            block_idx = res_log2 - self.init_res_log2

            # Early layer.
            if res > init_res:
                layer_name = f'layer{2 * block_idx - 1}'
                self.add_module(
                    layer_name,
                    ModulateConvLayer(in_channels=in_channels,
                                      out_channels=out_channels,
                                      resolution=res,
                                      w_dim=w_dim,
                                      kernel_size=1,
                                      add_bias=True,
                                      scale_factor=2,
                                      filter_kernel=filter_kernel,
                                      demodulate=demodulate,
                                      use_wscale=use_wscale,
                                      wscale_gain=wscale_gain,
                                      lr_mul=lr_mul,
                                      noise_type=noise_type,
                                      activation_type='lrelu',
                                      conv_clamp=conv_clamp,
                                      eps=eps))
            if block_idx == 0:
                if self.rgb_init_res_out:
                    self.color_init_res = ConvLayer(
                        in_channels=out_channels,
                        out_channels=image_channels,
                        kernel_size=1,
                        add_bias=True,
                        scale_factor=1,
                        filter_kernel=None,
                        use_wscale=use_wscale,
                        wscale_gain=wscale_gain,
                        lr_mul=lr_mul,
                        activation_type='linear',
                        conv_clamp=conv_clamp,
                    )
                continue
            # Second layer (kernel 1x1) without upsampling.
            layer_name = f'layer{2 * block_idx}'
            self.add_module(
                layer_name,
                ModulateConvLayer(in_channels=out_channels,
                                  out_channels=out_channels,
                                  resolution=res,
                                  w_dim=w_dim,
                                  kernel_size=1,
                                  add_bias=True,
                                  scale_factor=1,
                                  filter_kernel=None,
                                  demodulate=demodulate,
                                  use_wscale=use_wscale,
                                  wscale_gain=wscale_gain,
                                  lr_mul=lr_mul,
                                  noise_type=noise_type,
                                  activation_type='lrelu',
                                  conv_clamp=conv_clamp,
                                  eps=eps))

            # Output convolution layer for each resolution (if needed).
            layer_name = f'output{block_idx}'
            self.add_module(
                layer_name,
                ModulateConvLayer(in_channels=out_channels,
                                  out_channels=image_channels,
                                  resolution=res,
                                  w_dim=w_dim,
                                  kernel_size=1,
                                  add_bias=True,
                                  scale_factor=1,
                                  filter_kernel=None,
                                  demodulate=False,
                                  use_wscale=use_wscale,
                                  wscale_gain=wscale_gain,
                                  lr_mul=lr_mul,
                                  noise_type='none',
                                  activation_type='linear',
                                  conv_clamp=conv_clamp,
                                  eps=eps))

        # Used for upsampling output images for each resolution block for sum.
        self.register_buffer('filter', upfirdn2d.setup_filter(filter_kernel))

    def get_nf(self, res):
        """Gets number of feature maps according to current resolution."""
        return min(self.fmaps_base // res, self.fmaps_max)

    def set_space_of_latent(self, space_of_latent):
        """Sets the space to which the latent code belong.

        Args:
            space_of_latent: The space to which the latent code belong. Case
                insensitive. Support `W` and `Y`.
        """
        space_of_latent = space_of_latent.upper()
        for module in self.modules():
            if isinstance(module, ModulateConvLayer):
                setattr(module, 'space_of_latent', space_of_latent)

    def forward(self,
                x,
                wp,
                lod=None,
                noise_mode='const',
                fused_modulate=False,
                impl='cuda',
                fp16_res=None,
                nerf_out=False):
        lod = self.lod.item() if lod is None else lod

        results = {}

        # Cast to `torch.float16` if needed.
        if fp16_res is not None and self.init_res >= fp16_res:
            x = x.to(torch.float16)

        for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
            cur_lod = self.final_res_log2 - res_log2
            block_idx = res_log2 - self.init_res_log2

            layer_idxs = [2 * block_idx - 1, 2 *
                          block_idx] if block_idx > 0 else [
                              2 * block_idx,
                          ]
            # determine forward until cur resolution
            if lod < cur_lod + 1:
                for layer_idx in layer_idxs:
                    if layer_idx == 0:
                        # image = x[:,:3]
                        if self.rgb_init_res_out:
                            cur_image = self.color_init_res(x,
                                                          runtime_gain=1,
                                                          impl=impl)
                        else:
                            cur_image = x[:, :3]
                        continue
                    layer = getattr(self, f'layer{layer_idx}')
                    x, style = layer(
                        x,
                        wp[:, layer_idx],
                        noise_mode=noise_mode,
                        fused_modulate=fused_modulate,
                        impl=impl,
                    )
                    results[f'style{layer_idx}'] = style
                    if layer_idx % 2 == 0:
                        output_layer = getattr(self, f'output{layer_idx // 2}')
                        y, style = output_layer(
                            x,
                            wp[:, layer_idx + 1],
                            fused_modulate=fused_modulate,
                            impl=impl,
                        )
                        results[f'output_style{layer_idx // 2}'] = style
                        if layer_idx == 0:
                            cur_image = y.to(torch.float32)
                        else:
                            if not nerf_out:
                                cur_image = y.to(
                                    torch.float32) + upfirdn2d.upsample2d(
                                        cur_image, self.filter, impl=impl)
                            else:
                                cur_image = y.to(torch.float32) + cur_image

                        # Cast to `torch.float16` if needed.
                        if layer_idx != self.num_layers - 2:
                            res = self.init_res * (2**(layer_idx // 2))
                            if fp16_res is not None and res * 2 >= fp16_res:
                                x = x.to(torch.float16)
                            else:
                                x = x.to(torch.float32)

            # color interpolation
            if cur_lod - 1 < lod <= cur_lod:
                image = cur_image
            elif cur_lod < lod < cur_lod + 1:
                alpha = np.ceil(lod) - lod
                image = F.interpolate(image, scale_factor=2, mode='nearest')
                image = cur_image * alpha + image * (1 - alpha)
            elif lod >= cur_lod + 1:
                image = F.interpolate(image, scale_factor=2, mode='nearest')

        if self.final_tanh:
            image = torch.tanh(image)
        results['image'] = image

        return results


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight,
                                      a=0.2,
                                      mode='fan_in',
                                      nonlinearity='leaky_relu')


class InstanceNormLayer(nn.Module):
    """Implements instance normalization layer."""

    def __init__(self, num_features, epsilon=1e-8, affine=False):
        super().__init__()
        self.eps = epsilon
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1, 1))
            self.bias = nn.Parameter(torch.Tensor(1, num_features, 1, 1, 1))
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def forward(self, x, weight=None, bias=None):
        x = x - torch.mean(x, dim=[2, 3, 4], keepdim=True)
        norm = torch.sqrt(
            torch.mean(x**2, dim=[2, 3, 4], keepdim=True) + self.eps)
        x = x / norm
        isnot_input_none = weight is not None and bias is not None
        assert (isnot_input_none and not self.affine) or (not isnot_input_none
                                                          and self.affine)
        if self.affine:
            x = x * self.weight + self.bias
        else:
            x = x * weight + bias
        return x


class UpsamplingLayer(nn.Module):

    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        if self.scale_factor <= 1:
            return x
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')