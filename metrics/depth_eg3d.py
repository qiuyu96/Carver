# python3.8
"""Contains the class to evaluate accuracy of face depth maps output by
3D-aware GANs."""

import os.path
import time
import numpy as np

import torch
import torch.nn.functional as F

from utils.misc import get_cache_dir
from models import build_model
from .base_gan_metric import BaseGANMetric
from models.rendering.point_sampler import sample_camera_extrinsics


class DepthEG3DMetric(BaseGANMetric):
    """Defines the class for evaluation of geometry described in EG3D."""

    def __init__(self,
                 name='DepthEG3D',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 latent_dim=512,
                 latent_codes=None,
                 label_dim=0,
                 labels=None,
                 seed=0,
                 fake_num=1024,
                 scale=10.0,
                 random_pose=False,
                 eg3d_mode=True):
        super().__init__(name=name,
                         work_dir=work_dir,
                         logger=logger,
                         tb_writer=tb_writer,
                         batch_size=batch_size,
                         latent_num=fake_num,
                         latent_dim=latent_dim,
                         latent_codes=latent_codes,
                         label_dim=label_dim,
                         labels=labels,
                         seed=seed)
        self.fake_num = fake_num
        self.random_pose = random_pose
        self.eg3d_mode = eg3d_mode

        self.scale = scale

        bfm_folder = os.path.join(get_cache_dir(), 'BFM')
        weight_path = os.path.join(get_cache_dir(), 'epoch_20.pth')
        self.deep_facerecon = build_model('DeepFaceRecon',
                                          bfm_folder=bfm_folder,
                                          weight_path=weight_path)
        self.deep_facerecon.eval().cuda()

    def extract_depth(self, data_loader, generator, generator_kwargs):
        """Extracts depth maps from the generated images via the face
        reconstruction model."""
        training_set = data_loader.dataset
        fake_num = self.fake_num
        batch_size = 1
        g1 = torch.Generator(device=self.device)
        g1.manual_seed(self.seed)

        G = generator
        G_kwargs = generator_kwargs
        G_mode = G.training  # save model training mode.
        G.eval()

        self.logger.info(f'Extracting depth maps from the generated face images'
                         f' {self.log_tail}.',
                         is_verbose=True)
        self.logger.init_pbar()
        pbar_task = self.logger.add_pbar_task('Depth', total=fake_num)
        all_depths = []
        for start in range(0, self.replica_latent_num, batch_size):
            end = min(start + batch_size, self.replica_latent_num)
            with torch.no_grad():
                batch_codes = torch.randn((end - start, *self.latent_dim),
                                          generator=g1,
                                          device=self.device)
                if self.random_pose:
                    batch_labels = sample_camera_extrinsics(
                        batch_size=(end - start),
                        radius_strategy='fix',
                        radius_fix=1.0,
                        polar_strategy='normal',
                        polar_mean=np.pi / 2,
                        polar_stddev=0.155,
                        azimuthal_strategy='normal',
                        azimuthal_mean=np.pi / 2,
                        azimuthal_stddev=0.3)['cam2world_matrix']
                else:
                    np.random.seed(self.seed)
                    batch_labels = [
                        training_set.get_pose(
                            np.random.randint(len(training_set)))
                            for _ in range(end - start)
                    ]
                    batch_labels = torch.from_numpy(
                        np.stack(batch_labels)).pin_memory().to(self.device)

                if self.eg3d_mode:
                    batch_results = G(batch_codes, batch_labels, **G_kwargs)
                else:
                    batch_results = G(batch_codes,
                                      cam2world_matrix=batch_labels,
                                      **G_kwargs)

                batch_images = batch_results['image']
                batch_images = F.interpolate(batch_images, size=(224, 224))
                batch_images = (batch_images + 1.0) / 2.0
                batch_gt_depths = self.deep_facerecon(batch_images)[
                    'depth']

                batch_depths = batch_results['image_depth']
                batch_depths = F.interpolate(batch_depths, size=(224, 224))

                batch_pair_depths = torch.cat([batch_depths, batch_gt_depths],
                                              dim=1)
                gathered_pair_depths = self.gather_batch_results(
                    batch_pair_depths)
                self.append_batch_results(gathered_pair_depths, all_depths)
            self.logger.update_pbar(pbar_task, (end - start) * self.world_size)
        self.logger.close_pbar()
        all_depths = self.gather_all_results(all_depths)[:fake_num]

        if self.is_chief:
            assert all_depths.shape == (fake_num, 2, 224, 224)
        else:
            assert len(all_depths) == 0
            all_depths = None

        if G_mode:
            G.train()  # restore model training mode.

        self.sync()
        return all_depths

    def evaluate(self, data_loader, generator, generator_kwargs):
        pair_depths = self.extract_depth(data_loader, generator,
                                         generator_kwargs)
        if self.is_chief:
            assert pair_depths.shape[1] == 2
            all_depth_l2_dist = []
            for i in range(self.fake_num):
                gan_depth = pair_depths[i, 0]
                gt_depth = pair_depths[i, 1]

                mask = np.logical_or(np.isclose(gt_depth, 0),
                                     np.isclose(gan_depth, 0))

                gan_depth = gan_depth / self.scale
                gan_depth[mask] = 0
                gan_depth_valid = gan_depth[np.logical_not(mask)]
                gan_depth_valid = gan_depth_valid - np.mean(gan_depth_valid)
                gan_depth_valid = gan_depth_valid / np.std(gan_depth_valid)

                gt_depth = gt_depth / self.scale
                gt_depth[mask] = 0
                gt_depth_valid = gt_depth[np.logical_not(mask)]
                gt_depth_valid = gt_depth_valid - np.mean(gt_depth_valid)
                gt_depth_valid = gt_depth_valid / np.std(gt_depth_valid)

                depth_l2_dist = np.mean((gt_depth_valid - gan_depth_valid)**2)
                all_depth_l2_dist.append(depth_l2_dist)

            all_depth_l2_dist = np.array(all_depth_l2_dist)
            all_depth_l2_dist = all_depth_l2_dist[~np.isnan(all_depth_l2_dist)]
            all_depth_l2_dist = all_depth_l2_dist[~np.isinf(all_depth_l2_dist)]
            depth_error = np.mean(all_depth_l2_dist)

            result = {self.name: float(depth_error)}
        else:
            assert pair_depths is None
            result = None
        self.sync()
        return result

    def _is_better_than(self, metric_name, new, ref):
        """Lower depth error is better."""
        if metric_name == self.name:
            return ref is None or new < ref
        return None

    def save(self, result, target_filename=None, log_suffix=None, tag=None):
        if not self.is_chief:
            assert result is None
            self.sync()
            return

        assert isinstance(result, dict)
        depth_error = result[self.name]
        assert isinstance(depth_error, float)
        prefix = f'Evaluating `{self.name}`: '
        if log_suffix is None:
            msg = f'{prefix}{depth_error:.3f}.'
        else:
            msg = f'{prefix}{depth_error:.3f}, {log_suffix}.'
        self.logger.info(msg)

        with open(os.path.join(self.work_dir, f'{self.name}.txt'), 'a+') as f:
            date = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f'[{date}] {msg}\n')

        # Save to TensorBoard if needed.
        if self.tb_writer is not None:
            if tag is None:
                self.logger.warning('`Tag` is missing when writing data to '
                                    'TensorBoard, hence, the data may be mixed '
                                    'up!')
            self.tb_writer.add_scalar(f'Metrics/{self.name}', depth_error,
                                      tag)
            self.tb_writer.flush()
        self.sync()

    def info(self):
        metric_info = super().info()
        metric_info['Num samples'] = self.fake_num
        return metric_info
