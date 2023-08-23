# python3.8
"""Contains the class to evaluate pose accuracy of face images output by
3D-aware GANs."""

import os.path
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

import torch
import torch.nn.functional as F

from utils.misc import get_cache_dir
from models import build_model
from models.rendering.point_sampler import sample_camera_extrinsics
from .base_gan_metric import BaseGANMetric


class PoseEG3DMetric(BaseGANMetric):
    """Defines the class for evaluation of pose accuracy described in EG3D."""

    def __init__(self,
                 name='PoseEG3D',
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

        bfm_folder = os.path.join(get_cache_dir(), 'BFM')
        weight_path = os.path.join(get_cache_dir(), 'epoch_20.pth')
        self.deep_facerecon = build_model('DeepFaceRecon',
                                          bfm_folder=bfm_folder,
                                          weight_path=weight_path)
        self.deep_facerecon.eval().cuda()

    def extract_pose(self, data_loader, generator, generator_kwargs):
        """Extracts poses from the generated images via the face reconstruction
        model."""
        training_set = data_loader.dataset
        fake_num = self.fake_num
        batch_size = 1
        g1 = torch.Generator(device=self.device)
        g1.manual_seed(self.seed)

        G = generator
        G_kwargs = generator_kwargs
        G_mode = G.training  # save model training mode.
        G.eval()

        self.logger.info(f'Extracting poses from the generated fake face '
                         f'images {self.log_tail}.',
                         is_verbose=True)
        self.logger.init_pbar()
        pbar_task = self.logger.add_pbar_task('Pose', total=fake_num)
        all_features = []

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
                    batch_images = G(batch_codes, batch_labels,
                                     **G_kwargs)['image']
                else:
                    batch_images = G(batch_codes,
                                     cam2world_matrix=batch_labels,
                                     **G_kwargs)['image']

                batch_poses = batch_labels.flatten(1)[:, :16]  # [N, 16]
                batch_images = F.interpolate(batch_images, size=(224, 224))
                batch_images = (batch_images + 1.0) / 2.0
                batch_gt_results = self.deep_facerecon(batch_images)
                # `padding_tensor` is only to keep shape consistency.
                padding_tensor = torch.zeros((end - start, 10)).to(self.device)
                batch_gt_poses = torch.cat([batch_gt_results['angle'],
                                            batch_gt_results['trans'],
                                            padding_tensor], dim=1)  # [N, 16]
                batch_pair_poses = torch.cat(
                    [batch_poses.unsqueeze(1),
                     batch_gt_poses.unsqueeze(1)],
                    dim=1)  # [N, 2, 16]
                gathered_pair_poses = self.gather_batch_results(
                    batch_pair_poses)
                self.append_batch_results(gathered_pair_poses, all_features)
            self.logger.update_pbar(pbar_task, (end - start) * self.world_size)
        self.logger.close_pbar()
        all_features = self.gather_all_results(all_features)[:fake_num]

        if self.is_chief:
            assert all_features.shape == (fake_num, 2, 16)
        else:
            assert len(all_features) == 0
            all_features = None

        if G_mode:
            G.train()  # restore model training mode.

        self.sync()
        return all_features

    @staticmethod
    def compute_rotation(angles):
        """
        Converts Euler angles (radian) to rotation matrices.
        Args:
            angles (np.array): Euler angles in radian, with shape [N, 3].

        Return:
            rot (np.array): Rotation matrics, with shape [N, 3, 3].
        """

        batch_size = angles.shape[0]
        assert batch_size == 1
        x, y, z = angles[0, 0], angles[0, 1], angles[0, 2],

        rot_x = np.array([
            1, 0, 0,
            0, np.cos(x), -np.sin(x),
            0, np.sin(x), np.cos(x)
        ]).reshape([batch_size, 3, 3])

        rot_y = np.array([
            np.cos(y), 0, np.sin(y),
            0, 1, 0,
            -np.sin(y), 0, np.cos(y)
        ]).reshape([batch_size, 3, 3])

        rot_z = np.array([
            np.cos(z), -np.sin(z), 0,
            np.sin(z), np.cos(z), 0,
            0, 0, 1
        ]).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot

    def evaluate(self, data_loader, generator, generator_kwargs):
        pair_poses = self.extract_pose(data_loader, generator,
                                       generator_kwargs)

        if self.is_chief:
            all_pose_l2_dist = []
            for i in range(self.fake_num):
                gan_cam2world = pair_poses[i, 0]  # [16]
                gt_pose = pair_poses[i, 1][None]  # [1, 16]

                gan_cam2world = gan_cam2world.reshape(4, 4)
                if self.eg3d_mode:
                    gan_cam2world[:, 1:3] = -gan_cam2world[:, 1:3]
                gan_rotmat = R.from_matrix(gan_cam2world[:3, :3])
                gan_rotvec = gan_rotmat.as_rotvec()

                gt_angle = gt_pose[:, :3]  # [1, 3]
                gt_trans = gt_pose[:, 3:6]  # [1, 3]
                gt_world2cam = np.eye(4)
                gt_world2cam[:3, :3] = self.compute_rotation(gt_angle)[0]
                gt_world2cam[:3, 3] = gt_trans
                to_cam = np.eye(4)
                to_cam[2, 2] = -1
                to_cam[2, 3] = 10
                gt_world2cam = to_cam @ gt_world2cam
                gt_world2cam[2, :] = -gt_world2cam[2, :]
                gt_cam2world = np.linalg.inv(gt_world2cam)
                gt_rotmat = R.from_matrix(gt_cam2world[:3, :3])
                gt_rotvec = gt_rotmat.as_rotvec()

                pose_l2_dist = np.mean((gan_rotvec - gt_rotvec)**2)
                all_pose_l2_dist.append(pose_l2_dist)

            all_pose_l2_dist = np.array(all_pose_l2_dist)
            pose_error = np.mean(all_pose_l2_dist)

            result = {self.name: float(pose_error)}
        else:
            assert pair_poses is None
            result = None
        self.sync()
        return result

    def _is_better_than(self, metric_name, new, ref):
        """Lower pose error is better."""
        if metric_name == self.name:
            return ref is None or new < ref
        return None

    def save(self, result, target_filename=None, log_suffix=None, tag=None):
        if not self.is_chief:
            assert result is None
            self.sync()
            return

        assert isinstance(result, dict)
        pose_error = result[self.name]
        assert isinstance(pose_error, float)
        prefix = f'Evaluating `{self.name}`: '
        if log_suffix is None:
            msg = f'{prefix}{pose_error:.5f}.'
        else:
            msg = f'{prefix}{pose_error:.5f}, {log_suffix}.'
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
            self.tb_writer.add_scalar(f'Metrics/{self.name}', pose_error, tag)
            self.tb_writer.flush()
        self.sync()

    def info(self):
        metric_info = super().info()
        metric_info['Num samples'] = self.fake_num
        return metric_info