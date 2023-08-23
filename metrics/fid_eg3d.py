# python3.8
"""Contains the class to evaluate EG3D model with
Frechet Inception Distance (FID) .

FID metric is introduced in paper https://arxiv.org/pdf/1706.08500.pdf
"""

import os.path
import time
import numpy as np

import torch
import torch.nn.functional as F

from models import build_model
from utils.misc import get_cache_dir
from .base_gan_metric import BaseGANMetric
from .utils import compute_fid_from_feature
from models.rendering.point_sampler import sample_camera_extrinsics

__all__ = ['FIDEG3DMetric', 'FID50KEG3D', 'FID50KFullEG3D']

FEATURE_DIM = 2048  # Dimension of inception feature.


class FIDEG3DMetric(BaseGANMetric):
    """Defines the class for FID metric computation."""

    def __init__(self,
                 name='FID',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 latent_dim=512,
                 latent_codes=None,
                 label_dim=0,
                 labels=None,
                 seed=0,
                 real_num=-1,
                 fake_num=-1,
                 image_size=512,
                 random_pose=False):
        """Initializes the class with number of real/fakes samples for FID.

        Args:
            real_num: Number of real images used for FID evaluation. If not set,
                all images from the given evaluation dataset will be used.
                (default: -1)
            fake_num: Number of fake images used for FID evaluation.
                (default: -1)
        """
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
        self.real_num = real_num
        self.fake_num = fake_num
        self.image_size = image_size
        self.random_pose = random_pose

        # Build inception model for feature extraction.
        self.inception_model = build_model('InceptionModel', align_tf=True)

    def extract_real_features(self, data_loader):
        """Extracts inception features from real data."""
        if self.real_num < 0:
            real_num = len(data_loader.dataset)
        else:
            real_num = min(self.real_num, len(data_loader.dataset))

        dataset_name = os.path.splitext(
            os.path.basename(data_loader.dataset.root_dir))[0]
        cache_name = (f'{dataset_name}_{self.image_size}_{real_num}'
                      f'_inception_feature.npy')
        cache_path = os.path.join(get_cache_dir(), cache_name)

        if os.path.exists(cache_path):
            self.logger.info(f'Loading statistics of real data from cache '
                             f'`{cache_path}` {self.log_tail}.',
                             is_verbose=True)
            all_features = np.load(cache_path) if self.is_chief else None
            self.sync()
            return all_features

        self.logger.info(f'Extracting inception features from real data '
                         f'{self.log_tail}.',
                         is_verbose=True)
        self.logger.init_pbar()
        pbar_task = self.logger.add_pbar_task('Real', total=real_num)
        all_features = []
        batch_size = data_loader.batch_size
        replica_num = self.get_replica_num(real_num)
        for batch_idx in range(len(data_loader)):
            if batch_idx * batch_size >= replica_num:
                # NOTE: Here, we always go through the entire dataset to make
                # sure the next evaluator can visit the data loader from the
                # beginning.
                _batch_data = next(data_loader)
                continue
            with torch.no_grad():
                batch_data = next(data_loader)['image'].cuda().detach()
                batch_features = self.inception_model(batch_data)
                gathered_features = self.gather_batch_results(batch_features)
                self.append_batch_results(gathered_features, all_features)
            self.logger.update_pbar(pbar_task, batch_size * self.world_size)
        self.logger.close_pbar()
        all_features = self.gather_all_results(all_features)[:real_num]

        self.logger.info(f'Saving statistics of real data to cache '
                         f'`{cache_path}` {self.log_tail}.')
        if self.is_chief:
            assert all_features.shape == (real_num, FEATURE_DIM)
            np.save(cache_path, all_features)
        else:
            assert len(all_features) == 0
            all_features = None
        self.sync()
        return all_features

    def extract_fake_features(self, data_loader, generator, generator_kwargs):
        """Extracts inception features from fake data."""
        training_set = data_loader.dataset
        fake_num = self.fake_num
        batch_size = self.batch_size
        g1 = torch.Generator(device=self.device)
        g1.manual_seed(self.seed)

        G = generator
        G_kwargs = generator_kwargs
        G_mode = G.training  # save model training mode.
        G.eval()

        self.logger.info(f'Extracting inception features from fake data '
                         f'{self.log_tail}.',
                         is_verbose=True)
        self.logger.init_pbar()
        pbar_task = self.logger.add_pbar_task('Fake', total=fake_num)
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
                    batch_labels = [
                        training_set.get_pose(
                            np.random.randint(len(training_set)))
                            for _ in range((end - start))
                    ]
                    batch_labels = torch.from_numpy(
                        np.stack(batch_labels)).pin_memory().to(self.device)
                batch_images = G(batch_codes, batch_labels,
                                 **G_kwargs)['image']
                batch_features = self.inception_model(batch_images)
                gathered_features = self.gather_batch_results(batch_features)
                self.append_batch_results(gathered_features, all_features)
            self.logger.update_pbar(pbar_task, (end - start) * self.world_size)
        self.logger.close_pbar()
        all_features = self.gather_all_results(all_features)[:fake_num]

        if self.is_chief:
            assert all_features.shape == (fake_num, FEATURE_DIM)
        else:
            assert len(all_features) == 0
            all_features = None

        if G_mode:
            G.train()  # restore model training mode.

        self.sync()
        return all_features

    def evaluate(self, data_loader, generator, generator_kwargs):
        real_features = self.extract_real_features(data_loader)
        fake_features = self.extract_fake_features(data_loader, generator,
                                                   generator_kwargs)
        if self.is_chief:
            fid = compute_fid_from_feature(fake_features, real_features)
            result = {self.name: fid}
        else:
            assert real_features is None and fake_features is None
            result = None
        self.sync()
        return result

    def _is_better_than(self, metric_name, new, ref):
        """Lower FID is better."""
        if metric_name == self.name:
            return ref is None or new < ref
        return None

    def save(self, result, target_filename=None, log_suffix=None, tag=None):
        if not self.is_chief:
            assert result is None
            self.sync()
            return

        assert isinstance(result, dict)
        fid = result[self.name]
        assert isinstance(fid, float)
        prefix = f'Evaluating `{self.name}`: '
        if log_suffix is None:
            msg = f'{prefix}{fid:.3f}.'
        else:
            msg = f'{prefix}{fid:.3f}, {log_suffix}.'
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
            self.tb_writer.add_scalar(f'Metrics/{self.name}', fid, tag)
            self.tb_writer.flush()
        self.sync()

    def info(self):
        metric_info = super().info()
        metric_info['Num real samples'] = self.real_num
        metric_info['Num fake samples'] = self.fake_num
        return metric_info


class FID50KEG3D(FIDEG3DMetric):
    """Defines the class for FID50K metric computation.

    50_000 real/fake samples will be used for feature extraction.
    """

    def __init__(self,
                 name='FID50K',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 latent_dim=512,
                 latent_codes=None,
                 label_dim=0,
                 labels=None,
                 seed=0,
                 image_size=512,
                 random_pose=False):
        super().__init__(name=name,
                         work_dir=work_dir,
                         logger=logger,
                         tb_writer=tb_writer,
                         batch_size=batch_size,
                         latent_dim=latent_dim,
                         latent_codes=latent_codes,
                         label_dim=label_dim,
                         labels=labels,
                         seed=seed,
                         real_num=50_000,
                         fake_num=50_000,
                         image_size=image_size,
                         random_pose=random_pose)


class FID50KFullEG3D(FIDEG3DMetric):
    """Defines the class for FID50KFull metric computation.

    50_000 fake samples and ALL real samples will be used for feature
    extraction.
    """

    def __init__(self,
                 name='FID50KFull',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 latent_dim=512,
                 latent_codes=None,
                 label_dim=0,
                 labels=None,
                 seed=0,
                 image_size=512,
                 random_pose=False):
        super().__init__(name=name,
                         work_dir=work_dir,
                         logger=logger,
                         tb_writer=tb_writer,
                         batch_size=batch_size,
                         latent_dim=latent_dim,
                         latent_codes=latent_codes,
                         label_dim=label_dim,
                         labels=labels,
                         seed=seed,
                         real_num=-1,
                         fake_num=50_000,
                         image_size=image_size,
                         random_pose=random_pose)


class FID2KFullEG3D(FIDEG3DMetric):
    """Defines the class for FID2KFull metric computation.

    2048 fake samples and ALL real samples will be used for feature
    extraction.
    """

    def __init__(self,
                 name='FID2KFull',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 latent_dim=512,
                 latent_codes=None,
                 label_dim=0,
                 labels=None,
                 seed=0,
                 image_size=512):
        super().__init__(name=name,
                         work_dir=work_dir,
                         logger=logger,
                         tb_writer=tb_writer,
                         batch_size=batch_size,
                         latent_dim=latent_dim,
                         latent_codes=latent_codes,
                         label_dim=label_dim,
                         labels=labels,
                         seed=seed,
                         real_num=-1,
                         fake_num=2_048,
                         image_size=image_size)