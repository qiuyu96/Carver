# python3.8
"""Contains the class to evaluate EG3D model by saving snapshots.

Basically, this class traces the quality of images synthesized by EG3D model.
"""

import os.path
import numpy as np

import torch
import torch.nn.functional as F

from utils.visualizers import GridVisualizer
from utils.image_utils import postprocess_image
from .base_gan_metric import BaseGANMetric

__all__ = ['GANSnapshot_EG3D_Image', 'GANSnapshot_EG3D_Depth']


class GANSnapshot_EG3D_Image(BaseGANMetric):
    """Defines the class for saving IMAGE snapshots synthesized by GANs."""

    def __init__(self,
                 name='snapshot_eg3d_image',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 latent_num=-1,
                 latent_dim=512,
                 latent_codes=None,
                 label_dim=0,
                 labels=None,
                 seed=0,
                 min_val=-1.0,
                 max_val=1.0,
                 random_pose=False
                 ):
        """Initializes the class with number of samples for each snapshot.

        Args:
            latent_num: Number of latent codes used for each snapshot.
                (default: -1)
            min_val: Minimum pixel value of the synthesized images. This field
                is particularly used for image visualization. (default: -1.0)
            max_val: Maximum pixel value of the synthesized images. This field
                is particularly used for image visualization. (default: 1.0)
        """
        super().__init__(name=name,
                         work_dir=work_dir,
                         logger=logger,
                         tb_writer=tb_writer,
                         batch_size=batch_size,
                         latent_num=latent_num,
                         latent_dim=latent_dim,
                         latent_codes=latent_codes,
                         label_dim=label_dim,
                         labels=labels,
                         seed=seed)
        self.min_val = min_val
        self.max_val = max_val
        self.visualizer = GridVisualizer()
        self.random_pose = random_pose

    def synthesize(self, data_loader, generator, generator_kwargs):
        """Synthesizes image with the generator."""
        training_set = data_loader.dataset
        latent_num = self.latent_num
        batch_size = self.batch_size
        g1 = torch.Generator(device=self.device)
        g1.manual_seed(self.seed)

        G = generator
        G_kwargs = generator_kwargs
        G_mode = G.training  # save model training mode.
        G.eval()

        self.logger.info(f'Synthesizing {latent_num} images {self.log_tail}.',
                         is_verbose=True)
        self.logger.init_pbar()
        pbar_task = self.logger.add_pbar_task('Synthesis', total=latent_num)
        all_images = []
        for start in range(0, self.replica_latent_num, batch_size):
            end = min(start + batch_size, self.replica_latent_num)
            with torch.no_grad():
                batch_codes = torch.randn((batch_size, *self.latent_dim),
                                              generator=g1,
                                              device=self.device)
                if self.random_pose:
                    batch_labels = self.sample_random_pose().to(self.device)
                else:
                    batch_labels = [
                        training_set.get_pose(
                            np.random.randint(len(training_set)))
                            for _ in range(self.batch_size)
                    ]
                    batch_labels = torch.from_numpy(
                        np.stack(batch_labels)).pin_memory().to(self.device)
                batch_images = G(batch_codes, batch_labels,
                                 **G_kwargs)['image']
                gathered_images = self.gather_batch_results(batch_images)
                self.append_batch_results(gathered_images, all_images)
            self.logger.update_pbar(pbar_task, (end - start) * self.world_size)
        self.logger.close_pbar()
        all_images = self.gather_all_results(all_images)[:latent_num]

        if self.is_chief:
            assert all_images.shape[0] == latent_num
        else:
            assert len(all_images) == 0
            all_images = None

        if G_mode:
            G.train()  # restore model training mode.

        self.sync()
        return all_images

    def evaluate(self, _data_loader, generator, generator_kwargs):
        images = self.synthesize(_data_loader, generator, generator_kwargs)
        if self.is_chief:
            result = {self.name: images}
        else:
            assert images is None
            result = None
        self.sync()
        return result

    def _is_better_than(self, metric_name, new, ref):
        """GAN snapshot is not supposed to judge performance."""
        return None

    def save(self, result, target_filename=None, log_suffix=None, tag=None):
        if not self.is_chief:
            assert result is None
            self.sync()
            return

        assert isinstance(result, dict)
        images = result[self.name]
        assert isinstance(images, np.ndarray)
        images = postprocess_image(
            images, min_val=self.min_val, max_val=self.max_val)
        filename = target_filename or self.name
        save_path = os.path.join(self.work_dir, f'{filename}.png')
        self.visualizer.visualize_collection(images, save_path)

        prefix = f'Evaluating `{self.name}` with {self.latent_num} samples'
        if log_suffix is None:
            msg = f'{prefix}.'
        else:
            msg = f'{prefix}, {log_suffix}.'
        self.logger.info(msg)

        # Save to TensorBoard if needed.
        if self.tb_writer is not None:
            if tag is None:
                self.logger.warning('`Tag` is missing when writing data to '
                                    'TensorBoard, hence, the data may be mixed '
                                    'up!')
            self.tb_writer.add_image(self.name, self.visualizer.grid, tag,
                                     dataformats='HWC')
            self.tb_writer.flush()
        self.sync()


class GANSnapshot_EG3D_Depth(BaseGANMetric):
    """Defines the class for saving DEPTH snapshots synthesized by GANs."""

    def __init__(self,
                 name='snapshot_eg3d_image',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 latent_num=-1,
                 latent_dim=512,
                 latent_codes=None,
                 label_dim=0,
                 labels=None,
                 seed=0,
                 min_val=-1.0,
                 max_val=1.0,
                 random_pose=False):
        """Initializes the class with number of samples for each snapshot.

        Args:
            latent_num: Number of latent codes used for each snapshot.
                (default: -1)
            min_val: Minimum pixel value of the synthesized images. This field
                is particularly used for image visualization. (default: -1.0)
            max_val: Maximum pixel value of the synthesized images. This field
                is particularly used for image visualization. (default: 1.0)
        """
        super().__init__(name=name,
                         work_dir=work_dir,
                         logger=logger,
                         tb_writer=tb_writer,
                         batch_size=batch_size,
                         latent_num=latent_num,
                         latent_dim=latent_dim,
                         latent_codes=latent_codes,
                         label_dim=label_dim,
                         labels=labels,
                         seed=seed)
        self.min_val = min_val
        self.max_val = max_val
        self.visualizer = GridVisualizer()
        self.random_pose = random_pose

    def synthesize(self, data_loader, generator, generator_kwargs):
        """Synthesizes image with the generator."""
        training_set = data_loader.dataset
        latent_num = self.latent_num
        batch_size = self.batch_size
        g1 = torch.Generator(device=self.device)
        g1.manual_seed(self.seed)

        G = generator
        G_kwargs = generator_kwargs
        G_mode = G.training  # save model training mode.
        G.eval()

        self.logger.info(
            f'Synthesizing {latent_num} depth images {self.log_tail}.',
            is_verbose=True)
        self.logger.init_pbar()
        pbar_task = self.logger.add_pbar_task('Synthesis', total=latent_num)
        all_images = []
        for start in range(0, self.replica_latent_num, batch_size):
            end = min(start + batch_size, self.replica_latent_num)
            with torch.no_grad():
                batch_codes = torch.randn((batch_size, *self.latent_dim),
                                              generator=g1,
                                              device=self.device)
                if self.random_pose:
                    batch_labels = self.sample_random_pose().to(self.device)
                else:
                    batch_labels = [
                        training_set.get_pose(
                            np.random.randint(len(training_set)))
                            for _ in range(self.batch_size)
                    ]
                    batch_labels = torch.from_numpy(
                        np.stack(batch_labels)).pin_memory().to(self.device)
                batch_images = G(batch_codes, batch_labels,
                                 **G_kwargs)['image_depth']
                gathered_images = self.gather_batch_results(batch_images)
                self.append_batch_results(gathered_images, all_images)
            self.logger.update_pbar(pbar_task, (end - start) * self.world_size)
        self.logger.close_pbar()
        all_images = self.gather_all_results(all_images)[:latent_num]

        if self.is_chief:
            assert all_images.shape[0] == latent_num
        else:
            assert len(all_images) == 0
            all_images = None

        if G_mode:
            G.train()  # restore model training mode.

        self.sync()
        return all_images

    def evaluate(self, _data_loader, generator, generator_kwargs):
        images = self.synthesize(_data_loader, generator, generator_kwargs)
        if self.is_chief:
            result = {self.name: images}
        else:
            assert images is None
            result = None
        self.sync()
        return result

    def _is_better_than(self, metric_name, new, ref):
        """GAN snapshot is not supposed to judge performance."""
        return None

    def save(self, result, target_filename=None, log_suffix=None, tag=None):
        if not self.is_chief:
            assert result is None
            self.sync()
            return

        assert isinstance(result, dict)
        images = result[self.name]
        assert isinstance(images, np.ndarray)
        images = postprocess_image(
            images, min_val=images.min(), max_val=images.max())
        filename = target_filename or self.name
        save_path = os.path.join(self.work_dir, f'{filename}.png')
        self.visualizer.visualize_collection(images, save_path)

        prefix = f'Evaluating `{self.name}` with {self.latent_num} samples'
        if log_suffix is None:
            msg = f'{prefix}.'
        else:
            msg = f'{prefix}, {log_suffix}.'
        self.logger.info(msg)

        # Save to TensorBoard if needed.
        if self.tb_writer is not None:
            if tag is None:
                self.logger.warning('`Tag` is missing when writing data to '
                                    'TensorBoard, hence, the data may be mixed '
                                    'up!')
            self.tb_writer.add_image(self.name, self.visualizer.grid, tag,
                                     dataformats='HWC')
            self.tb_writer.flush()
        self.sync()
