# python3.8
"""Contains the class to evaluate identity consistency of face images generated
by 3D-aware GANs."""

import os.path
import time
import numpy as np

import cv2
import dlib
from skimage import transform as trans

import torch
import torch.nn.functional as F

from utils.misc import get_cache_dir
from models import build_model
from .base_gan_metric import BaseGANMetric
from models.rendering.point_sampler import sample_camera_extrinsics

FEATURE_DIM = 512  # Dimension of resnet100 feature.


class FaceIDMetric(BaseGANMetric):
    """Defines the class for face identity cosine distance computation."""

    def __init__(self,
                 name='FaceID',
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
                 align_face=False,
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
        self.align_face = align_face
        self.random_pose = random_pose
        self.eg3d_mode = eg3d_mode

        if self.align_face:
            self.face_aligner = FaceAligner()

        # Build inception model for feature extraction.
        self.iresnet100 = build_model('IResNet100', fp16=False)
        weight_path = os.path.join(get_cache_dir(),
                                   'glint360k_cosface_r100.pth')
        self.iresnet100.load_state_dict(torch.load(weight_path))
        self.iresnet100.eval().cuda()

    @staticmethod
    def tensor2image(tensor):
        """Converts tensor to image.

        Args:
            tensor (torch.tensor): Input tensor, with shape [C, H, W].

        Returns:
            image (np.array): Output image, with shape [H, W, C].
        """
        tensor = tensor.permute(1, 2, 0)
        image = tensor.detach().cpu().numpy()[:, :, ::-1]
        image = (image + 1) * 127.5
        image = image.clip(0, 255)
        image = image.astype(np.uint8)
        return image

    @staticmethod
    def image2tensor(image):
        """Converts image to tensor.

        Args:
            image (np.array): Output image, with shape [H, W, C].

        Returns:
            tensor (torch.tensor): Input tensor, with shape [C, H, W].
        """
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 127.5 - 1
        tensor = torch.from_numpy(image)
        tensor = tensor.permute(2, 0, 1)
        return tensor

    def align_batch_images(self, batch_images):
        """Aligns a batch of face images according to the detected facial
        landmarks.

        Args:
            batch_images (torch.tensor): Input images, with shape [N, C, H, W].

        Returns:
            aligned_batch_images (torch.tensor): Output aligned images, with the
        same shape as the input.
        """
        all_aligned_images = []
        for i in range(batch_images.shape[0]):
            image = self.tensor2image(batch_images[i])
            landmarks = self.face_aligner.detect_landmarks(image)
            if landmarks is None:
                return batch_images
            aligned_image = self.face_aligner.norm_crop(image, landmarks)
            aligned_image = self.image2tensor(aligned_image).to(self.device)
            all_aligned_images.append(aligned_image.unsqueeze(0))
        all_aligned_images = torch.cat(all_aligned_images,
                                       dim=0).to(self.device)

        return all_aligned_images

    def extract_features(self, data_loader, generator, generator_kwargs):
        """Extracts resnet100 features from two views of the fake face
        images."""
        training_set = data_loader.dataset
        fake_num = self.fake_num
        # Note: Here, only a batch size of `1` is supported due to unknown bugs.
        batch_size = 1
        g1 = torch.Generator(device=self.device)
        g1.manual_seed(self.seed)

        G = generator
        G_kwargs = generator_kwargs
        G_mode = G.training  # save model training mode.
        G.eval()

        # When evaluating consistency of face identity, ensure that the mapping
        # condition remains fixed.
        label_swapped = torch.tensor([
            1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 2, 0, 0, 0, 1, 4, 0, 0, 0, 4, 0,
            0, 0, 1
        ]).type(torch.float32)
        label_swapped = label_swapped.unsqueeze(0).to(self.device)

        self.logger.info(f'Extracting resnet100 features from two views of the '
                         f'fake face images {self.log_tail}.',
                         is_verbose=True)
        self.logger.init_pbar()
        pbar_task = self.logger.add_pbar_task('FaceID', total=fake_num)
        all_features = []

        for start in range(0, self.replica_latent_num, batch_size):
            end = min(start + batch_size, self.replica_latent_num)
            with torch.no_grad():
                batch_codes = torch.randn((end - start, *self.latent_dim),
                                          generator=g1,
                                          device=self.device)
                if self.random_pose:
                    batch_labels = sample_camera_extrinsics(
                        batch_size=2 * (end - start),
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
                            for _ in range(2 * (end - start))
                    ]
                    batch_labels = torch.from_numpy(
                        np.stack(batch_labels)).pin_memory().to(self.device)

                if self.eg3d_mode:
                    G_kwargs.update(
                        label_swapped=label_swapped.repeat(end - start, 1))
                    batch_images_view_A = G(batch_codes,
                                            batch_labels[:(end - start)],
                                            **G_kwargs)['image']
                    batch_images_view_B = G(batch_codes,
                                            batch_labels[(end - start):],
                                            **G_kwargs)['image']
                else:
                    batch_images_view_A = G(
                        batch_codes,
                        cam2world_matrix=batch_labels[:(end - start)],
                        **G_kwargs)['image']
                    batch_images_view_B = G(
                        batch_codes,
                        cam2world_matrix=batch_labels[(end - start):],
                        **G_kwargs)['image']

                if self.align_face:
                    batch_images_view_A = self.align_batch_images(
                        batch_images_view_A)
                    batch_images_view_B = self.align_batch_images(
                        batch_images_view_B)
                batch_images_view_A = F.interpolate(batch_images_view_A,
                                                    size=112)
                batch_images_view_B = F.interpolate(batch_images_view_B,
                                                    size=112)
                batch_features_view_A = self.iresnet100(batch_images_view_A)
                batch_features_view_A = F.normalize(batch_features_view_A,
                                                    dim=1)
                batch_features_view_B = self.iresnet100(batch_images_view_B)
                batch_features_view_B = F.normalize(batch_features_view_B,
                                                    dim=1)
                batch_features = torch.cat([
                    batch_features_view_A.unsqueeze(1),
                    batch_features_view_B.unsqueeze(1)
                ], dim=1)
                gathered_features = self.gather_batch_results(batch_features)
                self.append_batch_results(gathered_features, all_features)
            self.logger.update_pbar(pbar_task, (end - start) * self.world_size)
        self.logger.close_pbar()
        all_features = self.gather_all_results(all_features)[:fake_num]

        if self.is_chief:
            assert all_features.shape == (fake_num, 2, FEATURE_DIM)
        else:
            assert len(all_features) == 0
            all_features = None

        if G_mode:
            G.train()  # restore model training mode.

        self.sync()
        return all_features

    def evaluate(self, data_loader, generator, generator_kwargs):
        face_features = self.extract_features(data_loader, generator,
                                              generator_kwargs)
        if self.is_chief:
            assert face_features.shape[1] == 2
            f_A = face_features[:, 0]
            f_B = face_features[:, 1]
            cos_dist = np.sum(f_A * f_B, axis=1).mean()
            result = {self.name: float(cos_dist)}
        else:
            assert face_features is None
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
        cos_dist = result[self.name]
        assert isinstance(cos_dist, float)
        prefix = f'Evaluating `{self.name}`: '
        if log_suffix is None:
            msg = f'{prefix}{cos_dist:.3f}.'
        else:
            msg = f'{prefix}{cos_dist:.3f}, {log_suffix}.'
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
            self.tb_writer.add_scalar(f'Metrics/{self.name}', cos_dist, tag)
            self.tb_writer.flush()
        self.sync()

    def info(self):
        metric_info = super().info()
        metric_info['Num samples'] = self.fake_num
        return metric_info


class FaceAligner(object):
    def __init__(self, detect_5_keypoints=True):
        super().__init__()
        predictor_path = os.path.join(
                get_cache_dir(), 'shape_predictor_68_face_landmarks.dat')
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.detect_5_keypoints = detect_5_keypoints

        self.lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
        arcface_src_pts = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
             [41.5493, 92.3655], [70.7299, 92.2041]],
            dtype=np.float32)
        self.arcface_src_pts = np.expand_dims(arcface_src_pts, axis=0)

    def detect_landmarks(self, img):
        dets = self.detector(img, 1)
        if len(dets) < 1:
            return None
        pts = self.predictor(img, dets[0])
        landmarks = np.zeros((68, 2), dtype=int)
        for i in range(68):
            landmarks[i, :] = (pts.part(i).x, pts.part(i).y)
        if self.detect_5_keypoints:
            landmarks = self.extract_5p(landmarks)
        return landmarks

    def extract_5p(self, lm):
        lm5p = np.stack([
            lm[self.lm_idx[0], :],
            np.mean(lm[self.lm_idx[[1, 2]], :], 0),
            np.mean(lm[self.lm_idx[[3, 4]], :], 0),
            lm[self.lm_idx[5], :],
            lm[self.lm_idx[6], :]], axis=0)
        lm5p = lm5p[[1, 2, 0, 3, 4], :]
        return lm5p

    def estimate_norm(self, lmk):
        assert lmk.shape == (5, 2)
        tform = trans.SimilarityTransform()
        lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
        min_M = []
        min_index = []
        min_error = float('inf')
        src = self.arcface_src_pts
        for i in np.arange(src.shape[0]):
            tform.estimate(lmk, src[i])
            M = tform.params[0:2, :]
            results = np.dot(M, lmk_tran.T)
            results = results.T
            error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
            if error < min_error:
                min_error = error
                min_M = M
                min_index = i
        return min_M, min_index

    def norm_crop(self, img, landmark, image_size=112):
        M, _ = self.estimate_norm(landmark)
        warped = cv2.warpAffine(img,
                                M, (image_size, image_size),
                                borderValue=0.0)
        return warped
