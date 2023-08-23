# python3.8
"""Test inference of 3d generation, including obtaining images, videos,
as well as geometries."""

import os
import argparse
import numpy as np
import cv2
import tqdm
import mrcfile
import skvideo.io

import torch
import torch.nn.functional as F

from utils.misc import bool_parser
from models import build_model
from models.rendering import PointSampler
from models.rendering.utils import LookAtPoseSampler


def create_voxel(N=256, voxel_corner=[0, 0, 0], voxel_length=2.0):
    """Creates a voxel grid.

    Args:
        N (int): Number of points in each side of the generated voxels.
            Defaults to 256.
        voxel_corner (list): Corner coordinate of the voxel, which represents
            (bottom, left, down) of the voxel. Defaults to [0, 0, 0].
        voxel_length (float): Side length of the voxel. Defaults to 2.0.

    Returns:
        A dictionary, containing:
            - `voxel_grid`: voxel grid, with shape [1, N * N * N, 3].
            - `voxel_origin`: origin of the voxel grid, with shape [3].
            - `voxel_size`: voxel grid size, i.e. the distance between two
                adjacent points in the voxel grid.
    """
    voxel_origin = np.array(voxel_corner) - voxel_length / 2
    voxel_size = voxel_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    grid = torch.zeros(N ** 3, 3)

    # Get the x, y, z index of each point in the grid.
    grid[:, 2] = overall_index % N
    grid[:, 1] = (overall_index.float() / N) % N
    grid[:, 0] = ((overall_index.float() / N) / N) % N

    # Get the x, y, z coordinate of each point in the grid.
    grid[:, 0] = (grid[:, 0] * voxel_size) + voxel_origin[2]
    grid[:, 1] = (grid[:, 1] * voxel_size) + voxel_origin[1]
    grid[:, 2] = (grid[:, 2] * voxel_size) + voxel_origin[0]

    voxel = {
        'voxel_grid': grid.unsqueeze(0),
        'voxel_origin': voxel_origin,
        'voxel_size': voxel_size
    }

    return voxel


def postprocess_image(image, min_val=-1.0, max_val=1.0):
    """Post-processes image to pixel range [0, 255] with dtype `uint8`.

    This function is particularly used to handle the results produced by deep
    models.

    NOTE: The input image is assumed to be with format `NCHW`, and the returned
    image will always be with format `NHWC`.

    Args:
        image (np.ndarray): The input image for post-processing, with shape
            [N, C, H, W].
        min_val (float): Expected minimum value of the input image.
        max_val (float): Expected maximum value of the input image.

    Returns:
        The post-processed image (np.ndarray), with shape [N, H, W, C].
    """
    assert isinstance(image, np.ndarray)

    image = image.astype(np.float64)
    image = (image - min_val) / (max_val - min_val) * 255
    image = np.clip(image + 0.5, 0, 255).astype(np.uint8)

    assert image.ndim == 4 and image.shape[1] in [1, 3, 4]

    return image.transpose(0, 2, 3, 1)


def sample_pose(point_sampling_kwargs=None):
    """Samples camera pose.

    Args:
        point_sampling_kwargs (dictionary): Point sampling related keywork
            arguments. Defaults to None.

    Returns:
        cam2world_matrix: Camera to world matrix, with shape [N, 16].
    """
    if point_sampling_kwargs is None:
        point_sampling_kwargs = {}
    point_sampler = PointSampler()
    sampling_point_res = point_sampler(**point_sampling_kwargs)
    cam2world_matrix = sampling_point_res['cam2world_matrix']

    return cam2world_matrix.flatten(1)


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Run 3D inference.')

    parser.add_argument('--model', type=str, required=True,
                        help='Path to the pre-trained model weights.')
    parser.add_argument('--work_dir', type=str,
                        default='work_dirs/visualize_3d',
                        help='Working directory for 3D inference.')
    parser.add_argument('--num', type=int, default=10,
                        help='Number of samples used for testing.')
    parser.add_argument('--truncation_psi', type=float, default=0.7,
                        help='Truncation.')
    parser.add_argument('--truncation_cutoff', type=int, default=14,
                        help='Number of fake data used for testing.')
    parser.add_argument('--save_image', type=bool_parser, default=True,
                        help='Whether to test saving snapshot for scene.')
    parser.add_argument('--save_shape', type=bool_parser, default=False,
                        help='Whether to test extracting shapes.')
    parser.add_argument('--save_video', type=bool_parser, default=False,
                        help='Whether to test saving video.')
    parser.add_argument('--rendering_resolution', type=int, default=64,
                        help='Neural rendering resolution.')
    parser.add_argument('--num_points', type=int, default=48,
                        help='Number of uniform samples to take per ray.')
    parser.add_argument('--ray_start', type=float, default=2.25,
                        help='Near point along each ray to start taking '
                             'samples.')
    parser.add_argument('--ray_end', type=float, default=3.3,
                        help='Far point along each ray to end taking samples.')
    parser.add_argument('--avg_camera_radius', type=float, default=2.7,
                        help='Specified camera orbit radius.'),
    parser.add_argument('--avg_camera_pivot', type=list, default=[0, 0, 0.2],
                        help='Center of camera rotation.'),
    parser.add_argument('--fov', type=float, default=12.0,
                        help='Field of view.')
    parser.add_argument('--focal', type=float, default=4.2647,
                        help='Focal length of camera.')
    parser.add_argument('--step', type=int, default=30,
                        help='Replica rank on the current node. This field is '
                             'required by `torch.distributed.launch`.')
    parser.add_argument('--coordinate_scale', type=float, default=1.0,
                        help='The side-length of the bounding box spanned by '
                             'the tri-planes.')
    parser.add_argument('--shape_res', type=int, default=512,
                        help='Resolution of the shape cube grid.')
    parser.add_argument('--seed', type=int, default=0, help='Radom seed.')

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.work_dir, exist_ok=True)

    state_dict = torch.load(args.model)
    G = build_model(**state_dict['model_kwargs_init']['generator'])
    G.load_state_dict(state_dict['models']['generator_smooth'])
    G.eval().cuda()
    G_kwargs = dict(truncation_psi=args.truncation_psi,
                    truncation_cutoff=args.truncation_cutoff,
                    noise_mode='const')

    assert (state_dict['models']['generator_smooth']['rendering_resolution'] ==
            args.rendering_resolution)

    # Create a tensor for padding to make the shape of pose to be [1, 25].
    padding_tensor = torch.zeros((1, 9), device=device)

    avg_camera_pivot = torch.tensor(args.avg_camera_pivot, device=device)

    # Note: When generating multi-view images (in diffrent `poses) of the
    # same identity (with same `batch_codes`), it is necessary to keep
    # `label_swapped` fixed due to 'generator pose conditioning'.
    # `label_swapped` is closely associated with the generation of latents
    # in W+ space, which will affect the face identity of the generated
    # images.
    pose_swapped = LookAtPoseSampler.sample(np.pi / 2,
                                            np.pi / 2,
                                            avg_camera_pivot,
                                            radius=args.avg_camera_radius,
                                            device=device).flatten(1)
    pose_swapped = torch.cat([pose_swapped, padding_tensor], dim=1)

    # Predefine camera trajectory, used for saving videos.
    trajectory = []
    angle_p = -0.2
    for angle_y in np.linspace(-0.5, 0.5, args.step):
        video_pose = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                              np.pi / 2 + angle_p,
                                              avg_camera_pivot,
                                              radius=args.avg_camera_radius,
                                              device=device).flatten(1)
        video_pose = torch.cat([video_pose, padding_tensor], dim=1)
        trajectory.append(video_pose)

    for i in tqdm.tqdm(range(args.num)):
        batch_codes = torch.from_numpy(
            np.random.RandomState(args.seed + i).randn(1, G.z_dim)).to(device)

        if args.save_image:
            angle_p = -0.2
            pose_idx = 0
            for angle_y, angle_p in [(.4, angle_p), (0, angle_p),
                                     (-.4, angle_p)]:
                poses = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                 np.pi / 2 + angle_p,
                                                 avg_camera_pivot,
                                                 radius=args.avg_camera_radius,
                                                 device=device).flatten(1)
                poses = torch.cat([poses, padding_tensor], dim=1)
                images = G(batch_codes,
                           poses,
                           label_swapped=pose_swapped,
                           **G_kwargs)['image']
                image = postprocess_image(images.detach().cpu().numpy())[0]
                cv2.imwrite(
                    os.path.join(args.work_dir, f'{i:06d}_{pose_idx}.jpg'),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                pose_idx = pose_idx + 1

        if args.save_video:
            frames = []
            for _, video_pose in enumerate(trajectory):
                images = G(batch_codes,
                           video_pose,
                           label_swapped=pose_swapped,
                           **G_kwargs)['image']  # [N, C, H, W], N = batch_size
                frames.append(images.detach().cpu())
            frames = torch.stack(frames,
                                 dim=0)  # [n, N, C, H, W], n = len(trajectory)
            frames = frames.permute(1, 0, 2, 3, 4)  # [N, n, C, H, W]
            frames = frames.detach().cpu().numpy()

            for idx in range(frames.shape[0]):
                traj_frames = frames[idx]  # [n, C, H, W]
                traj_frames = postprocess_image(traj_frames,
                                                min_val=-1,
                                                max_val=1)
                writer = skvideo.io.FFmpegWriter(
                    os.path.join(args.work_dir, f'{i:06d}.mp4'),
                    outputdict={'-pix_fmt': 'yuv420p', '-crf': '21'})
                for frame_idx in range(traj_frames.shape[0]):
                    writer.writeFrame(traj_frames[frame_idx])
                writer.close()

        if args.save_shape:
            max_batch = 1000000
            shape_res = args.shape_res

            # Create a voxel grid and feed the coordinates of its points to the
            # trained network to obtain the densities.
            voxel = create_voxel(N=shape_res,
                                 voxel_corner=[0, 0, 0],
                                 voxel_length=args.coordinate_scale * 1)
            voxel_grid = voxel['voxel_grid'].to(device) # [1, shape_res ** 3, 3]
            densities = torch.zeros(
                (voxel_grid.shape[0], voxel_grid.shape[1], 1), device=device)

            head = 0
            with tqdm.tqdm(total=voxel_grid.shape[1]) as pbar:
                with torch.no_grad():
                    while head < voxel_grid.shape[1]:
                        density = G.sample(
                            voxel_grid[:, head:head + max_batch],
                            batch_codes,
                            pose_swapped,
                            **G_kwargs)['density']
                        densities[:, head:head + max_batch] = density
                        head = head + max_batch
                        pbar.update(max_batch)

            densities = densities.reshape(
                (shape_res, shape_res, shape_res)).cpu().numpy()
            densities = np.flip(densities, 0)

            # Trim the border of the extracted cube.
            pad = int(30 * shape_res / 256)
            pad_value = -1000
            densities[:pad] = pad_value
            densities[-pad:] = pad_value
            densities[:, :pad] = pad_value
            densities[:, -pad:] = pad_value
            densities[:, :, :pad] = pad_value
            densities[:, :, -pad:] = pad_value

            with mrcfile.new_mmap(os.path.join(args.work_dir, f'{i:06d}.mrc'),
                                  overwrite=True,
                                  shape=densities.shape,
                                  mrc_mode=2) as mrc:
                mrc.data[:] = densities


if __name__ == '__main__':
    main()