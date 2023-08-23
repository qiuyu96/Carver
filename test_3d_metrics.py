# python3.8
"""Test 3D metrics."""

import argparse

import torch

from datasets import build_dataset
from models import build_model
from metrics import build_metric
from utils.loggers import build_logger
from utils.parsing_utils import parse_bool
from utils.parsing_utils import parse_json
from utils.dist_utils import init_dist
from utils.dist_utils import exit_dist


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Run 3D metric test.')
    parser.add_argument('--eg3d_mode', type=parse_bool, default=True,
                        help='Whether to evaluate in EG3D mode. (default: '
                             '%(default)s)')
    parser.add_argument('--random_pose', type=parse_bool, default=False,
                        help='Whether to evaluate with random pose. (default: '
                             '%(default)s)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to the dataset used for metric computation.')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the pre-trained model weights.')
    parser.add_argument('--G_kwargs', type=parse_json, default={},
                        help='Runtime keyword arguments for generator. Please '
                             'wrap the argument into single quotes with '
                             'keywords in double quotes. Beside, remove any '
                             'whitespace to avoid mis-parsing. For example, to '
                             'turn on truncation with probability 0.5 on 2 '
                             'layers, pass '
                             '`--G_kwargs \'{"truncation_psi":0.5,\'`. '
                             '(default: %(default)s)')
    parser.add_argument('--work_dir', type=str,
                        default='work_dirs/metric_tests',
                        help='Working directory for metric test. (default: '
                             '%(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for generating fake images. '
                             '(default: %(default)s)')
    parser.add_argument('--real_num', type=int, default=-1,
                        help='Number of real data used for testing. Negative '
                             'means using all data. (default: %(default)s)')
    parser.add_argument('--fake_num', type=int, default=1024,
                        help='Number of fake data used for testing. (default: '
                             '%(default)s)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size used for metric computation. '
                             '(default: %(default)s)')
    parser.add_argument('--test_fid', type=parse_bool, default=False,
                        help='Whether to test FID. (default: %(default)s)')
    parser.add_argument('--test_identity', type=parse_bool, default=False,
                        help='Whether to test identity. (default: %(default)s)')
    parser.add_argument('--align_face', type=parse_bool, default=False,
                        help='Whether to align face images before face '
                             'identity evluation. (default: %(defalut)s)')
    parser.add_argument('--test_depth', type=parse_bool, default=False,
                        help='Whether to test depth. (default: %(default)s)')
    parser.add_argument('--test_pose', type=parse_bool, default=False,
                        help='Whether to test pose. (default: %(default)s)')
    parser.add_argument('--test_reprojection_error', type=parse_bool,
                        default=False, help='Whether to test reprojection '
                                            'error. (default: %(default)s)')
    parser.add_argument('--test_snapshot', type=parse_bool, default=False,
                        help='Whether to test saving snapshot. '
                             '(default: %(default)s)')
    parser.add_argument('--test_snapshot_multiview', type=parse_bool,
                        default=False,
                        help='Whether to test saving multiview snapshot. '
                             '(default: %(default)s)')
    parser.add_argument('--launcher', type=str, default='pytorch',
                        choices=['pytorch', 'slurm'],
                        help='Distributed launcher. (default: %(default)s)')
    parser.add_argument('--backend', type=str, default='nccl',
                        choices=['nccl', 'gloo', 'mpi'],
                        help='Distributed backend. (default: %(default)s)')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Replica rank on the current node. This field is '
                             'required by `torch.distributed.launch`. '
                             '(default: %(default)s)')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Initialize distributed environment.
    init_dist(launcher=args.launcher, backend=args.backend)

    # CUDNN settings.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    state = torch.load(args.model)
    G = build_model(**state['model_kwargs_init']['generator_smooth'])
    G.load_state_dict(state['models']['generator_smooth'])
    G.eval().cuda()

    data_transform_kwargs = dict(
        image_size=G.resolution, image_channels=G.image_channels)
    dataset_kwargs = dict(dataset_type='EG3DDataset',
                          root_dir=args.dataset,
                          annotation_path=None,
                          annotation_meta=None,
                          max_samples=args.real_num,
                          mirror=False,
                          transform_kwargs=data_transform_kwargs)
    data_loader_kwargs = dict(data_loader_type='iter',
                              repeat=1,
                              num_workers=4,
                              prefetch_factor=2,
                              pin_memory=True)
    data_loader = build_dataset(for_training=False,
                                batch_size=args.batch_size,
                                dataset_kwargs=dataset_kwargs,
                                data_loader_kwargs=data_loader_kwargs)

    if torch.distributed.get_rank() == 0:
        logger = build_logger('normal', logfile=None, verbose_log=True)
    else:
        logger = build_logger('dummy')

    real_num = (len(data_loader.dataset)
                if args.real_num < 0 else args.real_num)
    print(f'Image size: {G.resolution}')
    if args.test_fid:
        logger.info('========== Test FID ==========')
        assert args.eg3d_mode == True and args.random_pose == False
        metric = build_metric('FIDEG3DMetric',
                              name=f'fid{args.fake_num}_real{real_num}',
                              work_dir=args.work_dir,
                              logger=logger,
                              seed=args.seed,
                              batch_size=args.batch_size,
                              image_size=G.resolution,
                              latent_dim=G.z_dim,
                              label_dim=G.label_dim,
                              real_num=args.real_num,
                              fake_num=args.fake_num)
        result = metric.evaluate(data_loader, G, args.G_kwargs)
        metric.save(result)
    if args.test_identity:
        logger.info('========== Test Identity ==========')
        metric = build_metric('FaceIDMetric',
                              name=f'faceid_{args.fake_num}',
                              work_dir=args.work_dir,
                              logger=logger,
                              seed=args.seed,
                              batch_size=args.batch_size,
                              latent_dim=G.z_dim,
                              label_dim=G.label_dim,
                              fake_num=args.fake_num,
                              align_face=args.align_face,
                              random_pose=args.random_pose,
                              eg3d_mode=args.eg3d_mode)
        result = metric.evaluate(data_loader, G, args.G_kwargs)
        metric.save(result)
    if args.test_depth:
        logger.info('========== Test Depth ==========')
        metric = build_metric('DepthEG3DMetric',
                              name=f'depth_{args.fake_num}',
                              work_dir=args.work_dir,
                              logger=logger,
                              seed=args.seed,
                              batch_size=args.batch_size,
                              latent_dim=G.z_dim,
                              label_dim=G.label_dim,
                              fake_num=args.fake_num,
                              random_pose=args.random_pose,
                              eg3d_mode=args.eg3d_mode)
        result = metric.evaluate(data_loader, G, args.G_kwargs)
        metric.save(result)
    if args.test_pose:
        logger.info('========== Test Pose ==========')
        metric = build_metric('PoseEG3DMetric',
                              name=f'pose_{args.fake_num}',
                              work_dir=args.work_dir,
                              logger=logger,
                              seed=args.seed,
                              batch_size=args.batch_size,
                              latent_dim=G.z_dim,
                              label_dim=G.label_dim,
                              fake_num=args.fake_num,
                              random_pose=args.random_pose,
                              eg3d_mode=args.eg3d_mode)
        result = metric.evaluate(data_loader, G, args.G_kwargs)
        metric.save(result)
    if args.test_reprojection_error:
        logger.info('========== Test Reprojection Error ==========')
        metric = build_metric('ReprojectionError',
                              name=f'reproj_error_{args.fake_num}',
                              work_dir=args.work_dir,
                              logger=logger,
                              seed=args.seed,
                              batch_size=args.batch_size,
                              latent_dim=G.z_dim,
                              label_dim=G.label_dim,
                              fake_num=args.fake_num)
        result = metric.evaluate(data_loader, G, args.G_kwargs)
        metric.save(result)
    if args.test_snapshot:
        logger.info('========== Test GAN Snapshot ==========')
        metric = build_metric('GANSnapshot_EG3D_Image',
                              name='eg3d_image_snapshot',
                              work_dir=args.work_dir,
                              logger=logger,
                              seed=args.seed,
                              batch_size=args.batch_size,
                              latent_dim=G.z_dim,
                              label_dim=G.label_dim,
                              latent_num=min(args.fake_num, 50))
        result = metric.evaluate(data_loader, G, args.G_kwargs)
        metric.save(result)
    if args.test_snapshot_multiview:
        logger.info('========== Test GAN Snapshot Multiview==========')
        metric = build_metric('GANSnapshotMultiView',
                              name='snapshot_multiview',
                              work_dir=args.work_dir,
                              logger=logger,
                              batch_size=args.batch_size,
                              latent_dim=G.latent_dim,
                              label_dim=G.label_dim,
                              latent_num=4)
        result = metric.evaluate(data_loader, G, args.G_kwargs)
        metric.save(result)

    # Exit distributed environment.
    exit_dist()


if __name__ == '__main__':
    main()
