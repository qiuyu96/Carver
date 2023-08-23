# python3.8
"""Collects all metrics."""

from .gan_snapshot import GANSnapshot
from .fid import FIDMetric as FID
from .fid import FID50K
from .fid import FID50KFull
from .inception_score import ISMetric as IS
from .inception_score import IS50K
from .kid import KIDMetric as KID
from .kid import KID50K
from .kid import KID50KFull
from .fid_eg3d import FIDEG3DMetric
from .fid_eg3d import FID50KEG3D
from .fid_eg3d import FID50KFullEG3D
from .fid_eg3d import FID2KFullEG3D
from .gan_snapshot_eg3d import GANSnapshot_EG3D_Image
from .gan_snapshot_eg3d import GANSnapshot_EG3D_Depth
from .gan_snapshot_multiview import GANSnapshotMultiView
from .face_identity import FaceIDMetric
from .depth_eg3d import DepthEG3DMetric
from .pose_eg3d import PoseEG3DMetric
from .reprojection_error import ReprojectionError

__all__ = ['build_metric']

_METRICS = {
    'GANSnapshot': GANSnapshot,
    'FID': FID,
    'FID50K': FID50K,
    'FID50KFull': FID50KFull,
    'IS': IS,
    'IS50K': IS50K,
    'KID': KID,
    'KID50K': KID50K,
    'KID50KFull': KID50KFull,
    'FIDEG3DMetric': FIDEG3DMetric,
    'FID50KEG3D': FID50KEG3D,
    'FID50KFullEG3D': FID50KFullEG3D,
    'FID2KFullEG3D': FID2KFullEG3D,
    'GANSnapshot_EG3D_Image': GANSnapshot_EG3D_Image,
    'GANSnapshot_EG3D_Depth': GANSnapshot_EG3D_Depth,
    'GANSnapshotMultiView': GANSnapshotMultiView,
    'FaceIDMetric': FaceIDMetric,
    'DepthEG3DMetric': DepthEG3DMetric,
    'PoseEG3DMetric': PoseEG3DMetric,
    'ReprojectionError': ReprojectionError
}


def build_metric(metric_type, **kwargs):
    """Builds a metric evaluator based on its class type.

    Args:
        metric_type: Type of the metric, which is case sensitive.
        **kwargs: Configurations used to build the metric.

    Raises:
        ValueError: If the `metric_type` is not supported.
    """
    if metric_type not in _METRICS:
        raise ValueError(f'Invalid metric type: `{metric_type}`!\n'
                         f'Types allowed: {list(_METRICS)}.')
    return _METRICS[metric_type](**kwargs)
