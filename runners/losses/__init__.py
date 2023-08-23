# python3.8
"""Collects all loss functions."""

from .eg3d_loss import EG3DLoss
from .pigan_loss import PiGANLoss
from .volumegan_loss import VolumeGANLoss
from .stylenerf_loss import StyleNeRFLoss
from .graf_loss import GRAFLoss
from .gram_loss import GRAMLoss
from .epigraf_loss import EpiGRAFLoss
from .stylesdf_loss import StyleSDFLoss
from .giraffe_loss import GIRAFFELoss

__all__ = ['build_loss']

_LOSSES = {
    'EG3DLoss': EG3DLoss,
    'PiGANLoss': PiGANLoss,
    'VolumeGANLoss': VolumeGANLoss,
    'StyleNeRFLoss': StyleNeRFLoss,
    'GRAFLoss': GRAFLoss,
    'GRAMLoss': GRAMLoss,
    'EpiGRAFLoss': EpiGRAFLoss,
    'StyleSDFLoss': StyleSDFLoss,
    'GIRAFFELoss': GIRAFFELoss,
}


def build_loss(runner, loss_type, **kwargs):
    """Builds a loss based on its class type.

    Args:
        runner: The runner on which the loss is built.
        loss_type: Class type to which the loss belongs, which is case
            sensitive.
        **kwargs: Additional arguments to build the loss.

    Raises:
        ValueError: If the `loss_type` is not supported.
    """
    if loss_type not in _LOSSES:
        raise ValueError(f'Invalid loss type: `{loss_type}`!\n'
                         f'Types allowed: {list(_LOSSES)}.')
    return _LOSSES[loss_type](runner, **kwargs)
