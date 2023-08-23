# python3.8
"""Collects all runners."""

from .eg3d_runner import EG3DRunner
from .pigan_runner import PiGANRunner
from .volumegan_runner import VolumeGANRunner
from .stylenerf_runner import StyleNeRFRunner
from .graf_runner import GRAFRunner
from .gram_runner import GRAMRunner
from .epigraf_runner import EpiGRAFRunner
from .stylesdf_runner import StyleSDFRunner
from .giraffe_runner import GIRAFFERunner

__all__ = ['build_runner']

_RUNNERS = {
    'EG3DRunner': EG3DRunner,
    'PiGANRunner': PiGANRunner,
    'VolumeGANRunner': VolumeGANRunner,
    'StyleNeRFRunner': StyleNeRFRunner,
    'GRAFRunner': GRAFRunner,
    'GRAMRunner': GRAMRunner,
    'EpiGRAFRunner': EpiGRAFRunner,
    'StyleSDFRunner': StyleSDFRunner,
    'GIRAFFERunner': GIRAFFERunner,
}


def build_runner(config):
    """Builds a runner with given configuration.

    Args:
        config: Configurations used to build the runner.

    Raises:
        ValueError: If the `config.runner_type` is not supported.
    """
    if not isinstance(config, dict) or 'runner_type' not in config:
        raise ValueError('`runner_type` is missing from configuration!')

    runner_type = config['runner_type']
    if runner_type not in _RUNNERS:
        raise ValueError(f'Invalid runner type: `{runner_type}`!\n'
                         f'Types allowed: {list(_RUNNERS)}.')
    return _RUNNERS[runner_type](config)
