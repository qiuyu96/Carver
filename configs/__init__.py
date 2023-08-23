# python3.8
"""Collects all configs."""

from .eg3d_config import EG3DConfig
from .pigan_config import PiGANConfig
from .volumegan_config import VolumeGANConfig
from .ablation3d_config import Ablation3DConfig
from .stylenerf_config import StyleNeRFConfig
from .graf_config import GRAFConfig
from .gram_config import GRAMConfig
from .epigraf_config import EpiGRAFConfig
from .stylesdf_config import StyleSDFConfig
from .giraffe_config import GIRAFFEConfig

__all__ = ['CONFIG_POOL', 'build_config']

CONFIG_POOL = [
    EG3DConfig,
    PiGANConfig,
    VolumeGANConfig,
    Ablation3DConfig,
    StyleNeRFConfig,
    GRAFConfig,
    GRAMConfig,
    EpiGRAFConfig,
    StyleSDFConfig,
    GIRAFFEConfig,
]


def build_config(invoked_command, kwargs):
    """Builds a configuration based on the invoked command.

    Args:
        invoked_command: The command that is invoked.
        kwargs: Keyword arguments passed from command line, which will be used
            to build the configuration.

    Raises:
        ValueError: If the `invoked_command` is missing.
    """
    for config in CONFIG_POOL:
        if config.name == invoked_command:
            return config(kwargs)
    raise ValueError(f'Invoked command `{invoked_command}` is missing!\n')
