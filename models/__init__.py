# python3.8
"""Collects all models."""

from .perceptual_model import PerceptualModel
from .inception_model import InceptionModel
from .iresnet import iresnet100
from .deep_facerecon import DeepFaceRecon
from .eg3d_generator import EG3DGenerator
from .eg3d_discriminator import DualDiscriminator
from .eg3d_discriminator import SingleDiscriminator
from .pigan_generator import PiGANGenerator
from .pigan_discriminator import PiGANDiscriminator
from .volumegan_generator import VolumeGANGenerator
from .volumegan_discriminator import VolumeGANDiscriminator
from .ablation3d_generator import Ablation3DGenerator
from .stylenerf_generator import StyleNeRFGenerator
from .stylenerf_discriminator import StyleNeRFDiscriminator
from .graf_generator import GRAFGenerator
from .graf_discriminator import GRAFDiscriminator
from .gram_generator import GRAMGenerator
from .gram_discriminator import GRAMDiscriminator
from .epigraf_generator import EpiGRAFGenerator
from .epigraf_discriminator import EpiGRAFDiscriminator
from .stylesdf_generator import StyleSDFGenerator
from .stylesdf_discriminator import StyleSDFDiscriminator
from .stylesdf_discriminator import StyleSDFDiscriminator_full
from .giraffe_generator import GIRAFFEGenerator
from .giraffe_discriminator import GIRAFFEDiscriminator

__all__ = ['build_model']

_MODELS = {
    'PerceptualModel': PerceptualModel.build_model,
    'InceptionModel': InceptionModel.build_model,
    'IResNet100': iresnet100,
    'DeepFaceRecon': DeepFaceRecon,
    'EG3DGenerator': EG3DGenerator,
    'EG3DDiscriminator': DualDiscriminator,
    'EG3DSingleDiscriminator': SingleDiscriminator,
    'PiGANGenerator': PiGANGenerator,
    'PiGANDiscriminator': PiGANDiscriminator,
    'VolumeGANGenerator': VolumeGANGenerator,
    'VolumeGANDiscriminator': VolumeGANDiscriminator,
    'Ablation3DGenerator': Ablation3DGenerator,
    'StyleNeRFGenerator': StyleNeRFGenerator,
    'StyleNeRFDiscriminator': StyleNeRFDiscriminator,
    'GRAFGenerator': GRAFGenerator,
    'GRAFDiscriminator': GRAFDiscriminator,
    'GRAMGenerator': GRAMGenerator,
    'GRAMDiscriminator': GRAMDiscriminator,
    'EpiGRAFGenerator': EpiGRAFGenerator,
    'EpiGRAFDiscriminator': EpiGRAFDiscriminator,
    'StyleSDFGenerator': StyleSDFGenerator,
    'StyleSDFDiscriminator': StyleSDFDiscriminator,
    'StyleSDFFullDiscriminator': StyleSDFDiscriminator_full,
    'GIRAFFEGenerator': GIRAFFEGenerator,
    'GIRAFFEDiscriminator': GIRAFFEDiscriminator,
}


def build_model(model_type, **kwargs):
    """Builds a model based on its class type.

    Args:
        model_type: Class type to which the model belongs, which is case
            sensitive.
        **kwargs: Additional arguments to build the model.

    Raises:
        ValueError: If the `model_type` is not supported.
    """
    if model_type not in _MODELS:
        raise ValueError(f'Invalid model type: `{model_type}`!\n'
                         f'Types allowed: {list(_MODELS)}.')
    return _MODELS[model_type](**kwargs)
