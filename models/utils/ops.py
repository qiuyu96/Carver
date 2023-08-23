# python3.8
"""Contains some utility operators."""

import math
import torch
import torch.distributed as dist
import torch.nn.functional as F

__all__ = ['all_gather', 'upsample', 'downsample']


def all_gather(tensor):
    """Gathers tensor from all devices and executes averaging."""
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    tensor_list = [torch.ones_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor, async_op=False)
    return torch.stack(tensor_list, dim=0).mean(dim=0)


def upsample(img_nerf, size, filter=None):
    up = size // img_nerf.size(-1)
    if up <= 1:
        return img_nerf

    if filter is not None:
        from third_party.stylegan2_official_ops import upfirdn2d
        for _ in range(int(math.log2(up))):
            img_nerf = upfirdn2d.downsample2d(img_nerf, filter, up=2)
    else:
        img_nerf = F.interpolate(img_nerf, (size, size),
                                 mode='bilinear',
                                 align_corners=False)
    return img_nerf


def downsample(img0, size, filter=None):
    down = img0.size(-1) // size
    if down <= 1:
        return img0

    if filter is not None:
        from third_party.stylegan2_official_ops import upfirdn2d
        for _ in range(int(math.log2(down))):
            img0 = upfirdn2d.downsample2d(img0, filter, down=2)
    else:
        img0 = F.interpolate(img0, (size, size),
                             mode='bilinear',
                             align_corners=False)
    return img0