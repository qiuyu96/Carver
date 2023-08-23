# python3.8
"""Contains the implementation of discriminator described in GIRAFFE.

Paper: https://arxiv.org/pdf/2011.12100.pdf

Official PyTorch implementation: https://github.com/autonomousvision/giraffe
"""

import math
import torch.nn as nn


class GIRAFFEDiscriminator(nn.Module):

    def __init__(self, in_dim=3, n_feat=512, img_size=64):
        super(GIRAFFEDiscriminator, self).__init__()

        self.in_dim = in_dim
        n_layers = int(math.log2(img_size) - 2)
        self.blocks = nn.ModuleList([
            nn.Conv2d(
                in_dim, int(n_feat / (2**(n_layers - 1))), 4, 2, 1, bias=False)
        ] + [
            nn.Conv2d(int(n_feat / (2**(n_layers - i))),
                      int(n_feat / (2**(n_layers - 1 - i))),
                      4,
                      2,
                      1,
                      bias=False) for i in range(1, n_layers)
        ])

        self.conv_out = nn.Conv2d(n_feat, 1, 4, 1, 0, bias=False)
        self.actvn = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, **kwargs):
        batch_size = x.shape[0]
        if x.shape[1] != self.in_dim:
            x = x[:, :self.in_dim]
        for layer in self.blocks:
            x = self.actvn(layer(x))

        out = self.conv_out(x)
        out = out.reshape(batch_size, 1)

        return {'score': out}
