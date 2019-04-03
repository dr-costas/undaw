#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn

__author__ = 'Konstantinos Drossos - Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['ModelKaggle']


class ModelKaggle(nn.Module):
    """The Kaggle model.

    This class implements the Kaggle model and it is\
    a refactored version of the code used for the AUDASC \
    method. The original code can be found at the `online
    GitHub repo of the AUDASC method \
    <https://github.com/shayangharib/AUDASC/blob/master/modules/model.py>`_.
    """

    def __init__(self):
        """Initialization of the model.
        """
        super(ModelKaggle, self).__init__()

        block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=48,
                      kernel_size=11, stride=(2, 3), padding=5),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2))
        )

        block_2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=128,
                      kernel_size=5, stride=(2, 3), padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        block_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=192,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        block_4 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        block_5 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128,
                      kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2))
        )

        self.model = nn.Sequential(
            block_1, block_2, block_3,
            block_4, block_5
        )

    def forward(self, x):
        """Forward pass.

        :param x: The input.
        :type x: torch.Tensor
        :return: The output of the model.
        :rtype: torch.Tensor
        """
        return self.model(x)

# EOF
