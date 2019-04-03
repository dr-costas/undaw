#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn

__author__ = 'Konstantinos Drossos - Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['BaseClassifier']


class BaseClassifier(nn.Module):
    """The base classifier.

    This class implements just the forward passing \
    of a classifier. It is created for convenience.
    """

    def __init__(self):
        """Stub initialization method.
        """
        super(BaseClassifier, self).__init__()
        raise NotImplementedError(
            'This is the base classifier.'
            'Please implement a classifier, extending this one.'
        )

    def forward(self, x):
        """The forward pass of the label classifier.

        :param x: The input.
        :type x: torch.Tensor
        :return: The prediction of the label classifier.
        :rtype: torch.Tensor
        """
        return self.classifier(x.view(x.size()[0], -1))

# EOF
