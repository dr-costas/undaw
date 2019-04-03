#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.nn import Sequential, Linear, ReLU

from ._base_classifier import BaseClassifier

__author__ = 'Konstantinos Drossos - Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['ClassifierDomain']


class ClassifierDomain(BaseClassifier):
    """The domain classifier/discriminator.

    This class implements the classifier that \
    classifies the domain of the data (i.e. source \
    or target domains).

    The classifier is a two-layered feed-forward neural \
    network, with ReLU non-linearity after the first layer. \
    There is no non-linearity after the last layer.
    """

    def __init__(self, input_dim, nb_outputs):
        """Initialization of the discriminator.

        :param input_dim: The input dimensionality.
        :type input_dim: int
        :param nb_outputs: The amount of outputs.
        :type nb_outputs: int
        """
        super(ClassifierDomain, self).__init__()

        self.classifier = Sequential(
            Linear(in_features=input_dim, out_features=256),
            ReLU(),
            Linear(in_features=256, out_features=nb_outputs)
        )

# EOF
