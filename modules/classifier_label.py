#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.nn import Sequential, Linear, Dropout, ReLU

from ._base_classifier import BaseClassifier

__author__ = 'Konstantinos Drossos - Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['ClassifierLabel']


class ClassifierLabel(BaseClassifier):
    """The label classifier.

    This class implements the classifier that \
    classifies the acoustic scenes.

    The classifier is a three-layered feed-forward neural \
    network, with ReLU non-linearity after the first and \
    second layers. There is no non-linearity after the \
    last layer.
    """

    def __init__(self, input_dim, nb_output_classes):
        """Initialization of the label classifier.

        :param input_dim: Input dimensionality
        :type input_dim: int
        :param nb_output_classes: The amount of output classes.
        :type nb_output_classes: int
        """
        super(ClassifierLabel, self).__init__()

        self.classifier = Sequential(
            Linear(in_features=input_dim, out_features=256),
            ReLU(), Dropout(.25),
            Linear(in_features=256, out_features=256),
            ReLU(), Dropout(.25),
            Linear(in_features=256, out_features=nb_output_classes),
        )

# EOF
