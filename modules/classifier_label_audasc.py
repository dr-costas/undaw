#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.nn import Sequential, Linear, Dropout, ReLU

from ._base_classifier import BaseClassifier

__author__ = 'Konstantinos Drossos - Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['LabelClassifierAUDASC']


class LabelClassifierAUDASC(BaseClassifier):
    """The label classifier.

    This class is adapted from the label classifier \
    for the AUDASC method. The original code can be \
    found at the `online GitHub repo\
    <https://github.com/shayangharib/AUDASC/blob/master/modules/label_classifier.py>`_
    """

    def __init__(self, nb_output_classes):
        """Initialization of the label classifier.

        :param nb_output_classes: The number of classes to classify\
                                 (i.e. amount of outputs).
        :type nb_output_classes: int
        """
        super(LabelClassifierAUDASC, self).__init__()

        self.classifier = Sequential(
            Linear(in_features=1536, out_features=256),
            ReLU(), Dropout(.25),
            Linear(in_features=256, out_features=256),
            ReLU(), Dropout(.25),
            Linear(in_features=256, out_features=nb_output_classes)
        )

# EOF
