#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import torch

from sklearn.metrics import accuracy_score

__author__ = 'Konstantinos Drossos - Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['get_accuracy']


def get_accuracy(y_hat, y_true):
    """Calculates the accuracy.

    Calculates the accuracy from the `y_hat` predictions\
    and the `y_true` ground truth values.

    :param y_hat: The predictions.
    :type y_hat: torch.Tensor
    :param y_true: The ground truth values.
    :type y_true: torch.Tensor
    :return: The mean accuracy.
    :rtype: float
    """
    y_hat_ready = y_hat.max(-1)[1] if y_hat.ndimension() > 1 \
        else y_hat.ge(0.5).long()

    if y_true.device.type != 'cpu':
        y_true = y_true.cpu()
        y_hat_ready = y_hat_ready.cpu()

    y = y_true.numpy()
    y_hat_ready = y_hat_ready.numpy()

    accuracy = accuracy_score(y, y_hat_ready)

    return accuracy

# EOF
