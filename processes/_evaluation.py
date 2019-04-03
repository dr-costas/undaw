#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

import torch

from helpers import metrics, printing

__author__ = 'Konstantinos Drossos - Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['evaluation']


def evaluation(model, classifier, eval_data, device, return_predictions=False):
    """Evaluates the model on the provided data.

    This function evaluates the adaptation of the adapted \
    model, by performing acoustic scene classification using \
    the adapted model and the target domain data.

    :param model: The model to be evaluated.
    :type model: torch.nn.Module
    :param classifier: The classifier.
    :type classifier: torch.nn.Module
    :param eval_data: The data to use.
    :type eval_data: torch.utils.data.DataLoader
    :param device: The device to use.
    :type device: str
    :param return_predictions: Shall we return predictions?
    :type return_predictions: bool
    :return: None or ground truth and predictions.
    :rtype: None|list[torch.Tensor],list[torch.Tensor]
    """
    model.eval()
    classifier.eval()

    acc_total = []

    start_time = time.time()

    predictions = []
    ground_truth = []

    for data in eval_data:
        x = data[0].float().to(device)
        y_true = data[1].long().argmax(1).to(device)

        h = model(x)
        y_hat = classifier(h)

        acc_total.append(metrics.get_accuracy(y_hat, y_true))

        predictions.append(y_hat.max(-1)[1])
        ground_truth.append(y_true)

    end_time = time.time() - start_time

    printing.print_evaluation_results(torch.Tensor(acc_total).mean(), end_time)

    if return_predictions:
        return ground_truth, predictions

# EOF
