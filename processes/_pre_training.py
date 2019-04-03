#!/usr/bin/env python
# -*- coding: utf-8 -*-

from time import time

import numpy as np
from torch.nn import functional

from helpers import printing, metrics

__author__ = 'Konstantinos Drossos - Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['pre_training']


def pre_training(nb_epochs, training_data, validation_data, model, classifier,
                 optimizer, patience, device):
    """The pre-training of the model.

    This function pre-trains the acoustic scene classification \
    model, using a typical supervised learning scenario for \
    acoustic scene classification.

    In the case that a pre-trained model is used, this function \
    is not used.

    :param nb_epochs: The amount of max epochs.
    :type nb_epochs: int
    :param training_data: The training data.
    :type training_data: torch.utils.data.DataLoader
    :param validation_data: The validation data.
    :type validation_data: torch.utils.data.DataLoader
    :param model: The model to use.
    :type model: torch.nn.Module
    :param classifier: The classifier to use.
    :type classifier: torch.nn.Module
    :param optimizer: The optimizer for the model
    :type optimizer: torch.optim.Optimizer
    :param patience: The amount of epochs for validation patience.
    :type patience: int
    :param device: The device to use.
    :type device: str
    :return: The optimized model and the optimized classifier.
    :rtype: torch.nn.Module, torch.nn.Module
    """
    best_val_acc = -1
    patience_cntr = 0

    for epoch in range(nb_epochs):
        start_time = time()

        model = model.train()
        classifier = classifier.train()

        model, classifier, tr_loss, tr_acc = _training(
            training_data, model, classifier, optimizer, device)

        va_loss, va_acc = None, None

        if validation_data is not None:
            model = model.eval()
            classifier = classifier.eval()
            va_loss, va_acc = _validation(validation_data, model, classifier, device)

            if best_val_acc < va_acc:
                best_val_acc = va_acc
                patience_cntr = 0
            else:
                patience_cntr += 1

        end_time = time() - start_time

        printing.print_pre_training_results(
            epoch=epoch, training_loss=tr_loss, validation_loss=va_loss,
            training_accuracy=tr_acc, validation_accuracy=va_acc,
            time_elapsed=end_time
        )

        if patience_cntr > patience > 0:
            break

    printing.print_msg('', start='', end='\n\n')

    return model, classifier


def _training(the_data, model, classifier, optimizer, device):
    """One epoch of training.

    :param the_data: The data.
    :type the_data: torch.utils.data.DataLoader
    :param model: The model to use.
    :type model: torch.nn.Module
    :param classifier: The classifier to use.
    :type classifier: torch.nn.Module
    :param optimizer: The optimizer for the model.
    :type optimizer: torch.optim.Optimizer
    :param device: The device to use.
    :type device: str
    :return: The optimized model, the optimized classifier, \
             the loss, and the accuracy for one epoch.
    :rtype: torch.nn.Module, torch.nn.Module, float, float
    """
    epoch_loss = []
    epoch_accuracy = []

    for data in the_data:
        x = data[0].float().to(device)
        y_true = data[1].long().argmax(1).to(device)

        h = model(x)
        y_hat = classifier(h)

        loss = functional.cross_entropy(y_hat, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        epoch_accuracy.append(metrics.get_accuracy(functional.softmax(y_hat, dim=1), y_true))

    return model, classifier, np.mean(epoch_loss), np.mean(epoch_accuracy)


def _validation(the_data, model, classifier, device):
    """The validation for one epoch.

    :param the_data: The data.
    :type the_data: torch.utils.data.DataLoader
    :param model: The model to use.
    :type model: torch.nn.Module
    :param classifier: The classifier to use.
    :type classifier: torch.nn.Module
    :param device: The device to use.
    :type device: str
    :return: The loss and the accuracy for the validation data.
    :rtype: float, float
    """
    epoch_loss = []
    epoch_accuracy = []

    for data in the_data:
        x = data[0].float().to(device)
        y_true = data[1].long().argmax(1).to(device)

        h = model(x)
        y_hat = classifier(h)

        epoch_loss.append(functional.cross_entropy(y_hat, y_true).item())
        epoch_accuracy.append(metrics.get_accuracy(functional.softmax(y_hat, dim=1), y_true))

    return np.mean(epoch_loss), np.mean(epoch_accuracy)


# EOF
