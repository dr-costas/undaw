#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import optim

from modules import ModelKaggle, ModelAUDASC, \
    ClassifierDomain, ClassifierLabel, \
    LabelClassifierAUDASC

__author__ = 'Konstantinos Drossos - Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['get_asc_model', 'get_label_classifier',
           'get_domain_classifier', 'get_optimizer']


def get_asc_model(settings):
    """Returns the requested model.

    :param settings: The settings.
    :type settings: dict
    :return: The model.
    :rtype: torch.nn.Module
    """
    model = ModelAUDASC() if settings['models']['asc_model']['pre_trained'] \
        else ModelKaggle()
    return model.to(settings['general_settings']['device'])


def get_label_classifier(settings):
    """Gets the label classifier.

    :param settings: The settings.
    :type settings: dict
    :return: The classifier.
    :rtype: torch.nn.Module
    """
    return _get_classifier(
        classifier_type='labels',
        input_dim=settings['models']['label_classifier']['input_dim'],
        output_classes=settings['models']['label_classifier']['classes'],
        use_pre_trained=settings['models']['asc_model']['pre_trained']).to(
        settings['general_settings']['device']
    )


def get_domain_classifier(settings):
    """Gets the label classifier.

    :param settings: The settings.
    :type settings: dict
    :return: The classifier.
    :rtype: torch.nn.Module
    """
    return _get_classifier(
        classifier_type='labels',
        input_dim=settings['models']['domain_classifier']['input_dim'],
        output_classes=settings['models']['domain_classifier']['classes'],
        use_pre_trained=False).to(
        settings['general_settings']['device']
    )


def get_optimizer(optimizer_type, model, settings):
    """Gets the specified optimizer.

    :param optimizer_type: The type of optimizer (ASC or discriminator).
    :type optimizer_type: str
    :param model: The model(s) to get the parameters from.
    :type model: torch.nn.Module|list[torch.nn.Module]
    :param settings: Keywords for optimizer.
    :type settings: dict|None
    :return: The optimizer.
    :rtype: torch.optim.Optimizer
    """
    if type(model) in [tuple, list]:
        params = []
        for m in model:
            params.extend(list(m.parameters()))
    else:
        params = list(model.parameters())

    kwargs = {'params': params, 'lr': settings['models'][optimizer_type]['lr']}
    if settings['models'][optimizer_type]['keywords'] is not None:
        kwargs.update(settings['models'][optimizer_type]['keywords'])

    if settings['models'][optimizer_type]['type'].lower() == 'rmsprop':
        opt = optim.RMSprop
    elif settings['models'][optimizer_type]['type'].lower() == 'adam':
        opt = optim.Adam
    else:
        raise AttributeError('Unknown optimizer `{}`'.format(
            settings['models'][optimizer_type]['type']))

    return opt(**kwargs)


def _get_classifier(classifier_type, input_dim, output_classes, use_pre_trained):
    """Returns the requested classifier.

    :param classifier_type: The type of the classifier.
    :type classifier_type: str
    :param input_dim: Input dimensionality.
    :type input_dim: int
    :param output_classes: Amount of output classes.
    :type output_classes: int
    :param use_pre_trained: Use the pre-trained AUDASC \
                           label classifier.
    :type use_pre_trained: bool
    :return: The classifier.
    :rtype: torch.nn.Module
    """
    if use_pre_trained:
        return LabelClassifierAUDASC(output_classes)
    if classifier_type.lower() in ['label', 'labels']:
        c = ClassifierLabel
    elif classifier_type.lower() in ['domain', 'domains']:
        c = ClassifierDomain
    else:
        raise AttributeError('Unknown classifier `{}`'.format(classifier_type))

    return c(input_dim, output_classes)

# EOF
