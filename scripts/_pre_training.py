#!/usr/bin/env python
# -*- coding: utf-8 -*-

from helpers import printing, models, modules_functions
from data_handlers import get_data_loader
from processes import pre_training

__author__ = 'Konstantinos Drossos - Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['do_the_pre_training']


def do_the_pre_training(source_domain_training_data, settings):
    """Performs the pre-training.

    This functions creates/loads the model, creates\
    the data loaders, creates the optimizers, and
    calls :func:`the adaptation process <processes.pre_training>`.

    This function is not used if pre-trained models are\
    used for the method.

    :param source_domain_training_data: The source domain training data.
    :type source_domain_training_data: torch.utils.data.DataLoader
    :param settings: The settings to be used.
    :type settings: dict
    """
    source_model = models.get_asc_model(settings)
    classifier = models.get_label_classifier(settings)

    source_model = source_model.train()
    classifier = classifier.train()

    optimizer_model_source = models.get_optimizer(
        'optimizer_source_asc', [source_model, classifier], settings
    )

    with printing.InformAboutProcess(
            'Creating validation data loader for device: {} '.format(
                ', '.join(settings['data']['source_domain_device'])),
    ):
        source_domain_validation_data = get_data_loader(
            for_devices=settings['data']['source_domain_device'],
            split='validation', shuffle=True, drop_last=True,
            batch_size=settings['data']['batch_size'],
            data_path=settings['data']['data_path'], workers=settings['data']['workers']
        )

    printing.print_msg('Starting pre-training process', start='\n\n-- ')
    model, classifier = pre_training(
        nb_epochs=settings['pre_training']['nb_epochs'],
        training_data=source_domain_training_data,
        validation_data=source_domain_validation_data,
        model=source_model, classifier=classifier,
        optimizer=optimizer_model_source,
        patience=settings['pre_training']['patience'],
        device=settings['general_settings']['device']
    )

    modules_functions.save_model_state(
        settings['models']['base_dir_name'],
        settings['models']['asc_model']['source_model_f_name'],
        model
    )

    modules_functions.save_model_state(
        settings['models']['base_dir_name'],
        settings['models']['label_classifier']['f_name'],
        classifier
    )

# EOF
