#!/usr/bin/env python
# -*- coding: utf-8 -*-

from helpers import printing, models, modules_functions
from data_handlers import get_data_loader
from processes import adaptation

__author__ = 'Konstantinos Drossos - Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['do_adaptation']


def do_adaptation(source_domain_training_data, settings):
    """Performs the adaptation.

    This functions creates/loads the model, creates\
    the data loaders, creates the optimizers, and
    calls :func:`the adaptation process <processes.adaptation>`.

    :param source_domain_training_data: The source domain data.
    :type source_domain_training_data: torch.utils.data.DataLoader
    :param settings: The settings to be used.
    :type settings: dict
    :return: The adapted model.
    :rtype: torch.nn.Module
    """
    with printing.InformAboutProcess(
            'Creating training data loader for device: {} '.format(
                ', '.join(settings['data']['target_domain_device'])),
    ):
        target_domain_training_data = get_data_loader(
            for_devices=settings['data']['target_domain_device'],
            split='training', shuffle=True, drop_last=True,
            batch_size=settings['data']['batch_size'],
            data_path=settings['data']['data_path'],
            workers=settings['data']['workers']
        )

    source_model = models.get_asc_model(settings)
    classifier = models.get_label_classifier(settings)

    source_model = modules_functions.load_model_state(
        settings['models']['base_dir_name'],
        settings['models']['asc_model']['source_model_f_name'],
        source_model
    ).to(settings['general_settings']['device'])

    classifier = modules_functions.load_model_state(
        settings['models']['base_dir_name'],
        settings['models']['label_classifier']['f_name'],
        classifier
    ).to(settings['general_settings']['device'])

    target_model = models.get_asc_model(settings)

    target_model = modules_functions.load_model_state(
        settings['models']['base_dir_name'],
        settings['models']['asc_model']['source_model_f_name'],
        target_model
    ).to(settings['general_settings']['device'])

    discriminator = models.get_domain_classifier(settings)

    source_model = source_model.eval()
    classifier = classifier.eval()

    target_model = target_model.train()
    discriminator = discriminator.train()

    optimizer_model_target = models.get_optimizer(
        'optimizer_target_asc', target_model, settings
    )

    optimizer_discriminator = models.get_optimizer(
        'optimizer_discriminator', discriminator, settings
    )

    printing.print_msg('Starting adaptation process.', start='\n\n-- ')

    target_model = adaptation(
        epochs=settings['adaptation']['nb_epochs'],
        source_model=source_model, target_model=target_model,
        classifier=classifier, discriminator=discriminator,
        source_data=source_domain_training_data,
        target_data=target_domain_training_data,
        optimizer_target=optimizer_model_target,
        optimizer_discriminator=optimizer_discriminator,
        device=settings['general_settings']['device'],
        labels_loss_w=settings['adaptation']['labels_loss_w'],
        first_iter=settings['adaptation']['first_iter'],
        n_critic=settings['adaptation']['n_critic']
    )

    modules_functions.save_model_state(
        settings['models']['base_dir_name'],
        settings['models']['asc_model']['target_model_f_name'],
        target_model
    )

    target_model = target_model.eval()

    return target_model

# EOF
