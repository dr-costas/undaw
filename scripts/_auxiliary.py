#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data_handlers import get_data_loader
from helpers import printing

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = ['_get_source_training_data_loader']


def _get_source_training_data_loader(settings):
    with printing.InformAboutProcess(
            'Creating training data loader for device: {} '.format(
                ', '.join(settings['data']['source_domain_device'])),
    ):
        return get_data_loader(
            for_devices=settings['data']['source_domain_device'],
            split='training', shuffle=True, drop_last=True,
            batch_size=settings['data']['batch_size'],
            data_path=settings['data']['data_path'], workers=settings['data']['workers']
        )

# EOF
