#!/usr/bin/env python
# -*- coding: utf-8 -*-

from helpers import printing, models, modules_functions
from data_handlers import get_data_loader
from processes import evaluation

__author__ = 'Konstantinos Drossos - Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['do_evaluation', 'do_testing']

_info_msg = 'Starting {} process of {} model on {} domain.'


def _get_models_and_data(settings, is_testing):
    """Retrieves the models and the data to be used.

    This function retrieves the models and the data\
    to be used for the evaluation process, depending\
    on whether there is a validation or a testing case.

    :param settings: The settings to be used.
    :type settings: dict
    :param is_testing: Are we doing testing?
    :type is_testing: bool
    :return: The models and the data.
    :rtype: torch.nn.Module, torch.nn.Module, torch.nn.Module, \
            torch.utils.data.DataLoader, torch.utils.data.DataLoader
    """
    source_model = models.get_asc_model(settings)
    classifier = models.get_label_classifier(settings)
    target_model = models.get_asc_model(settings)

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

    target_model = modules_functions.load_model_state(
        settings['models']['base_dir_name'],
        settings['models']['asc_model']['target_model_f_name'],
        target_model
    ).to(settings['general_settings']['device'])

    source_model = source_model.eval()
    target_model = target_model.eval()
    classifier = classifier.eval()

    with printing.InformAboutProcess(
            'Creating training data loader for device: {} '.format(
                ', '.join(settings['data']['source_domain_device'])),
    ):
        s_d_v_data = get_data_loader(
            for_devices=settings['data']['source_domain_device'],
            split='validation' if not is_testing else 'test',
            shuffle=True if not is_testing else False, drop_last=True,
            batch_size=settings['data']['batch_size'],
            data_path=settings['data']['data_path'], workers=settings['data']['workers']
        )

    with printing.InformAboutProcess(
            'Creating training data loader for device: {} '.format(
                ', '.join(settings['data']['target_domain_device'])),
    ):
        t_d_v_data = get_data_loader(
            for_devices=settings['data']['target_domain_device'],
            split='validation' if not is_testing else 'test',
            shuffle=True if not is_testing else False, drop_last=True,
            batch_size=settings['data']['batch_size'],
            data_path=settings['data']['data_path'], workers=settings['data']['workers']
        )

    return source_model, target_model, classifier, s_d_v_data, t_d_v_data


def do_evaluation(settings):
    """Performs the evaluation.

    This functions creates/loads the model, creates\
    the data loaders, creates the optimizers, and
    calls :func:`the evaluation process <processes.evaluation>`.

    The results are printed on the stdout.

    :param settings: The settings to be used.
    :type settings: dict
    """
    source_m, target_m, classifier, data_loader_s, data_loader_t = _get_models_and_data(
        settings, is_testing=False)

    kwargs_s = {
        'classifier': classifier,
        'eval_data': data_loader_s,
        'device': settings['general_settings']['device']
    }

    kwargs_t = {
        'classifier': classifier,
        'eval_data': data_loader_t,
        'device': settings['general_settings']['device']
    }

    printing.print_msg(_info_msg.format('evaluation', 'source', 'source'), start='\n\n-- ')
    evaluation(model=source_m, **kwargs_s)

    printing.print_msg(_info_msg.format('evaluation', 'target', 'source'), start='\n\n-- ')
    evaluation(model=target_m, **kwargs_s)

    printing.print_msg(_info_msg.format('evaluation', 'source', 'target'), start='\n\n-- ')
    evaluation(model=source_m, **kwargs_t)

    printing.print_msg(_info_msg.format('evaluation', 'target', 'target'), start='\n\n-- ')
    evaluation(model=target_m, **kwargs_t)
    printing.print_msg('', start='\n')


def do_testing(settings):
    """Performs the testing.

    This functions creates/loads the model, creates\
    the data loaders, creates the optimizers, and
    calls :func:`the evaluation process <processes.evaluation>`.

    The results are printed on the stdout.

    :param settings: The settings to be used.
    :type settings: dict
    """
    source_m, target_m, classifier, data_loader_s, data_loader_t = _get_models_and_data(
        settings, is_testing=True)

    kwargs_s = {
        'classifier': classifier,
        'eval_data': data_loader_s,
        'device': settings['general_settings']['device'],
        'return_predictions': settings['aux_settings']['confusion_matrices']['print_them']
    }

    kwargs_t = {
        'classifier': classifier,
        'eval_data': data_loader_t,
        'device': settings['general_settings']['device'],
        'return_predictions': settings['aux_settings']['confusion_matrices']['print_them']
    }

    scene_labels = [
        'airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
        'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram'
    ]

    printing.print_msg(_info_msg.format('testing', 'source', 'source'), start='\n\n-- ')
    predictions_non_adapted_source = evaluation(model=source_m, **kwargs_s)

    printing.print_msg(_info_msg.format('testing', 'target', 'source'), start='\n\n-- ')
    predictions_adapted_source = evaluation(model=target_m, **kwargs_s)

    printing.print_msg(_info_msg.format('testing', 'source', 'target'), start='\n\n-- ')
    predictions_non_adapted_target = evaluation(model=source_m, **kwargs_t)

    printing.print_msg(_info_msg.format('testing', 'target', 'target'), start='\n\n-- ')
    predictions_adapted_target = evaluation(model=target_m, **kwargs_t)

    if settings['aux_settings']['confusion_matrices']['print_them']:
        with printing.InformAboutProcess('Creating confusion matrices figures', start='\n\n--'):
            printing.print_confusion_matrices(
                predictions_non_adapted_source, predictions_adapted_source,
                predictions_non_adapted_target, predictions_adapted_target,
                scene_labels, settings['aux_settings']['confusion_matrices']
            )

    printing.print_msg('', start='\n')

# EOF
