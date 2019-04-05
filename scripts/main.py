#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import path
import yaml

from helpers import printing, argument_parsing
from scripts._pre_training import do_the_pre_training
from scripts._adaptation import do_adaptation
from scripts._evaluation import do_evaluation, do_testing
from scripts._auxiliary import _get_source_training_data_loader

__author__ = 'Konstantinos Drossos - Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['main']


def main():
    """The main entry point for the code.
    """
    arg_parser = argument_parsing.get_argument_parser()
    args = arg_parser.parse_args()

    with open(path.join('settings', '{}.yaml'.format(args.config_file))) as f:
        settings = yaml.load(f)

    printing.print_date_and_time()
    printing.inform_about_device(settings['general_settings']['device'])
    printing.print_msg('', start='')
    printing.print_yaml_settings(settings)

    if settings['process_flow']['do_pre_training'] or \
            settings['process_flow']['do_adaptation']:
        source_domain_training_data = _get_source_training_data_loader(settings)
    else:
        source_domain_training_data = None

    if settings['process_flow']['do_pre_training']:
        do_the_pre_training(source_domain_training_data, settings)

    if settings['process_flow']['do_adaptation']:
        do_adaptation(source_domain_training_data, settings)
        del source_domain_training_data
        printing.print_msg('', start='\n')

    if settings['process_flow']['do_evaluation']:
        do_evaluation(settings)

    if settings['process_flow']['do_testing']:
        do_testing(settings)


if __name__ == '__main__':
    main()

# EOF
