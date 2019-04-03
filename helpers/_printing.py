#!/usr/bin/env python
# -*- coding: utf-8 -*-

from contextlib import ContextDecorator
from datetime import datetime
import os

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

__author__ = 'Konstantinos Drossos - Tampere University'
__docformat__ = 'reStructuredText'
__all__ = [
    'print_msg', 'inform_about_device', 'print_date_and_time',
    'print_processes_message', 'InformAboutProcess', 'print_yaml_settings',
    'print_pre_training_results', 'print_adaptation_results', 'print_evaluation_results',
    'print_confusion_matrices'
]


_time_f_spec = '5.2'
_acc_f_spec = '6.2'
_loss_f_spec = '7.3'
_epoch_f_spec = '4'


def print_msg(the_msg, start='-- ', end='\n', flush=True):
    """Prints a message.

    :param the_msg: The message.
    :type the_msg: str
    :param start: Starting decoration.
    :type start: str
    :param end: Ending character.
    :type end: str
    :param flush: Flush buffer now?
    :type flush: bool
    """
    print('{}{}'.format(start, the_msg), end=end, flush=flush)


def print_yaml_settings(the_settings):
    """Prints the settings in the YAML settings file.

    :param the_settings: The settings dict
    :type the_settings: dict
    """
    def _print_dict_yaml_settings(the_dict, indentation, start):
        """Prints a nested dict.

        :param the_dict: The nested dict.
        :type the_dict: dict
        :param indentation: Indentation for the printing.
        :type indentation: str
        :param start: Starting decoration.
        :type start: str
        """
        k_l_ = max(*([len(_k) for _k in the_dict.keys()] + [0]))
        for k_, v_ in the_dict.items():
            print_msg('{k_:<{k_l_:d}s}:'.format(k_=k_, k_l_=k_l_),
                      start='{}{}|-- '.format(start, indentation),
                      end=' ', flush=True)
            start = ''
            if type(v_) == dict:
                _print_dict_yaml_settings(v_, '{}{}'.format(indentation, ' ' * 5), '\n')
            elif type(v_) == list:
                print_msg(', '.join(map(str, v_)), start='')
            else:
                print_msg(v_, start='')

    try:
        print_msg('ModelDCASE description: {}'.format(the_settings['model_description']),
                  start='\n-- ')
    except KeyError:
        pass

    print_msg('Settings: ', end='\n\n')

    dict_to_print = {k__: v__ for k__, v__ in the_settings.items() if k__ != 'model_description'}
    k_len = max(*[len(k__) for k__ in dict_to_print.keys()])

    for k, v in dict_to_print.items():
        k_len = max(k_len, len(k))
        print_msg('{}:'.format(k), start=' ' * 2, end=' ')
        if type(v) == dict:
            _print_dict_yaml_settings(v, ' ' * 3, '\n')
        else:
            print_msg(v, start='')
        print_msg('', start='')


def inform_about_device(the_device):
    """Prints an informative message about the device that we are using.

    :param the_device: The device.
    :type the_device: str
    """
    print_msg('Using device: `{}`.'.format(the_device))


def print_date_and_time():
    """Prints the date and time of `now`.
    """
    print_msg(datetime.now().strftime('%Y-%m-%d %H:%M'), start='\n\n-- ')


class InformAboutProcess(ContextDecorator):
    def __init__(self, starting_msg, ending_msg='done', start='-- ', end='\n'):
        """Context manager and decorator for informing about a process.

        :param starting_msg: The starting message, printed before the process starts.
        :type starting_msg: str
        :param ending_msg: The ending message, printed after process ends.
        :type ending_msg: str
        :param start: Starting decorator for the string to be printed.
        :type start: str
        :param end: Ending decorator for the string to be printed.
        :type end: str
        """
        super(InformAboutProcess, self).__init__()
        self.starting_msg = starting_msg
        self.ending_msg = ending_msg
        self.start_dec = start
        self.end_dec = end

    def __enter__(self):
        print_msg('{}... '.format(self.starting_msg), start=self.start_dec, end='')

    def __exit__(self, *exc_type):
        print_msg('{}.'.format(self.ending_msg), start='', end=self.end_dec)
        return False


def print_processes_message(workers):
    """Prints a message for how many processes are used.

    :param workers: The amount of processes.
    :type workers: int
    """
    msg_str = '| Using {} {} |'.format('single' if workers is None else workers,
                                       'processes' if workers > 1 else 'process')
    print_msg('\n'.join(['', '*' * len(msg_str), msg_str, '*' * len(msg_str)]), flush=True)


def print_pre_training_results(epoch, training_loss, validation_loss,
                               training_accuracy, validation_accuracy,
                               time_elapsed):
    """Prints the results of the pre-training step to console.

    :param epoch: The epoch.
    :type epoch: int
    :param training_loss: The loss of the training data.
    :type training_loss: float
    :param validation_loss: The loss of the validation data.
    :type validation_loss: float | None
    :param training_accuracy: The accuracy for the training data.
    :type training_accuracy: float
    :param validation_accuracy: The accuracy for the validation data.
    :type validation_accuracy: float | None
    :param time_elapsed: The time elapsed for the epoch.
    :type time_elapsed: float
    """
    the_msg = \
        'Epoch:{e:{e_spec}d} | ' \
        'Loss (tr/va):{l_tr:{l_f_spec}f}/{l_va:{l_f_spec}f} | ' \
        'Accuracy (tr/va):{acc_tr:{acc_f_spec}f}/{acc_va:{acc_f_spec}f} | ' \
        'Time:{t:{t_f_spec}f}'.format(
            e=epoch,
            l_tr=training_loss, l_va='None' if validation_loss is None else validation_loss,
            acc_tr=training_accuracy, acc_va='None' if validation_accuracy is None else validation_accuracy,
            t=time_elapsed,
            l_f_spec=_loss_f_spec, acc_f_spec=_acc_f_spec, t_f_spec=_time_f_spec,
            e_spec=_epoch_f_spec)

    print_msg(the_msg, start='  -- ')


def print_adaptation_results(epoch, labels_loss, mapping_loss,
                             discriminator_loss, d_d, d_g,
                             time_elapsed, g_p=None):
    """Prints the output of the adaptation process.

    :param epoch: The epoch.
    :type epoch: int
    :param labels_loss: The loss of the labels.
    :type labels_loss: float
    :param mapping_loss: The loss of the mappings.
    :type mapping_loss: float
    :param discriminator_loss: The total loss of the domain discriminator.
    :type discriminator_loss: float
    :param d_d: The loss of the domain discriminator.
    :type d_d: float
    :param d_g: The loss of the target model mappings.
    :type d_g: float
    :param g_p: The gradient penalty.
    :type g_p: float|None
    :param time_elapsed: The elapsed time for the epoch.
    :type time_elapsed: float
    """
    if g_p is None:
        dsc_str = 'Discriminator (total/D/G):' \
                  '{d_t:{acc_f_spec}f}/{d_d:{acc_f_spec}f}/{d_g:{acc_f_spec}f} | '.format(
                    d_t=discriminator_loss, d_d=d_d, d_g=d_g,
                    l_f_spec=_loss_f_spec, acc_f_spec=_loss_f_spec
        )
    else:
        dsc_str = 'Discriminator (total/D/G/GP):' \
                  '{d_t:{acc_f_spec}f}/{d_d:{acc_f_spec}f}/{d_g:{acc_f_spec}f}/{g_p:{acc_f_spec}f} | '.format(
                    d_t=discriminator_loss, d_d=d_d, d_g=d_g,
                    l_f_spec=_loss_f_spec, acc_f_spec=_loss_f_spec, g_p=g_p
        )
    the_msg = \
        'Epoch:{e:{e_spec}d} | ' \
        'Labels:{l:{l_f_spec}f} | ' \
        'Mappings:{m:{l_f_spec}f} | ' \
        '{d_str}' \
        'Time:{t:{t_f_spec}f}'.format(
            e=epoch,
            l=labels_loss,
            m=mapping_loss,
            d_str=dsc_str,
            t=time_elapsed,
            l_f_spec=_loss_f_spec, acc_f_spec=_loss_f_spec, t_f_spec=_time_f_spec,
            e_spec=_epoch_f_spec
        )

    print_msg(the_msg, start='  -- ')


def print_evaluation_results(accuracy, time_elapsed):
    """Prints the output of the adaptation process.

    :param accuracy: The accuracy.
    :type accuracy: float
    :param time_elapsed: The elapsed time for the epoch.
    :type time_elapsed: float
    """
    the_msg = 'Accuracy:{acc:{acc_f_spec}f} | Time:{t:{t_f_spec}f}'.format(
            acc=accuracy, t=time_elapsed,
            acc_f_spec=_acc_f_spec, t_f_spec=_time_f_spec)

    print_msg(the_msg, start='  -- ')


def print_confusion_matrices(predictions_non_adapted_source,
                             predictions_adapted_source,
                             predictions_non_adapted_target,
                             predictions_adapted_target,
                             scene_labels, confusion_matrices_settings):
    """Prints the confusion matrices.

    :param predictions_non_adapted_source: The predictions of the non-adapted model\
                                           on the source domain.
    :type predictions_non_adapted_source: torch.Tensor
    :param predictions_adapted_source: The predictions of the adapted model\
                                       on the source domain.
    :type predictions_adapted_source: torch.Tensor
    :param predictions_non_adapted_target: The predictions of the non-adapted model\
                                           on the target domain.
    :type predictions_non_adapted_target: torch.Tensor
    :param predictions_adapted_target: The predictions of the adapted model\
                                       on the target domain.
    :type predictions_adapted_target: torch.Tensor
    :param scene_labels: The labels for each acoustic scene.
    :type scene_labels: list[str]
    :param confusion_matrices_settings: The settings for the confusion matrices.
    :type confusion_matrices_settings: dict
    """

    def _print_cm(y_true, y_pred, labels, filename, dpi, include_colorbar,
                  file_format, color_scheme, tight_axes):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float)

        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape

        for i in range(nrows):
            for j in range(ncols):
                annot[i, j] = '{:.2f}'.format(cm_perc[i, j])

        cm = pd.DataFrame(cm, index=labels, columns=labels)
        cm.index.name = 'True label'
        cm.columns.name = 'Predicted label'
        fig, ax = plt.subplots()
        fmt = lambda x, pos: '{:.2f}'.format(int(x)/float(np.max(cm_sum)))
        seaborn.heatmap(cm, annot=annot, fmt='', ax=ax, cmap=color_scheme,
                        cbar_kws={'format': FuncFormatter(fmt)},
                        cbar=include_colorbar)
        for tick in ax.get_xticklabels():
            tick.set_rotation(55)

        plt.savefig(filename, format=file_format, dpi=dpi,
                    bbox_inches='tight' if tight_axes else None)

    scene_labels = sorted(scene_labels)

    data = [
        predictions_non_adapted_source,
        predictions_adapted_source,
        predictions_non_adapted_target,
        predictions_adapted_target
    ]

    if not os.path.exists(confusion_matrices_settings['save_path']):
        os.makedirs(confusion_matrices_settings['save_path'])

    s_p = os.path.join(confusion_matrices_settings['save_path'], '{}')

    file_names = [
        s_p.format('non_adapted_source_domain.eps'),
        s_p.format('adapted_source_domain.eps'),
        s_p.format('non_adapted_target_domain.eps'),
        s_p.format('adapted_target_domain.eps'),
    ]

    cm_kwagrs = {
        'dpi': confusion_matrices_settings['dpi'],
        'include_colorbar': confusion_matrices_settings['include_colorbar'],
        'file_format': confusion_matrices_settings['file_format'],
        'color_scheme': confusion_matrices_settings['color_scheme'],
        'tight_axes': confusion_matrices_settings['tight_axes']
    }

    for d, f in zip(data, file_names):
        y_true_non_adapted_source = [scene_labels[i] for ii in d[0] for i in ii]
        y_pred_non_adapted_source = [scene_labels[i] for ii in d[1] for i in ii]
        _print_cm(y_true_non_adapted_source, y_pred_non_adapted_source,
                  scene_labels, f, **cm_kwagrs)

# EOF
