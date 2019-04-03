#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

import torch
from torch.nn.functional import cross_entropy

from helpers import printing

__author__ = 'Konstantinos Drossos - Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['adaptation']


def adaptation(epochs, source_model, target_model, classifier, discriminator,
               source_data, target_data, optimizer_target, optimizer_discriminator,
               device, labels_loss_w=1., first_iter=0, n_critic=5):
    """The adaptation of the model.

    This function implements the adaptation process, based on the \
    WGAN algorithm and the extra loss term, added at the AUDASC method.

    The WGAN paper can be found online at `arXiv <https://arxiv.org/abs/1701.07875>`_.

    The AUDASC paper can be found online at `arXiv <https://arxiv.org/abs/1808.05777>`_.

    :param epochs: The amount of epochs.
    :type epochs: int
    :param source_model: The source model.
    :type source_model: torch.nn.Module
    :param target_model: The target model.
    :type target_model: torch.nn.Module
    :param classifier: The classifier.
    :type classifier: torch.nn.Module
    :param discriminator: The discriminator.
    :type discriminator: torch.nn.Module
    :param source_data: The source domain data.
    :type source_data: torch.utils.data.DataLoader
    :param target_data: The target domain data.
    :type target_data: torch.utils.data.DataLoader
    :param optimizer_target: The optimizer for the target model.
    :type optimizer_target: torch.optim.Optimizer
    :param optimizer_discriminator: The optimizer for the discriminator.
    :type optimizer_discriminator: torch.optim.Optimizer
    :param device: The device that we use.
    :type device: str
    :param labels_loss_w: Weighting for the loss of the labels.
    :type labels_loss_w: float
    :param first_iter: Amount of head start iterations for the discriminator.
    :type first_iter: int
    :param n_critic: Consecutive iterations for the critic
    :type n_critic: int
    :return: The optimized target model.
    :rtype: torch.nn.Module
    """
    cntr = 0
    best_adapted_model = dict()

    for epoch in range(epochs):
        epoch_labels_loss = []
        epoch_mappings_loss = []
        epoch_d_loss = []
        d_d = []
        d_g = []

        start_time = time.time()

        target_data_it = target_data.__iter__()

        for source_examples in source_data:
            try:
                target_examples = next(target_data_it)
            except StopIteration:
                target_data_it = target_data.__iter__()
                target_examples = next(target_data_it)

            source_x = source_examples[0].float().to(device)
            source_y = source_examples[1].long().argmax(1).to(device)

            target_x = target_examples[0].float().to(device)

            source_b_size = source_x.size()[0]

            h_target_m = target_model(torch.cat([source_x, target_x], dim=0))
            h_source_m = source_model(source_x).detach()

            labels_prediction = classifier(h_target_m[:source_b_size])

            for param in discriminator.parameters():
                param.data.clamp_(-.01, .01)

            domain_prediction = discriminator(torch.cat(
                [h_source_m, h_target_m[source_b_size:]],
                dim=0
            ))

            labels_loss = cross_entropy(labels_prediction, source_y).mul(labels_loss_w)

            domain_d_loss = domain_prediction[:source_b_size].mean().sub(
                domain_prediction[source_b_size:].mean()
            ).neg()

            domain_target_m_loss = domain_prediction[source_b_size:].mean().neg()

            target_m_loss = labels_loss + domain_target_m_loss

            if cntr == n_critic - 1:
                optimizer_target.zero_grad()
                target_m_loss.backward()
                optimizer_target.step()
                epoch_mappings_loss.append(domain_target_m_loss.item())
                epoch_labels_loss.append(labels_loss.item())
                cntr = 0
            else:
                if first_iter != 0:
                    first_iter -= 1
                else:
                    cntr += 1
                optimizer_discriminator.zero_grad()
                domain_d_loss.backward()
                optimizer_discriminator.step()
                epoch_d_loss.append(domain_d_loss.item())

            d_d.append(domain_prediction[:source_b_size].mean())
            d_g.append(domain_prediction[source_b_size:].mean())

        end_time = time.time() - start_time

        printing.print_adaptation_results(
            epoch, torch.Tensor(epoch_labels_loss).mean(),
            torch.Tensor(epoch_mappings_loss).mean(),
            torch.Tensor(epoch_d_loss).mean(),
            torch.Tensor(d_d).mean(),
            torch.Tensor(d_g).mean(),
            end_time
        )

    if best_adapted_model is not None:
        target_model.load_state_dict(best_adapted_model)

    return target_model

# EOF
