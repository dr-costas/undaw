#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import path
import pickle

import torch
from torch.utils.data import Dataset

__author__ = 'Konstantinos Drossos - Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['DomainDataset']


class DomainDataset(Dataset):
    """The dataset for one domain.

    This class extends the :class:`torch.utils.data.Dataset` \
    class, and represents the dataset for one domain (i.e. \
    source or target domain).
    """

    def __init__(self, data_device, split, data_path):
        """Initializes the dataset for a particular split and\
        one or more devices.

        :param data_device: The device(s) to be used.
        :type data_device: str | list[str]
        :param split: The split to be used.
        :type split: str
        :param data_path: The path of the data files.
        :type data_path: str
        """
        super(DomainDataset, self).__init__()

        if type(data_device) not in [tuple, list]:
            data_device = [data_device]

        if split in ['training', 'train']:
            split_str = 'training'
        elif split in ['validation', 'valid']:
            split_str = 'validation'
        else:
            split_str = 'test'

        features_file_name = '{}_features.p'.format(split_str)
        labels_file_name = '{}_scene_labels.p'.format(split_str)

        with open(path.join(data_path, features_file_name), 'rb') as f:
            self.features = pickle.load(f)

        with open(path.join(data_path, labels_file_name), 'rb') as f:
            self.labels = pickle.load(f)

        self.features = torch.cat(
            [self.features[k] for k in self.features.keys() if k in data_device],
            dim=0
        )

        self.labels = torch.cat(
            [self.labels[k] for k in self.labels.keys() if k in data_device],
            dim=0
        )

    def __len__(self):
        """The length of the dataset.

        :return: The length of the dataset.
        :rtype: int
        """
        return self.features.size()[0]

    def __getitem__(self, item):
        """The select example from the dataset.

        :param item: The index of the example.
        :type item: int
        :return: The X and Y for the example.
        :rtype: torch.Tensor, torch.Tensor
        """
        return self.features[item], self.labels[item]

# EOF
