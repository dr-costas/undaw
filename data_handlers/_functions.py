#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ._domain_dataset import DomainDataset
from torch.utils.data import DataLoader

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = ['get_data_loader']


def get_data_loader(for_devices, split, shuffle, drop_last,
                    batch_size, data_path, workers):
    """Returns a data loader object with the specified settings.

    :param for_devices: For which devices.
    :type for_devices: str|list[str]
    :param split: The split to use.
    :type split: str
    :param shuffle: Shall we shuffle the data?
    :type shuffle: bool
    :param drop_last: Shall we drop the last incomplete batch?
    :type drop_last: bool
    :param batch_size: The batch size to use.
    :type batch_size: int
    :param data_path: The data path of the data.
    :type data_path: str
    :param workers: Amount of threads.
    :type workers: int
    :return: The data loader.
    :rtype: torch.utils.data.DataLoader
    """
    return DataLoader(
        dataset=DomainDataset(
            data_device=for_devices, split=split,
            data_path=data_path
        ), batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, num_workers=workers
    )

# EOF
