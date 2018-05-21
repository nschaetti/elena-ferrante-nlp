#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import torchlanguage.transforms
import torch


#########################################
# Dataset
#########################################


# Load dataset
def load_dataset(root, save_transform):
    """
    Load dataset
    :return:
    """
    # Load from directory
    italian_books_dataset = torchlanguage.datasets.FileDirectory(root=root, save_transform=save_transform)

    # Reuters C50 dataset training
    italian_loader_train = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidation(italian_books_dataset),
        batch_size=1,
        shuffle=True
    )

    # Reuters C50 dataset test
    italian_loader_test = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidation(italian_books_dataset, train=False),
        batch_size=1,
        shuffle=True
    )
    return italian_books_dataset, italian_loader_train, italian_loader_test
# end load_dataset
