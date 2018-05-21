#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : core.classifiers.RCNLPTextClassifier.py
# Description : Echo State Network for text classification.
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 01.02.2017 17:59:05
# Lieu : Nyon, Suisse
#
# This file is part of the Reservoir Computing NLP Project.
# The Reservoir Computing Memory Project is a set of free software:
# you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Foobar is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#

import numpy as np
import torch.utils.data
from torch.autograd import Variable
import echotorch.nn as etnn
import echotorch.utils
import torchlanguage.transforms
from tools import argument_parsing, dataset, features, functions
import argparse


# Arguments
parser = argparse.ArgumentParser("Authorship Attribution on Elena Ferrante")
parser.add_argument("--root", type=str, default="./data")
parser.add_argument("--reservoir-size", type=int, default=1000)
parser.add_argument("--leaky-rate", type=float, default=0.01)
parser.add_argument("--spectral-radius", type=float, default=0.99)
parser.add_argument("--input-sparsity", type=float, default=0.1)
parser.add_argument("--input-scaling", type=float, default=0.5)
parser.add_argument("--w-sparsity", type=float, default=0.1)
parser.add_argument("--transformer", type=str, required=True)
parser.add_argument("--embedding-path", type=str, default="")
parser.add_argument("--n-gram", type=int, default=1)
parser.add_argument("--lang-model", type=str, default="en_vectors_web_lg")
parser.add_argument("--no-cuda", action='store_true', default=False, help="Enables CUDA training")
parser.add_argument("--save-transform", action='store_true', default=False)
args = parser.parse_args()

# Use CUDA?
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Load from directory
italian_book_dataset, italian_loader_train, italian_loader_test = dataset.load_dataset(args.root, args.save_transform)

# Print authors
print(u"Authors : {}".format(italian_book_dataset.classes))
print(u"{} documents".format(len(italian_book_dataset.samples)))
n_classes = italian_book_dataset.n_classes

# Log
print(u"Creating transformer...")

# Transformer
if args.transformer == "wv":
    # Where to find OOV
    embedding_transformer_pos = 1

    # Tranformer
    italian_book_dataset.transform = torchlanguage.transforms.Compose([
        torchlanguage.transforms.Token(model=args.lang_model),
        torchlanguage.transforms.GensimModel(
            model_path=args.embedding_path
        )
    ])

    # Input dim
    transformer_output_dim = 300
else:
    # Where to find OOV
    embedding_transformer_pos = 2

    # N-gram
    if args.n_gram == 1:
        char_transformer = torchlanguage.transforms.Character()
    elif args.n_gram == 2:
        char_transformer = torchlanguage.transforms.Character2Gram(overlapse=False)
    else:
        char_transformer = torchlanguage.transforms.Character3Gram(overlapse=False)
    # end if

    # Load embeddings
    token_to_ix, embedding_weights = features.load_character_embedding(args.path)
    embedding_dim = embedding_weights.size(1)

    # Transformer
    italian_book_dataset.transform = torchlanguage.transforms.Compose([
        char_transformer,
        torchlanguage.transforms.ToIndex(token_to_ix=token_to_ix),
        torchlanguage.transforms.Embedding(weights=embedding_weights, voc_size=len(token_to_ix)),
        torchlanguage.transforms.Reshape((-1, embedding_dim))
    ])

    # Input dim
    transformer_output_dim = embedding_dim
# end if

# Log
print(u"Creating ESN...")

# Echo State Network
esn = etnn.LiESN(
    input_dim=transformer_output_dim,
    hidden_dim=args.reservoir_size,
    output_dim=italian_book_dataset.n_classes,
    spectral_radius=args.spectral_radius,
    sparsity=args.input_sparsity,
    input_scaling=args.input_scaling,
    w_sparsity=args.w_sparsity,
    learning_algo='inv',
    leaky_rate=args.leaky_rate
)
if args.cuda:
    esn.cuda()
# end if

# Class to ID
class_to_ix = dict()
ix_to_class = dict()
for index, author in enumerate(italian_book_dataset.classes):
    class_to_ix[author] = index
    ix_to_class[index] = author
# end for

# Average
average_k_fold = np.zeros(10)

# OOV
training_oov = np.zeros(10)
test_oov = np.zeros(10)

# For each batch
for k in range(10):
    # Log
    print(u"Fold {}".format(k))

    # Choose fold
    italian_loader_train.dataset.set_fold(k)
    italian_loader_test.dataset.set_fold(k)

    # Counter oov
    oov = 0.0
    count = 0.0

    # Get training data for this fold
    for i, data in enumerate(italian_loader_train):
        # Inputs and labels
        inputs, labels, title = data

        # Input dim
        inputs = inputs.view(1, -1, transformer_output_dim)

        # Time labels
        time_labels = functions.to_timelabels(inputs.size(1), class_to_ix[labels[0]], n_classes)

        # To variable
        inputs, time_labels = Variable(inputs), Variable(time_labels)
        if args.cuda: inputs, time_labels = inputs.cuda(), time_labels.cuda()

        # Accumulate xTx and xTy
        esn(inputs, time_labels)

        # Count
        count += 1.0

        # OOV
        oov += italian_book_dataset.transform.transforms[embedding_transformer_pos].oov
    # end for
    training_oov[k] = oov / count

    # Finalize training

    esn.finalize()

    # Counters
    successes = 0.0
    count = 0.0
    oov = 0.0

    # Get test data for this fold
    for i, data in enumerate(italian_loader_test):
        # Inputs and labels
        inputs, labels, title = data

        # Input dim
        inputs = inputs.view(1, -1, transformer_output_dim)

        # Time labels
        time_labels = functions.to_timelabels(inputs.size(1), class_to_ix[labels[0]], n_classes)

        # To variable
        inputs, time_labels = Variable(inputs), Variable(time_labels)
        if args.cuda: inputs, time_labels = inputs.cuda(), time_labels.cuda()

        # Predict
        y_predicted = esn(inputs)

        # Normalized
        y_predicted -= torch.min(y_predicted)
        y_predicted /= torch.max(y_predicted) - torch.min(y_predicted)

        # Sum to one
        sums = torch.sum(y_predicted, dim=2)
        for t in range(y_predicted.size(1)):
            y_predicted[0, t, :] = y_predicted[0, t, :] / sums[0, t]
        # end for

        # Normalized
        y_predicted = echotorch.utils.max_average_through_time(y_predicted, dim=1)

        # Compare
        if torch.equal(y_predicted, torch.FloatTensor([class_to_ix[labels[0]]])):
            successes += 1.0
        # end if

        # Count
        count += 1.0

        # OOV
        oov += italian_book_dataset.transform.transformers[embedding_transformer_pos].oov
    # end for

    # Success rate
    average_k_fold[k] = successes / count * 100.0

    # OOV
    test_oov[k] = oov / count

    # Reset learning
    esn.reset()
# end for

# Print result
print(u"10-Fold cross validation accuracy : {}".format(np.average(average_k_fold)))

# Print OOV
print(u"OOV : {}/{}".format(np.average(training_oov), np.average(test_oov)))
