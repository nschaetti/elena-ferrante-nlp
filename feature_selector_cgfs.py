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
import torch.nn as nn
import argparse
from tools import settings
import torch.utils.data
from torch.autograd import Variable
from echotorch import datasets
from torch import optim
import os


# Arguments
parser = argparse.ArgumentParser("Authorship Attribution on Elena Ferrante")
parser.add_argument("--output", type=str, help="Embedding output file", default='.')
parser.add_argument("--embedding-path", type=str, help="Embedding path")
parser.add_argument("--n-gram", type=int, default=1)
parser.add_argument("--no-cuda", action='store_true', default=False, help="Enables CUDA training")
parser.add_argument("--batch-size", type=float, default=64)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Load from directory
italian_book_dataset, italian_loader_train, italian_loader_test = dataset.load_dataset('./data', False)

# Print authors
print(u"Authors : {}".format(italian_book_dataset.classes))
print(u"{} documents".format(len(italian_book_dataset.samples)))
n_classes = italian_book_dataset.n_classes

# Word embedding
transform = torchlanguage.transforms.Compose([
    torchlanguage.transforms.Token(model='it_core_news_sm'),
    torchlanguage.transforms.GensimModel(model_path=args.embedding_path),
    torchlanguage.transforms.ToNGram(n=args.n_gram),
    torchlanguage.transforms.Reshape((-1, args.n_gram, 300))
])
transformer_output_dim = 300
italian_book_dataset.transform = transform

# Class to ID
class_to_ix = dict()
ix_to_class = dict()
for index, author in enumerate(italian_book_dataset.classes):
    class_to_ix[author] = index
    ix_to_class[index] = author
# end for

# Loss function
loss_function = nn.NLLLoss()

# For each batch
for k in range(10):
    # Log
    print(u"Fold {}".format(k))

    # Choose fold
    italian_loader_train.dataset.set_fold(k)
    italian_loader_test.dataset.set_fold(k)

    # Model
    model = torchlanguage.models.CGFS(
        n_gram=args.n_gram,
        n_authors=settings.n_authors,
        n_features=settings.cgfs_output_dim[args.n_gram]
    )
    if args.cuda:
        model.cuda()
    # end if

    # Best model
    best_acc = 0.0

    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=settings.cgfs_lr,
        momentum=settings.cgfs_momentum
    )

    # Epoch
    for epoch in range(settings.cgfs_epoch):
        # Total losses
        training_loss = 0.0
        test_loss = 0.0

        # Get training data for this fold
        for i, data in enumerate(italian_loader_train):
            # Inputs and labels
            sample_inputs, labels, time_labels = data

            # View
            sample_inputs = sample_inputs.view((-1, 1, args.n_gram, settings.glove_embedding_dim))

            # Outputs
            sample_outputs = torch.LongTensor(sample_inputs.size(0)).fill_(class_to_ix[labels[0]])

            # For each batch
            for pos in np.arange(0, sample_inputs.size(0) - args.batch_size, args.batch_size):
                # To variable
                inputs, outputs = Variable(sample_inputs[pos:pos + args.batch_size]), Variable(
                    sample_outputs[pos:pos + args.batch_size])
                if args.cuda:
                    inputs, outputs = inputs.cuda(), outputs.cuda()
                # end if

                # Zero grad
                model.zero_grad()

                # Compute output
                log_probs = model(inputs)

                # Loss
                loss = loss_function(log_probs, outputs)

                # Backward and step
                loss.backward()
                optimizer.step()

                # Add
                training_loss += loss.data[0]
            # end for
        # end for

        # Counters
        total = 0.0
        success = 0.0

        # Get test data for this fold
        for i, data in enumerate(italian_loader_test):
            # Inputs and labels
            sample_inputs, labels, time_labels = data

            # View
            sample_inputs = sample_inputs.view((-1, 1, args.n_gram, settings.glove_embedding_dim))

            # Outputs
            sample_outputs = torch.LongTensor(sample_inputs.size(0)).fill_(class_to_ix[labels[0]])

            # For each batch
            for pos in np.arange(0, sample_inputs.size(0) - args.batch_size, args.batch_size):
                # To variable
                inputs, outputs = Variable(sample_inputs[pos:pos + args.batch_size]), Variable(
                    sample_outputs[pos:pos + args.batch_size])
                if args.cuda:
                    inputs, outputs = inputs.cuda(), outputs.cuda()
                # end if

                # Forward
                model_outputs = model(inputs)
                loss = loss_function(model_outputs, outputs)

                # Take the max as predicted
                _, predicted = torch.max(model_outputs.data, 1)

                # Add to correctly classified word
                success += (predicted == outputs.data).sum()
                total += predicted.size(0)

                # Add loss
                test_loss += loss.data[0]
            # end for
        # end for

        # Accuracy
        accuracy = success / total * 100.0

        # Print and save loss
        print(
            u"Fold {}, epoch {}, training loss {}, test loss {}, accuracy {}".format(k, epoch, training_loss, test_loss,
                                                                                     accuracy))

        # Save if best
        if accuracy > best_acc:
            best_acc = accuracy
            # Save model
            print(u"Saving model with best accuracy {}".format(best_acc))
            torch.save(model.state_dict(), open(
                os.path.join(args.output, u"cgfs." + str(k) + u".p"),
                'wb'))
        # end if
    # end for

    # Log best accuracy
    print(u"Fold {} with best accuracy {}".format(k, best_acc))

    # Reset model
    model = None
# end for
