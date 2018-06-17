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
parser.add_argument("--n-gram", type=str, help="Character n-gram", default='c1')
parser.add_argument("--text-length", type=int, help="Text length", default=20)
parser.add_argument("--n-filters", type=int, help="Number of filters", default=100)
parser.add_argument("--no-cuda", action='store_true', default=False, help="Enables CUDA training")
parser.add_argument("--batch-size", type=float, default=64)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Load from directory
italian_book_dataset, italian_loader_train, italian_loader_test = dataset.load_dataset('./data2', False)

# Print authors
print(u"Authors : {}".format(italian_book_dataset.classes))
print(u"{} documents".format(len(italian_book_dataset.samples)))
n_classes = italian_book_dataset.n_classes

# Word embedding
# Transforms
if args.n_gram == 'c1':
    transform = torchlanguage.transforms.Compose([
        torchlanguage.transforms.Character(),
        torchlanguage.transforms.ToIndex(start_ix=0),
        torchlanguage.transforms.ToNGram(n=args.text_length, overlapse=True),
        torchlanguage.transforms.Reshape((-1, args.text_length))
    ])
else:
    transform = torchlanguage.transforms.Compose([
        torchlanguage.transforms.Character2Gram(),
        torchlanguage.transforms.ToIndex(start_ix=0),
        torchlanguage.transforms.ToNGram(n=args.text_length, overlapse=True),
        torchlanguage.transforms.Reshape((-1, args.text_length))
    ])
# end if
transformer_output_dim = 150
italian_book_dataset.transform = transform

# Class to ID
class_to_ix = dict()
ix_to_class = dict()
for index, author in enumerate(italian_book_dataset.classes):
    class_to_ix[author] = index
    ix_to_class[index] = author
# end for

# Loss function
loss_function = nn.CrossEntropyLoss()

# For each batch
for k in range(10):
    # Log
    print(u"Fold {}".format(k))

    # Choose fold
    italian_loader_train.dataset.set_fold(k)
    italian_loader_test.dataset.set_fold(k)

    # Model
    model = torchlanguage.models.CCSAA(
        text_length=args.text_length,
        vocab_size=settings.ccsaa_voc_size,
        embedding_dim=settings.ccsaa_embedding_dim,
        n_classes=settings.n_authors,
        out_channels=(args.n_filters, args.n_filters, args.n_filters)
    )
    if args.cuda:
        model.cuda()
    # end if

    # Best model
    best_acc = 0.0
    best_author_acc = 0.0

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=settings.ccsaa_lr, momentum=settings.ccsaa_momentum)

    # Epoch
    for epoch in range(settings.cgfs_epoch):
        # Total losses
        training_loss = 0.0
        test_loss = 0.0

        # Aggregate
        aggregate_inputs = list()
        aggregate_outputs = list()

        # Get training data for this fold
        for i, data in enumerate(italian_loader_train):
            # Inputs and labels
            sample_inputs, labels, time_labels = data

            # Reshape
            sample_inputs = sample_inputs.view(-1, args.text_length)

            # Outputs
            sample_outputs = torch.LongTensor(sample_inputs.size(0)).fill_(class_to_ix[labels[0]])

            # Append
            if (i != 0 and i % 10 == 0) or i == 599:
                # Add
                if i == 599:
                    aggregate_inputs = torch.cat((aggregate_inputs, sample_inputs), dim=0)
                    aggregate_outputs = torch.cat((aggregate_outputs, sample_outputs), dim=0)
                # end if

                # Create permutation
                perm = np.random.choice(aggregate_inputs.size(0), aggregate_inputs.size(0))

                # Apply permutation
                aggregate_inputs = aggregate_inputs[torch.LongTensor(perm)]
                aggregate_outputs = aggregate_outputs[torch.LongTensor(perm)]

                # For each batch
                for pos in np.arange(0, aggregate_inputs.size(0) - args.batch_size, args.batch_size):
                    # To variable
                    inputs, outputs = Variable(aggregate_inputs[pos:pos + args.batch_size]), Variable(
                        aggregate_outputs[pos:pos + args.batch_size])
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

                # Reset aggregate
                aggregate_inputs = sample_inputs
                aggregate_outputs = sample_outputs
            elif i == 0:
                # Reset aggregate
                aggregate_inputs = sample_inputs
                aggregate_outputs = sample_outputs
            else:
                # Aggregate
                aggregate_inputs = torch.cat((aggregate_inputs, sample_inputs), dim=0)
                aggregate_outputs = torch.cat((aggregate_outputs, sample_outputs), dim=0)
            # end if
        # end for

        # Counters
        total = 0.0
        success = 0.0
        author_total = 0.0
        author_success = 0.0

        # Get test data for this fold
        for i, data in enumerate(italian_loader_test):
            # Inputs and labels
            sample_inputs, labels, time_labels = data

            # Reshape
            sample_inputs = sample_inputs.view(-1, args.text_length)

            # Outputs
            sample_outputs = torch.LongTensor(sample_inputs.size(0)).fill_(class_to_ix[labels[0]])

            # Label
            label = torch.LongTensor(1).fill_(class_to_ix[labels[0]])
            author_prob = torch.zeros(1, n_classes)
            if args.cuda:
                author_prob, label = author_prob.cuda(), label.cuda()
            # end if

            # For each batch
            prob_count = 0.0
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

                # Add
                author_prob += torch.sum(model_outputs.data, dim=0)

                # Take the max as predicted
                _, predicted = torch.max(model_outputs.data, 1)

                # Add to correctly classified word
                success += (predicted == outputs.data).sum()
                total += predicted.size(0)

                # Add loss
                test_loss += loss.data[0]

                # Prob count
                prob_count += inputs.size(0)
            # end for

            # Prob over time
            author_prob /= prob_count

            # Max over time
            _, author_predicted = torch.max(author_prob, dim=1)

            # Add to correctly classified word
            author_success += (author_predicted == label).sum()
            author_total += 1.0
        # end for

        # Accuracy
        accuracy = success / total * 100.0
        author_accuracy = author_success / author_total * 100.0

        # Print and save loss
        print(
            u"Fold {}, epoch {}, training loss {}, test loss {}, accuracy {}, author accuracy {}".format(k, epoch, training_loss, test_loss,
                                                                                     accuracy, author_accuracy))

        # Save if best
        if accuracy > best_acc:
            best_acc = accuracy
            # Save model
            print(u"Saving model with best accuracy {}".format(best_acc))
            torch.save(model.state_dict(), open(
                os.path.join(args.output, u"cgfs." + str(k) + u".p"),
                'wb'))
        # end if

        # Save if best
        if author_accuracy > best_author_acc:
            best_author_acc = author_accuracy
            # Save model
            print(u"Saving author model with best author accuracy {}".format(best_author_acc))
            torch.save(model.state_dict(), open(
                os.path.join(args.output, u"cgfs.author." + str(k) + u".p"),
                'wb'))
        # end if
    # end for

    # Log best accuracy
    print(u"Fold {} with best accuracy {}".format(k, best_acc))

    # Reset model
    model = None
# end for
