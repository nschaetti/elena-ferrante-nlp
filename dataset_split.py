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
import spacy
import codecs
import sys
import math


reload(sys)
sys.setdefaultencoding('utf8')

# Arguments
parser = argparse.ArgumentParser("Authorship Attribution on Elena Ferrante, split dataset")
parser.add_argument("--input", type=str, help="Embedding output fi  le", default='.')
parser.add_argument("--output", type=str, help="Embedding output file", default='.')
parser.add_argument("--split", type=int, help="Split text into how many parts", default=2)
args = parser.parse_args()

# Load model
print(u"Loading model")
nlp = spacy.load('it_core_news_sm')
print(u"Model loaded")

# For each file
for file_name in os.listdir(args.input):
    # File path
    file_path = os.path.join(args.input, file_name)
    sample_name = file_name[:-4]
    print(file_path)

    # Read sentence
    doc = nlp(codecs.open(file_path, 'r', encoding='utf-8').read())

    # Sentence list
    sentence_list = list()

    # For each sentence
    for sentence in doc.sents:
        sentence_list.append(sentence)
    # end for

    # Size part
    total_len = len(sentence_list)
    size_part = int(math.ceil(float(total_len) / float(args.split)))

    # For each part
    index = 1
    for i in np.arange(0, total_len - 1, size_part):
        # Sentence part
        part_sentences = sentence_list[i:i+size_part]

        # New file name
        new_file_name = sample_name + "_" + str(index) + ".txt"
        new_file_path = os.path.join(args.output, new_file_name)
        print(u"\t{}".format(new_file_path))

        # Open
        f = codecs.open(new_file_path, 'w', encoding='utf-8')

        # Write each sentence
        for s in part_sentences:
            f.write(unicode(s) + u" ")
        # end for

        # Close
        f.close()

        # Inc
        index += 1
    # end for
# end for
