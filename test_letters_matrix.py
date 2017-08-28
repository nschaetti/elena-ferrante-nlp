# -*- coding: utf-8 -*-
#
# File : authorship_attribution.py
# Description : .
# Date : 20th of February 2017
#
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>

import nsNLP
import argparse
import corpus as cp
import random
import string
import os
import json
import codecs

# Main function
if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="ElenaFerrante - Test 2-gram of letters matrix")

    # Argument
    parser.add_argument("--file", type=str, help="Text file", required=True)
    args = parser.parse_args()

    # Read text
    text = codecs.open(args.file, 'w', encoding='utf-8')

    # 2-grams
    grams = nsNLP.features.Letter2Grams(text, alphabet=u"abcdefghilmnopqrstuvz", punc=u".,;:'!?")

# end if
