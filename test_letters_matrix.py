# -*- coding: utf-8 -*-
#
# File : authorship_attribution.py
# Description : .
# Date : 20th of February 2017
#
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>

import nsNLP
import argparse
import codecs

ITALIAN_APHLABET = u"aàbcdeèéfghiìíîjklmnoòópqrstuùúvwxyzAÀBCDEÈÉFGHIÌÍÎJKLMNOÒÓPQRSTUÙÚVWXYZ"
ITALIAN_PUNC = u".,;:'!?«»"

# Main function
if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="ElenaFerrante - Test 2-gram of letters matrix")

    # Argument
    parser.add_argument("--file", type=str, help="Text file", required=True)
    args = parser.parse_args()

    # Read text
    text = codecs.open(args.file, 'r', encoding='utf-8').read()

    # 2-grams
    grams = nsNLP.features.Letter2Grams(text, alphabet=ITALIAN_APHLABET, punc=ITALIAN_PUNC)

    # Print 2-grams
    print(grams.get_beginning_2grams(uppercase=False))

# end if
