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
import matplotlib.pyplot as plt

ITALIAN_APHLABET = u"aàbcdeèéfghiìíîjklmnoòópqrstuùúvwxyzAÀBCDEÈÉFGHIÌÍÎJKLMNOÒÓPQRSTUÙÚVWXYZ"
ITALIAN_APHLABET_LOWER = u"aàbcdeèéfghiìíîjklmnoòópqrstuùúvwxyz"
ITALIAN_PUNC = u".,;:'!?«»"

# Main function
if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="ElenaFerrante - Test 2-gram of letters matrix")

    # Argument
    parser.add_argument("--file", type=str, help="Text file", required=True)
    parser.add_argument("--uppercase", action='store_true', help="Keep uppercases", default=False)
    parser.add_argument("--to-one", action='store_true', help="Put frequencies with the max to one", default=False)
    args = parser.parse_args()

    # Read text
    text = codecs.open(args.file, 'r', encoding='utf-8').read()

    # Letter stats
    grams = nsNLP.features.LetterStatistics(text, alphabet=ITALIAN_APHLABET, punc=ITALIAN_PUNC)

    # Get statistics
    grams_stats = dict()
    grams_stats['grams'] = grams.get_2grams(uppercase=args.uppercase, to_one=args.to_one)
    grams_stats['first_grams'] = grams.get_beginning_2grams(uppercase=args.uppercase, to_one=args.to_one)
    grams_stats['end_grams'] = grams.get_ending_2grams(uppercase=args.uppercase, to_one=args.to_one)
    grams_stats['punctuations'] = grams.get_punctuation(to_one=args.to_one)

    # Alphabet
    if args.uppercase:
        matrix_alphabet = ITALIAN_APHLABET
    else:
        matrix_alphabet = ITALIAN_APHLABET_LOWER
    # end if

    # Letters matrix
    grams_matrix = nsNLP.features.LettersMatrix(features_mapping=grams_stats, letters=matrix_alphabet,
                                                punctuations=ITALIAN_PUNC)

    # Generate matrix
    m = grams_matrix(matrix_format='numpy')

    # Show matrix
    plt.imshow(m, cmap='gray')
    plt.show()

# end if
