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
import matplotlib.pyplot as plt

ITALIAN_APHLABET = u"aàbcdeèéfghiìíîjklmnoòópqrstuùúvwxyzAÀBCDEÈÉFGHIÌÍÎJKLMNOÒÓPQRSTUÙÚVWXYZ"
ITALIAN_APHLABET_LOWER = u"aàbcdeèéfghiìíîjklmnoòópqrstuùúvwxyz"
ITALIAN_PUNC = u".,;:'!?«»"

# Main function
if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="ElenaFerrante - Text classification on the IQLA dataset")

    # Argument
    parser.add_argument("--dataset", type=str, help="Dataset's directory", required=True)
    parser.add_argument("--k", type=int, help="k-Fold cross validation", default=10)
    parser.add_argument("--n-authors", type=int, help="Number of authors (-1: all)", default=-1)
    parser.add_argument("--uppercase", action='store_true', help="Keep uppercases", default=False)
    parser.add_argument("--to-one", action='store_true', help="Put frequencies with the max to one", default=False)
    parser.add_argument("--n-max-lines", type=int, help="Max. number of lines in output text files", required=True)
    parser.add_argument("--n-min-lines", type=int, help="Min. number of lines in output text files", required=True)
    parser.add_argument("--n-texts", type=float,
                        help="Number of text files based on a multiplication factor (#lines * m)", required=True)
    args = parser.parse_args()

    # Alphabet
    if args.uppercase:
        matrix_alphabet = ITALIAN_APHLABET
    else:
        matrix_alphabet = ITALIAN_APHLABET_LOWER
    # end if

    # Load dataset
    iqla = cp.IQLACorpus(dataset_path=args.dataset)

    # 10-Fold Cross validation
    cross_validation = nsNLP.validation.CrossValidation(iqla.get_texts())

    # Cross validation
    k = 0
    for train_set, test_set in cross_validation:
        print(u"Fold {}".format(k))

        # Samplers
        samplers = dict()

        # Add each text to sampling
        for train_sample in train_set:
            try:
                samplers[train_sample.get_author()].add_text(train_sample.get_text())
            except KeyError:
                samplers[train_sample.get_author()] = nsNLP.validation.TextSampling(train_sample.get_text(),
                                                                                    args.n_min_lines, args.n_max_lines)
            # end try
        # end for

        # Train the model on each authors
        for author in samplers.keys():
            # Sampler
            sampler = samplers[author]

            # Number of samples
            n_samples = int(args.n_texts * sampler.n_lines)

            # Get the samples
            for i in range(n_samples):
                # Get a new sample
                sample_text = sampler()

                # Letter stats
                grams = nsNLP.features.LetterStatistics(sample_text, alphabet=ITALIAN_APHLABET, punc=ITALIAN_PUNC)

                # Letters matrix
                grams_matrix = nsNLP.features.LettersMatrix(features_mapping=grams(), letters=matrix_alphabet,
                                                            punctuations=ITALIAN_PUNC)

                # Generate matrix
                m = grams_matrix(matrix_format='numpy')

                # Show matrix
                plt.imshow(m, cmap='gray')
                plt.show()
            # end for
        # end for

        # Delete
        del samplers

        k += 1
    # end for

# end if
