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
import numpy as np
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
    parser.add_argument("--smoothing", type=str, help="Smoothing type (dp, jm)", default=-1)
    parser.add_argument("--smoothing-param", type=float, help="Smoothing parameter", default=-1)
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

    # Classification model
    classifier = nsNLP.statistical_models.NaiveBayesClassifier(classes=iqla.get_authors_list(),
                                                               smoothing=args.smoothing,
                                                               smoothing_param=args.smoothing_param)

    # Success rates
    success_rates = np.array([])

    # Cross validation
    k = 0
    for train_set, test_set in cross_validation:
        print(u"Fold {}".format(k))

        # For each samples
        for index, sample in enumerate(train_set):
            print u"\rLearning {}/{}".format(index+1, len(train_set)),
            # Bag of words
            bow = nsNLP.features.BagOfWords('en')

            # Train
            classifier.train(bow(sample.x()), sample.y())
        # end for
        print(u"")

        # Finalize
        classifier.finalize()

        # For each test samples
        count = 0.0
        successes = 0.0
        for index, sample in enumerate(test_set):
            # Bag of words
            bow = nsNLP.features.BagOfWords('en')

            # Predict
            predicted, _ = classifier(bow(sample.x()))

            # Test
            if predicted == sample.y():
                successes += 1.0
            # end if

            count += 1.0
        # end for

        # Success rate
        success_rate = successes / count

        # Show success rate
        print(u"Succeess rate : {}".format(success_rate))

        # Add
        success_rates = np.append(success_rates, [success_rate])

        k += 1
    # end for

    # Show average
    print(u"Average success rate : {}".format(np.average(success_rates)))

# end if
