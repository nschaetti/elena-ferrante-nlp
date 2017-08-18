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

# Main function
if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="ElenaFerrante - Text classification on the IQLA dataset")

    # Argument
    parser.add_argument("--dataset", type=str, help="Dataset's directory", required=True)
    parser.add_argument("--k", type=int, help="k-Fold cross validation", default=10)
    args = parser.parse_args()

    # Load dataset
    iqla = cp.IQLACorpus(dataset_path=args.dataset)

    # Cross validation
    cross_validation = nsNLP.validation.CrossValidation(iqla.get_texts())

    # Model
    model = nsNLP.statistical_models.SLTextClassifier(classes=iqla.get_authors_list(), smoothing='dp',
                                                      smoothing_param=0.1)

    # For each fold
    k = 0
    for training_set, test_set in cross_validation:
        print(u"Fold {}".format(k))

        # Sizes
        training_size = len(training_set)

        # Training
        index = 1
        for sample in training_set:
            print(u"Training on {}/{} : {}".format(index, training_size, sample.get_path()))
            model.train(sample.get_text(), sample.get_author().get_name(), verbose=True)
            index += 1
        # end for

        # Success rate on training set
        print(u"Training success rate : {}".format(nsNLP.tools.Metrics.success_rate(model, training_set)))

        # Success rate on test set
        print(u"Test success rate : {}".format(nsNLP.tools.Metrics.success_rate(model, test_set)))

        # Next fold
        k += 1
    # end for


# end if
