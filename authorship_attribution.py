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

    # For each fold
    """k = 0
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

        # Success rate on test set
        print(u"Test success rate : {}".format(nsNLP.tools.Metrics.success_rate(model, test_set)))

        # Next fold
        k += 1
    # end for"""

    # Models validation
    models_validation = nsNLP.validation.ModelsValidation(iqla.get_texts())

    # Add 1-gram statistical model with DP smoothing
    for smoothing_param in np.arange(1, 100001, 10000):
        models_validation.add_model(
            nsNLP.statistical_models.SLTextClassifier(classes=iqla.get_authors_list(), smoothing='dp',
                                                      smoothing_param=smoothing_param))
    # end for

    # Add 1-gram statistical model with JM smoothing
    for smoothing_param in np.arange(0.05, 1.05, 0.1):
        models_validation.add_model(
            nsNLP.statistical_models.SLTextClassifier(classes=iqla.get_authors_list(), smoothing='jm',
                                                      smoothing_param=smoothing_param))
    # end for

    # Add TF-IDF
    models_validation.add_model(nsNLP.tfidf.TFIDFTextClassifier(classes=iqla.get_authors_list()))

    # Add 2-gram statistical model with DP smoothing
    for smoothing_param in np.arange(0, 100000, 10000):
        models_validation.add_model(
            nsNLP.statistical_models.SL2GramTextClassifier(classes=iqla.get_authors_list(), smoothing='dp',
                                                           smoothing_param=smoothing_param))
    # end for

    # Add 2-gram statistical model with DP smoothing
    for smoothing_param in np.arange(0.05, 1.05, 0.1):
        models_validation.add_model(
            nsNLP.statistical_models.SL2GramTextClassifier(classes=iqla.get_authors_list(), smoothing='jm',
                                                           smoothing_param=smoothing_param))
    # end for

    # Compare models
    results, comparisons = models_validation.compare()

    # Display
    print(u"10-Fold cross validation for each models : ")
    for model_name in results.keys():
        print(u"{} : {}".format(model_name, results[model_name]))
    # end for
    print(u"")

    # Display t-tests results
    print(u"Two samples t-test results : ")
    for model1_name, model2_name in comparisons.keys():
        print(u"{} vs {} : {}".format(model1_name, model2_name, comparisons[(model1_name, model2_name)] * 100.0))
    # end for

# end if
