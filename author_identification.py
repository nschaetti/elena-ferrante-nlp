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
    parser = argparse.ArgumentParser(description="ElenaFerrante - Author identification on the IQLA dataset")

    # Argument
    parser.add_argument("--dataset", type=str, help="Dataset's directory", required=True)
    parser.add_argument("--author", type=str, help="Target author's name", required=True)
    parser.add_argument("--k", type=int, help="k-Fold cross validation", default=10)
    args = parser.parse_args()

    # Load dataset
    iqla = cp.IQLACorpus(dataset_path=args.dataset)

    # Cross validation
    cross_validation = nsNLP.validation.TwoClassesCrossValidation(k=7)

    # Model
    model = nsNLP.statistical_models.SLTextClassifier(classes=iqla.get_authors_list(), smoothing='dp',
                                                      smoothing_param=0.1)

    # For each sample in the database
    for sample in iqla.get_texts():
        if sample.get_author().get_name() == args.author:
            cross_validation.add_positive(sample)
        else:
            cross_validation.add_negative(sample)
        # end if
    # end for

    # For each fold
    k = 0
    for pos_train_set, neg_train_set, pos_test_set, neg_test_set in cross_validation:
        print(u"Fold {}".format(k))

        # Sizes
        pos_training_size = len(pos_train_set)
        neg_training_size = len(neg_train_set)

        # Positive training
        index = 1
        for sample in pos_train_set:
            print(u"Training positive sample {}".format(sample.get_path()))
            model.train(sample.get_text(), "positive", verbose=True)
            index += 1
        # end for

        # Negative training
        index = 1
        for sample in neg_train_set:
            print(u"Training negative sample {}".format(sample.get_path()))
            model.train(sample.get_text(), "negative", verbose=True)
            index += 1
        # end for

        # Init counter
        total = 0.0
        successes = 0.0

        # Test positive samples
        for sample in pos_test_set:
            # Prediction
            prediction, _ = model(sample)
            print(u"Testing positive sample {} : {}".format(sample.get_text(), prediction))

            # Test
            if prediction == "positive":
                successes += 1.0
            # end if
            total += 1.0
        # end for

        # Test negative samples
        for sample in neg_test_set:
            # Prediction
            prediction, _ = model(sample)
            print(u"Testing negative sample {} : {}".format(sample.get_text(), prediction))

            # Test
            if prediction == "negative":
                successes += 1.0
            # end if
            total += 1.0
        # end for

        # Success rate
        print(u"{} fold success rate : {}".format(successes/total*100.0))

        # Next fold
        k += 1
    # end for

# end if
