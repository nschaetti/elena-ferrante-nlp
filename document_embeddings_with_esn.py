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
import pickle
import logging


# Main function
if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="ElenaFerrante - Author clustering with author embeddings with ESN")

    # Argument
    parser.add_argument("--dataset", type=str, help="Dataset's directory", required=True)

    # Reservoir properties
    parser.add_argument("--reservoir-size", type=int, help="Reservoir's size", required=True)
    parser.add_argument("--spectral-radius", type=float, help="Spectral radius", default=0.99)
    parser.add_argument("--leak-rate", type=float, help="Reservoir's leak rate", default=1.0)
    parser.add_argument("--input-scaling", type=float, help="Input scaling", default=0.5)
    parser.add_argument("--input-sparsity", type=float, help="Input sparsity", default=0.05)
    parser.add_argument("--w-sparsity", type=float, help="W sparsity", default=0.05)

    # Output
    parser.add_argument("--output", type=str, help="Output image", required=True)
    parser.add_argument("--fig-size", type=float, help="Figure size (pixels)", default=1024.0)
    parser.add_argument("--node-csv", type=str, help="Output CSV file with weights between docs", default="")
    parser.add_argument("--weights-csv", type=str, help="Output CSV file with weights with authors node", default="")
    parser.add_argument("--links-csv", type=str, help="Output CSV file with weights with document links and weights", default="")
    parser.add_argument("--ordered-list", type=str, help="Output CSV file of ordered distances", default="")

    # Other
    parser.add_argument("--uppercase", action='store_true', help="Keep uppercases", default=False)
    parser.add_argument("--verbose", action='store_true', help="Verbose mode", default=False)
    parser.add_argument("--voc-size", type=int, help="Vocabulary size (in case of one-hot vector input)", default=5000)

    # Converters
    parser.add_argument("--converter", type=str, help="The text converter to use (fw, pos, tag, wv)", default='pos')
    parser.add_argument("--pca-model", type=str, help="PCA model to load", default=None)
    parser.add_argument("--in-components", type=int, help="Number of principal component to reduce inputs to",
                        default=-1)
    parser.add_argument("--log-level", type=int, help="Log level", default=20)
    args = parser.parse_args()

    # Load dataset
    iqla = cp.IQLACorpus(dataset_path=args.dataset)

    # Init logging
    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger(name=u"ElenaFerrante")

    # PCA model
    pca_model = None
    if args.pca_model is not None:
        pca_model = pickle.load(open(args.pca_model, 'r'))
    # end if

    # Choose a text to symbol converter.
    if args.converter == "pos":
        converter = nsNLP.esn_models.converters.PosConverter(resize=args.in_components, pca_model=pca_model)
    elif args.converter == "tag":
        converter = nsNLP.esn_models.converters.TagConverter(resize=args.in_components, pca_model=pca_model)
    elif args.converter == "fw":
        converter = nsNLP.esn_models.converters.FuncWordConverter(resize=args.in_components, pca_model=pca_model)
    elif args.converter == "wv":
        word2vec = nsNLP.embeddings.SpacyWord2Vec(lang='en')
        converter = nsNLP.esn_models.converters.WVConverter(resize=args.in_components, pca_model=pca_model)
    else:
        converter = nsNLP.esn_models.converters.OneHotConverter(voc_size=args.voc_size)
    # end if

    # Document's IDs
    n_docs = iqla.get_n_texts()

    # Create Echo Word Classifier
    classifier = nsNLP.esn_models.ESNTextClassifier(
        classes=range(n_docs),
        size=args.reservoir_size,
        input_scaling=args.input_scaling,
        leak_rate=args.leak_rate,
        input_sparsity=args.input_sparsity,
        converter=converter,
        spectral_radius=args.spectral_radius,
        w_sparsity=args.w_sparsity,
        use_sparse_matrix=True
    )

    # Tokenizer
    tokenizer = nsNLP.tokenization.NLTKTokenizer(lang='italian')

    # Documents and indexes information
    author2index = dict()
    index2author = dict()
    document2index = dict()
    index2document = dict()
    document2author = dict()
    index = 0

    # Train on each document
    for document in iqla.get_texts():
        # Conversions
        try:
            author2index[document.get_author().get_name()].append(index)
        except KeyError:
            author2index[document.get_author().get_name()] = list()
            author2index[document.get_author().get_name()].append(index)
        # end try
        index2author[index] = document.get_author().get_name()
        document2index[document.get_path()] = index
        index2document[index] = document.get_path()
        document2author[document.get_path()] = document.get_author().get_name()

        # Tokenizing
        tokens = tokenizer(document.get_text())

        # Train
        logger.info(u"Adding document {} as {} of length {}".format(document.get_path(), index, len(tokens)))
        classifier.train(tokens, index)

        # Next index
        index += 1
    # end for

    # Finalize model training
    classifier.finalize(verbose=args.verbose)

    # Get author embeddings
    document2vec = classifier.get_embeddings()
    logger.info(u"Document2Vec shape : {}x{}".format(len(document2vec.keys()), document2vec[0].shape[0]))

    # Similarity matrix & links matrix
    similarity_matrix = nsNLP.clustering.tools.DistanceMeasures.similarity_matrix(document2vec)
    links_matrix = nsNLP.clustering.tools.DistanceMeasures.link_matrix(similarity_matrix)

    # Output node CSV files
    if args.node_csv != "":
        nsNLP.visualisation.EmbeddingsVisualisation.node_csv(args.node_csv, range(n_docs), index2author)
    # end if

    # Output document weights CSV files
    if args.weights_csv != "":
        nsNLP.visualisation.EmbeddingsVisualisation.weights_csv(args.weights_csv, similarity_matrix)
    # end if

    # Output CSV file with links and weights
    if args.links_csv != "":
        nsNLP.visualisation.EmbeddingsVisualisation.weights_csv(args.weights_csv, similarity_matrix, links_matrix)
    # end if

    # Save TSNE
    nsNLP.visualisation.EmbeddingsVisualisation.tsne(document2vec, args.fig_size, args.output, index2author)

    # Save similarities ordered by distance
    nsNLP.visualisation.EmbeddingsVisualisation.ordered_distances_csv(args.ordered_list, document2vec, index2author)

# end if
