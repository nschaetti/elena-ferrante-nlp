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
import codecs
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

ITALIAN_APHLABET = u"aàbcdeèéfghiìíîjklmnoòópqrstuùúvwxyzAÀBCDEÈÉFGHIÌÍÎJKLMNOÒÓPQRSTUÙÚVWXYZ"
ITALIAN_APHLABET_LOWER = u"aàbcdeèéfghiìíîjklmnoòópqrstuùúvwxyz"
ITALIAN_PUNC = u".,;:'!?«»"

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
    parser.add_argument("--doc-csv", type=str, help="Output CSV file with weights between docs", default="")
    parser.add_argument("--author-csv", type=str, help="Output CSV file with weights with authors node", default="")
    parser.add_argument("--links-csv", type=str, help="Output CSV file with weights with document links and weights", default="")

    # Other
    parser.add_argument("--uppercase", action='store_true', help="Keep uppercases", default=False)
    parser.add_argument("--verbose", action='store_true', help="Verbose mode", default=False)
    parser.add_argument("--voc-size", type=int, help="Vocabulary size (in case of one-hot vector input)", default=5000)
    parser.add_argument("--sparse", action='store_true', help="Sparse matrix?", default=False)

    # Converters
    parser.add_argument("--converter", type=str, help="The text converter to use (fw, pos, tag, wv)", default='pos')
    parser.add_argument("--pca-model", type=str, help="PCA model to load", default=None)
    parser.add_argument("--in-components", type=int, help="Number of principal component to reduce inputs to",
                        default=-1)
    parser.add_argument("--log-level", type=int, help="Log level", default=20)
    args = parser.parse_args()

    # Alphabet
    if args.uppercase:
        matrix_alphabet = ITALIAN_APHLABET
    else:
        matrix_alphabet = ITALIAN_APHLABET_LOWER
    # end if

    # Load dataset
    iqla = cp.IQLACorpus(dataset_path=args.dataset)

    # Init logging
    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger(name="ElenaFerrante")

    # PCA model
    pca_model = None
    if args.pca_model is not None:
        pca_model = pickle.load(open(args.pca_model, 'r'))
    # end if

    # Choose a text to symbol converter.
    if args.converter == "pos":
        converter = nsNLP.PosConverter(resize=args.in_components, pca_model=pca_model)
    elif args.converter == "tag":
        converter = nsNLP.TagConverter(resize=args.in_components, pca_model=pca_model)
    elif args.converter == "fw":
        converter = nsNLP.FuncWordConverter(resize=args.in_components, pca_model=pca_model)
    elif args.converter == "wv":
        converter = nsNLP.WVConverter(resize=args.in_components, pca_model=pca_model)
    else:
        word2vec = nsNLP.Word2Vec(dim=args.voc_size, mapper='one-hot')
        converter = nsNLP.OneHotConverter(lang=args.lang, voc_size=args.voc_size, word2vec=word2vec)
    # end if

    # Document's IDs
    n_docs = iqla.get_n_texts()

    # Create Echo Word Classifier
    classifier = nsNLP.classifiers.EchoWordClassifier(
        classes=range(n_docs),
        size=args.reservoir_size,
        input_scaling=args.input_scaling,
        leak_rate=args.leak_rate,
        input_sparsity=args.input_sparsity,
        converter=converter,
        spectral_radius=args.spectral_radius,
        w_sparsity=args.w_sparsity,
        use_sparse_matrix=args.sparse
    )

    # Train each documents
    author2index = dict()
    index2author = dict()
    document2index = dict()
    index2document = dict()
    index = 0
    for document in iqla.get_texts():
        # Conversions
        author2index[document.get_author().get_name()].append(index)
        index2author[index] = document.get_author().get_name()
        document2index[document.get_path()] = index
        index2document[index] = document.get_path()

        # Train
        logger.info(u"Adding document {} as {}".format(document.get_path(), index))
        classifier.train(document.get_text(), index)

        # Next index
        index += 1
    # end for

    # Finalize model training
    classifier.finalize(verbose=args.verbose)

    # Get author embeddings
    document2vec = classifier.get_embeddings()
    logger.info(u"Author2vec shape : {}".format(document2vec.shape))

    # Similarity matrix
    similarity_matrix = np.zeros((iqla.get_n_authors(), iqla.get_n_authors()))

    # Compute similarity matrix
    for document1 in iqla.get_texts():
        for document2 in iqla.get_texts():
            if document1 != document2:
                document1_index = document2index[document1.get_path()]
                document2_index = document2index[document2.get_path()]
                document1_embedding = document2vec[document1_index]
                document2_embedding = document2vec[document2_index]
                distance = cosine_similarity(document1_embedding, document2_embedding)
                similarity_matrix[document1_index, document2_index] = distance
            # end if
        # end for
    # end for

    # Links matrix
    links_matrix = np.zeros((iqla.get_n_texts(), iqla.get_n_texts()))

    # Compute links matrix
    for index in range(iqla.get_n_texts()):
        # Get the row
        document_row = similarity_matrix[index, :]

        # Remove self relation
        document_row_cleaned = np.delete(document_row, index)

        # Threshold
        distance_threshold = 1.65 * np.std(document_row_cleaned)

        # Make
        links_matrix[index, document_row >= distance_threshold] = 1.0
    # end for

    # Output author CSV files
    if args.doc_csv != "":
        # Open the node file
        with codecs.open(args.author_csv) as f:
            # Header
            f.write(u"Id,Label")

            # For each doc
            for document in iqla.get_authors():
                document_index = document2index[document.get_path()]
                f.write(u"{},{}".format(document_index, document.get_author().get_name()))
            # end for
        # end with
    # end if

    # Output document weights CSV files
    if args.author_csv != "":
        # Open the edge file
        with codecs.open(args.doc_csv, 'w', encoding='utf-8') as f:
            # Header
            f.write(u"Source,Target,Weight,Author")

            # Compute distance between each documents
            for document1 in iqla.get_texts():
                for document2 in iqla.get_texts():
                    if document1 != document2:
                        document1_index = document2index[document1.get_path()]
                        document2_index = document2index[document2.get_path()]
                        document1_embedding = document2vec[document1_index]
                        document2_embedding = document2vec[document2_index]
                        distance = cosine_similarity(document1_embedding, document2_embedding)
                        f.write(u"{},{},{}".format(document1_index, document2_index, distance))
                    # end if
                # end for
            # end for
        # end with
    # end if

    # Output CSV file with links and weights
    if args.links_csv != "":
        # Open the file
        with codecs.open(args.links_csv, 'w', encoding='utf-8') as f:
            # Header
            f.write(u"Source,Target,Weight\n")

            # Compute distance between each authors
            for document1 in iqla.get_n_texts():
                for document2 in iqla.get_n_texts():
                    if document1 != document2:
                        document1_index = document2index[document1.get_path()]
                        document2_index = document2index[document2.get_path()]
                        document1_embedding = document2vec[document1_index]
                        document2_embedding = document2vec[document2_index]
                        distance = similarity_matrix[document1_embedding, document2_embedding]
                        link = links_matrix[document1_embedding, document2_embedding]
                        if link == 1.0:
                            f.write(u"{},{},{}\n".format(document1_index, document2_index, distance))
                        # end if
                    # end if
                # end for
            # end for
        # end with
    # end if

    # Reduce with t-SNE
    model = TSNE(n_components=2, random_state=0)
    reduced_matrix = model.fit_transform(document2vec.T)

    # Author embeddings matrix's size
    logger.info(u"Reduced matrix's size : {}".format(reduced_matrix.shape))

    # Show t-SNE
    plt.figure(figsize=(args.fig_size*0.003, args.fig_size*0.003), dpi=300)
    max_x = np.amax(reduced_matrix, axis=0)[0]
    max_y = np.amax(reduced_matrix, axis=0)[1]
    min_x = np.amin(reduced_matrix, axis=0)[0]
    min_y = np.amin(reduced_matrix, axis=0)[1]
    plt.xlim((min_x * 1.2, max_x * 1.2))
    plt.ylim((min_y * 1.2, max_y * 1.2))
    for document in iqla.get_texts():
        document_index = document2index[document.get_path()]
        plt.scatter(reduced_matrix[document_index, 0], reduced_matrix[document_index, 1], 0.5)
        plt.text(reduced_matrix[document_index, 0], reduced_matrix[document_index], document.get_author().get_name(),
                 fontsize=2.5)
    # end for

    # Save image
    plt.savefig(args.output)

# end if
