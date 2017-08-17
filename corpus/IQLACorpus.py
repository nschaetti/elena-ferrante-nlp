# -*- coding: utf-8 -*-
#
# File : corpus/IQLACorpus.py
# Description : .
# Date : 16/08/2017
#
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>

# Imports
import json
import os


# Class to access to the IQLA corpus
class IQLACorpus(object):
    """
    Class to access to the IQLA corpus
    """

    # Constructor
    def __init__(self, dataset_path):
        """
        Constructor
        :param dataset_path: Path to dataset
        """
        # Properties
        self._dataset_path = dataset_path
        self._authors = list()
        self._texts = list()

        # Load dataset
        self._load()
    # end __init__

    ########################################
    # Public
    ########################################

    # Get list of authors
    def authors(self):
        """
        Get list of authors
        :return:
        """
        pass
    # end authors

    # Get the number of authors
    def get_n_authors(self):
        """
        Get the number of authors
        :return:
        """
        return len(self._authors)
    # end get_n_authors

    # Get the number of texts
    def get_n_texts(self):
        """
        Get the number of texts
        :return:
        """
        return len(self._texts)
    # end get_n_texts

    ########################################
    # Override
    ########################################

    # Iterator
    def __iter__(self):
        """
        Iterator
        :return:
        """
        return self
    # end __iter__

    # Next
    def next(self):
        """
        Next
        :return:
        """
        pass
    # end next

    ########################################
    # Private
    ########################################

    # Load
    def _load(self):
        """
        Load
        :return:
        """
        self._authors = json.load(os.path.join(self._dataset_path, "authors.json"))
        self._texts = json.load(os.path.join(self._dataset_path, "authors.json"))
    # end _load

# end IQLACorpus
