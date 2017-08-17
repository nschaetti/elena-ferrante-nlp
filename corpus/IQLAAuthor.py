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


# Class to access to a IQLA author
class IQLAAuthor(object):
    """
    Class to access to a IQLA author
    """

    # Constructor
    def __init__(self, dataset_path, name, texts):
        """
        Constructor
        """
        # Properties
        self._dataset_path = dataset_path
        self._name = name
        self._texts = texts
    # end __init__

    ###########################################
    # Public
    ###########################################

    # Get name
    def get_name(self):
        """
        Get name
        :return:
        """
        return self._name
    # end get_name

    ############################################
    # Override
    ############################################

    # Get

# end IQLAAuthor
