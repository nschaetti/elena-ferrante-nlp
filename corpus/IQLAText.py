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


# Class to access to a IQLA text
class IQLAText(object):
    """
    Class to access to a IQLA text
    """

    # Constructor
    def __init__(self, text_path, author):
        """
        Constructor
        :param text_path:
        :param author:
        """
        self._text_path = text_path
        self._author = author
    # end __init__

    ########################################
    # Public
    ########################################

    # Get text
    def get_text(self):
        """
        Get text
        :return:
        """
        return open(self._text_path, 'r').read()
    # end text

    # Get author
    def get_author(self):
        """
        Get author
        :return:
        """
        return self._author
    # end author

    ########################################
    # Override
    ########################################

    # To string
    def __unicode__(self):
        """
        To string
        :return:
        """
        return u"IQLAText(path:{}, author:{})".format(self._text_path, self._author.get_name())
    # end __unicode__

    # To string
    def __str__(self):
        """
        To string
        :return:
        """
        return "IQLAText(path:{}, author:{})".format(self._text_path, self._author.get_name())
    # end __unicode__

# end IQLAText
