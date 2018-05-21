# -*- coding: utf-8 -*-
#
# File : authorship_attribution.py
# Description : .
# Date : 20th of February 2017
#
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>

# Imports
import os

# For each file
for file_name in os.listdir("data"):
    file_path = os.path.join("data", file_name)
    author = file_name[:file_name.find("_")]
    title = file_name[file_name.find("_")+1:]
    title = title[title.find("_")+1:]
    title = title.replace(" ", "")
    new_title = author + "_" + title
    os.rename(file_path, os.path.join("data", new_title))
# end for

