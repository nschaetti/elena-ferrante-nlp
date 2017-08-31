# -*- coding: utf-8 -*-
#
# File : authorship_attribution.py
# Description : .
# Date : 20th of February 2017
#
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>

import argparse
import corpus as cp
import random
import string
import os
import json
import codecs

# Main function
if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="ElenaFerrante - Text classification on the IQLA dataset")

    # Argument
    parser.add_argument("--dataset", type=str, help="Dataset's directory", required=True)
    parser.add_argument("--n-max-lines", type=int, help="Max. number of lines in output text files", required=True)
    parser.add_argument("--n-min-lines", type=int, help="Min. number of lines in output text files", required=True)
    parser.add_argument("--n-texts", type=float, help="Number of text files based on a multiplication factor (#lines * m)", required=True)
    parser.add_argument("--output", type=str, help="Output directory", required=True)
    args = parser.parse_args()

    # Load dataset
    iqla = cp.IQLACorpus(dataset_path=args.dataset)

    # Authors informations
    author_informations = dict()
    text_informations = dict()

    # For each author
    for author in iqla.get_authors():
        print(u"Author {}".format(author.get_name()))
        author_informations[author.get_name()] = list()

        # Get complete texts
        author_texts = ""
        for text_sample in author.get_texts():
            # Get text
            text = codecs.open(text_sample.get_path(), 'r', encoding='utf-8').read()

            # Add
            author_texts += u"\n" + text
        # end for

        # Remove blank line
        for i in range(20):
            author_texts = author_texts.replace(u"\n\n", u"\n")
        # end for

        # Get all lines
        text_lines = author_texts.split(u"\n")

        # Number of lines
        n_lines = len(text_lines)
        print(u"Author {} has {} lines".format(author.get_name(), n_lines))

        # Number of files
        n_texts = float(n_lines * args.n_texts)

        # Generate each file
        for i in range(n_texts):
            file_text = u""

            # Random number of lines
            n_random_lines = random.randint(args.n_min_lines, args.n_max_lines)

            # Select lines
            for j in range(n_random_lines):
                # Select line
                n_random_pos = random.randint(0, n_lines)

                # Add
                file_text += text_lines[n_random_pos]
            # end for

            # Filename
            file_name = author.get_name() + u"_" + unicode(i)

            # Write the file
            file_path = os.path.join(args.output, file_name + u".txt")
            codecs.open(file_path, 'w', encoding='utf-8').write(file_text)

            # Add info
            author_informations[author.get_name()].append(file_name)
            text_informations[file_name] = author.get_name()
        # end for
    # end for

    # Save informations
    json.dump(author_informations, open(os.path.join(args.output, "authors.json"), 'w'))
    json.dump(text_informations, open(os.path.join(args.output, "texts.json"), 'w'))

# end if
