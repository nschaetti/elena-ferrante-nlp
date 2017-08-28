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
    parser.add_argument("--n-lines", type=int, help="Number of lines to take in each text", required=True)
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

        # For each text
        for text_sample in author.get_texts():
            # Get text
            text = codecs.open(text_sample.get_path(), 'r', encoding='utf-8').read()

            # Log
            print(u"Parsing file {}".format(text_sample.get_path()))

            # For each line
            index = 1
            for line in text.split(u'\n'):
                if len(line) > 1:
                    # Open a file
                    if index % args.n_lines == 1:
                        random_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
                        file_path = os.path.join(args.output, random_name + u".txt")
                        print(u"Writing file {}".format(file_path))
                        f = codecs.open(file_path, 'w', encoding='utf-8')
                        text_informations[random_name] = author.get_name()
                        author_informations[author.get_name()].append(random_name)
                    # end if

                    # Write line
                    f.write(line + u'\n')

                    # Close the file
                    if index % args.n_lines == 0:
                        f.close()
                    # end if
                    index += 1
                # end if
            # end for
        # end for
    # end for

    # Save informations
    json.dump(author_informations, open(os.path.join(args.output, "authors.json"), 'w'))
    json.dump(text_informations, open(os.path.join(args.output, "texts.json"), 'w'))

# end if
