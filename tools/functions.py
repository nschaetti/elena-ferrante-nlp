#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import torch
import echotorch.nn as etnn


#########################################
# Other
#########################################


# Manage W
def manage_w(xp, args, keep_w):
    """
    Manage W
    :param xp:
    :param args:
    :param keep_w:
    :return:
    """
    # First params
    rc_size = int(args.get_space()['reservoir_size'][0])
    rc_w_sparsity = args.get_space()['w_sparsity'][0]

    # Create W matrix
    w = etnn.ESNCell.generate_w(rc_size, rc_w_sparsity)

    # Save classifier
    if keep_w:
        xp.save_object(u"w", w)
    # end if

    return w
# end manage_w


# Get params
def get_params(space):
    """
    Get params
    :param space:
    :return:
    """
    # Params
    reservoir_size = int(space['reservoir_size'])
    w_sparsity = space['w_sparsity']
    leak_rate = space['leak_rate']
    input_scaling = space['input_scaling']
    input_sparsity = space['input_sparsity']
    spectral_radius = space['spectral_radius']
    feature = space['transformer'][0][0]
    aggregation = space['aggregation'][0][0]
    state_gram = space['state_gram']
    feedbacks_sparsity = space['feedbacks_sparsity']
    lang = space['lang'][0][0]
    embedding = space['embedding'][0][0]
    return reservoir_size, w_sparsity, leak_rate, input_scaling, input_sparsity, spectral_radius, feature, aggregation, \
           state_gram, feedbacks_sparsity, lang, embedding
# end get_params


# Converter in
def converter_in(converters_desc, converter):
    """
    Is the converter in the desc
    :param converters_desc:
    :param converter:
    :return:
    """
    for converter_desc in converters_desc:
        if converter in converter_desc:
            return True
        # end if
    # end for
    return False
# end converter_in


# To time labels
def to_timelabels(length, ix, n_classes):
    """
    To time labels
    :param length:
    :param ix:
    :return:
    """
    time_labels = torch.zeros(1, length, n_classes)
    time_labels[0, :, ix] = 1.0
    return time_labels
# end to_timelabels
