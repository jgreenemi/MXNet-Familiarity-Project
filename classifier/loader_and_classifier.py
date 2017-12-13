# Pull in text data, normalize it for processing.

import itertools
import numpy as np
import re
from urllib.request import urlopen
from collections import Counter

datafile_url_prefix = 'https://raw.githubusercontent.com/jgreenemi/MXNet-Familiarity-Project/master/'
datafile_positive = 'rt-positive.txt'
datafile_negative = 'rt-negative.txt'

def clean_strings(string):
    """
    Clean up strings for better processing.
    :param string:
    :return:
    """

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)

    # I'm not sure what's the reason for tokenizing strings' concatenations in this manner.
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    # More can/should be added.

    # Trim erroneous spaces from strings, force to lowercase, and return.
    return string.strip().lower()


def load_data_with_labels():
    """
    Pull in data from files, return the split sentences and labels.
    :return:
    """