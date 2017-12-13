# Pull in text data, normalize it for processing.

import itertools
import numpy as np
import re
from collections import Counter
from pprint import pprint
from urllib.request import urlopen

datafile_url_prefix = 'https://raw.githubusercontent.com/jgreenemi/MXNet-Familiarity-Project/master/resources/'
datafile_positive = '{}rt-positive.txt'.format(datafile_url_prefix)
datafile_negative = '{}rt-negative.txt'.format(datafile_url_prefix)

def clean_strings(string):
    """
    Clean up strings for better processing.
    :param string:
    :return:
    """

    # In Python 3, these strings can be bytes type, and need to be converted to string type for re to work.
    string = str(string)

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
    Returns the split sentences and the labels.
    :return: Tuple of [numpy.ndarray, List]
    """

    # Retrieve and load text from files.
    posfile = urlopen(datafile_positive)
    negfile = urlopen(datafile_negative)

    positive_text = list(posfile.readlines())
    positive_text = [s.strip() for s in positive_text]
    negative_text = list(negfile.readlines())
    negative_text = [s.strip() for s in negative_text]

    # Split sentences into words by space delimiter.
    x_text = positive_text + negative_text
    x_text = [clean_strings(sentence) for sentence in x_text]
    x_text = [sentence.split(" ") for sentence in x_text]

    # And generate labels as ordered List objects.
    positive_labels = [1 for _ in positive_text]
    negative_labels = [0 for _ in negative_text]

    # Create a vector numpy.ndarray object for labels, to match up with the concatenated x_text vector List.
    y = np.concatenate([positive_labels, negative_labels], 0)

    return [x_text, y]


def _test_load_data_with_labels():
    """
    Expecting output like:

    <class 'list'>
    [["b'this", 'is', 'a', 'positive', 'sentence', 'with', 'happy', "comments!'"],
     ['b', "i'm", 'very', 'pleased', 'with', 'this'],
     ["b'this", 'is', 'a', 'negative', 'sentence', 'with', 'bad', 'comments', "'"],
     ["b'this", 'is', 'not', 'so', 'good', "'"]]
    <class 'numpy.ndarray'>
    array([1, 1, 0, 0])

    :return:
    """
    [x, y] = load_data_with_labels()
    print(type(x))
    pprint(x)
    print(type(y))
    pprint(y)


if __name__ == '__main__':
    _test_load_data_with_labels()
