# Pull in text data, normalize it for processing.

import itertools
import numpy as np
import re
from collections import Counter
from pprint import pprint
from urllib.request import urlopen

# Setting some constants here.
# Set this to False to reduce output.
print_to_stdout = True

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
    sentences = positive_text + negative_text
    sentences = [clean_strings(sentence) for sentence in sentences]
    sentences = [sentence.split(" ") for sentence in sentences]

    # And generate labels as ordered List objects.
    positive_labels = [1 for _ in positive_text]
    negative_labels = [0 for _ in negative_text]

    # Create a vector numpy.ndarray object for labels, to match up with the concatenated sentences vector List.
    labels = np.concatenate([positive_labels, negative_labels], 0)

    return [sentences, labels]


def pad_sentences(sentences, padding_word=''):
    """
    Pad all sentences to the same length for processing, matching the length of the longest sentence.
    :param sentences:
    :param padding_word:
    :return:
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []

    for sentence in sentences:
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)

    return padded_sentences


def build_vocab(sentences):
    """
    Builds vocabulary mapping from word to index, based on the sentences passed in.
    Return both the vocabulary mapping and inverse vocabulary mapping.
    The sentences passed in should already be padded to a uniform length.
    :param sentences:
    :return:
    """

    # Build vocab using Counter dict subclass.
    word_counts = Counter(itertools.chain(*sentences))

    # Map word from its index.
    vocabulary_inverse = [x[0] for x in word_counts.most_common()]

    # Map index from its word.
    vocabulary = {x: i for i, x in enumerate(vocabulary_inverse)}

    return [vocabulary, vocabulary_inverse]


def build_input_data(sentences, labels, vocabulary):
    """
    Map sentences and labels to vectors based on a vocabulary.
    :param sentences:
    :param labels:
    :param vocabulary:
    :return:
    """

    # Think I can make this more readable.
    x = np.array([
        [vocabulary[word] for word in sentence] for sentence in sentences
    ])

    y = np.array(labels)

    return [x, y]


def load_and_preprocess():
    """
    Load and preprocess the data based on the configured input files. Return the input vectors, the labels,
    vocabulary and inverse vocabulary.

    This is the major/main function of this script.
    :return:
    """

    # First, load and preprocess the data to get it ready for use in our algorithm.
    sentences, labels = load_data_with_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inverse = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)

    vocab_size = len(vocabulary)

    # Randomly shuffle the data to avoid ordering bias in dividing training and evaluation sets.
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split the training and evaluation sets.
    # *_train will use all items from the first to the 1000th from the end.
    # *_eval will use all items from the 1000th from the end, to the end.
    # This can be improved by using cross-validation, or at least a percentage-based split.
    x_train, x_eval = x_shuffled[:-1000], x_shuffled[-1000:]
    y_train, y_eval = y_shuffled[:-1000], y_shuffled[-1000:]

    # Set the sentence max length value for sentence_size.
    sentence_size = x_train.shape[1]

    # If printing output was set, print. Otherwise build the dictionary and return it.

    resulting_data = {
        'Train/Eval Split': '{}/{}'.format(len(y_train), len(y_eval)),
        'x_train': x_train,
        'x_eval': x_eval,
        'y_train': y_train,
        'y_eval': y_eval,
        'x_train_shape': x_train.shape,
        'x_eval_shape': x_eval.shape,
        'vocab_size': vocab_size,
        'sentence_size': sentence_size
    }

    if print_to_stdout:
        for k, v in resulting_data.items():
            # Don't print the full collections for x/y_train.
            if k not in ['x_train', 'y_train', 'x_eval', 'y_train']:
                print('{}: {}'.format(k, v))

    return resulting_data


####
# From here down are the unit test functions.
####


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

    try:
        [x, y] = load_data_with_labels()

        # Uncomment to print the results.
        #print(type(x))
        #pprint(x)
        #print(type(y))
        #pprint(y)
        return [True, '']

    except Exception as e:
        return [False, '_test_load_data_with_labels: {}'.format(e)]


def _test_build_input_data():
    try:
        #print('Loading data with labels.')
        sentences, labels = load_data_with_labels()
        #print('Building vocab.')
        vocabulary, vocab_inv = build_vocab(sentences)
        #print('Building input.')
        x, y = build_input_data(sentences, labels, vocabulary)

        # Uncomment to print the results.
        #print(type(x))
        #pprint(x)
        #print(type(y))
        #pprint(y)

        #print('Done!')
        return [True, '']

    except Exception as e:
        return [False, '_test_build_input_data: {}'.format(e)]


if __name__ == '__main__':
    # Run unit tests and raise an Exception if any fail.
    testresult = []
    testresult.append(_test_load_data_with_labels())
    testresult.append(_test_build_input_data())
    for testresult, testmsg in testresult:
        if not testresult:
            raise Exception('Tests did not pass: {}'.format(testmsg))

    # At this point all unit tests have completed, run main script.
    load_and_preprocess()
