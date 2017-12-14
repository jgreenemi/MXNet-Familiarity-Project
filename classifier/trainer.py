from .loader import load_and_preprocess
import mxnet as mx
import os
import sys


def train(vocab_size, sentence_size):
    """
    Load data into the algorithm and train on it. In the future this will likely become a method to a class for
    portability.
    :return:
    """

    # Set the batch size and placeholders for network inputs and outputs, to be updated later.
    # This is the size of the batches we'll train the network with. This and the num_embed later
    # should be loaded from a config/passed in params.
    batch_size = 50

    # This will be the input data placeholder.
    input_x = mx.sym.Variable('data')

    # This is the placeholder for output label.
    input_y = mx.sym.Variable('softmax_label')

    # Now we set up the first network layer (embedding)
    # Create an embedding layer to learn the representation of words in a lower dimensional subspace, akin to word2vec.
    # This is the number of dimensions to embed the words into. Appears to have been arbitrarily chosen.
    num_embed = 300

    embed_layer = mx.sym.Embedding(
        data=input_x,
        input_dim=vocab_size,
        output_dim=num_embed,
        name='vocab_embed'
    )

    # Reshape the embedded data for the next layer.
    conv_input = mx.sym.Reshape(
        data=embed_layer,
        target_shape=(batch_size, 1, sentence_size, num_embed)
    )

    return


def _test_train():
    try:
        train(
            vocab_size=5,
            sentence_size=10
        )
        return [True, '']
    except Exception as e:
        return [False, '_test_train: {}'.format(e)]


if __name__ == '__main__':
    # Run unit tests and raise an Exception if any fail.
    testresult = []
    testresult.append(_test_train())
    for testresult, testmsg in testresult:
        if not testresult:
            raise Exception('Tests did not pass: {}'.format(testmsg))

    # At this point all unit tests have completed, run main script.
    prepro_data = load_and_preprocess()
    train(
        vocab_size=prepro_data['vocab_size'],
        sentence_size=prepro_data['sentence_data']
    )
