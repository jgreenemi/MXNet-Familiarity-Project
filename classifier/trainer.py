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

    # Create convolution + max pooling layer combination for each filter operation.
    filter_list=[3, 4, 5]
    num_filter=100
    pooled_outputs = []

    # This bit is not explained deeply in the tutorial.
    # TODO need to review MXNet reference for mx.sym.Convolution/Activation/Pooling.
    for i, filter_size in enumerate(filter_list):
        convi = mx.sym.Convolution(
            data=conv_input,
            kernel=(filter_size, num_embed),
            num_filter=num_filter
        )
        relui = mx.sym.Activation(
            data=convi,
            act_type='relu'
        )
        pooli = mx.sym.Pooling(
            data=relui,
            pool_type='max',
            kernel=(sentence_size - filter_size + 1, 1),
            stride=(1, 1)
        )
        pooled_outputs.append(pooli)

    # Combine all the pooled outputs.
    total_filters = num_filter * len(filter_list)
    concat = mx.sym.Concat(*pooled_outputs, dim=1)

    # And finally, reshape it for the next layer to use.
    h_pool = mx.sym.Reshape(
        data=concat,
        target_shape=(batch_size, total_filters)
    )

    # Introduce dropout regularization to reduce bias in neurons.
    dropout = 0.5
    if dropout > 0.0:
        h_drop = mx.sym.Dropout(
            data=h_pool,
            p=dropout
        )
    else:
        h_drop = h_pool

    # Now add a fully connected layer. The softmax function applied to the output will yield our classification.
    # In theory, we up the num_label to match how many classes our data could fit into. In this case, we're
    # fitting the data to either a 0 or a 1, so we'll have num_label be equal to 2 for a binary classification.
    num_label = 2
    cls_weight = mx.sym.Variable('cls_weight')
    cls_bias = mx.sym.Variable('cls_bias')

    # TODO Determine if "num_hidden" does indeed just mean how many hidden layers there are to be, meaning we could
    # potentially improve on the accuracy of the algorithm by increasing this value.
    fc = mx.sym.FullyConnected(
        data=h_drop,
        weight=cls_weight,
        bias=cls_bias,
        num_hidden=num_label
    )

    # Determine the softmax output.
    sm = mx.sym.SoftmaxOutput(
        data=fc,
        label=input_y,
        name='softmax'
    )

    # Set the CNN pointer to the "back" of the network.
    cnn = sm

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
