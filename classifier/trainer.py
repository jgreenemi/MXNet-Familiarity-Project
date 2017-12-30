from loader import load_and_preprocess
from collections import namedtuple
from pprint import pprint
import math
import mxnet as mx
import numpy as np
import sys
import time
import traceback


def build_and_train(vocab_size, sentence_size, prepro_data):
    """
    Load data into the algorithm and train on it. In the future this will likely become a method to a class for
    portability.
    :return:
    """

    # Set the output directory for the .params and symbol.json files.
    outdir = './target/'

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

    # Define the structure of the CNN Model as a named tuple.
    CNNModel = namedtuple(
        'CNNModel',
        [
            'cnn_exec',
            'symbol',
            'data',
            'label',
            'param_blocks'
        ]
    )

    # Set what device to train on.
    ctx = mx.gpu(0)
    # Use the following if the training host doesn't have a GPU to use.
    #ctx = mx.cpu(0)

    arg_names = cnn.list_arguments()
    input_shapes = {}
    input_shapes['data'] = (batch_size, sentence_size)

    arg_shape, out_shape, aux_shape = cnn.infer_shape(**input_shapes)

    arg_arrays = [mx.nd.zeros(s, ctx) for s in arg_shape]
    args_grad = {}

    # TODO Not sure what's happening here.
    for shape, name in zip(arg_shape, arg_names):
        # Input, output.
        if name in ['softmax_label', 'data']:
            continue
        args_grad[name] = mx.nd.zeros(shape, ctx)

    cnn_exec = cnn.bind(
        ctx=ctx,
        args=arg_arrays,
        args_grad=args_grad,
        grad_req='add'
    )

    param_blocks = []
    arg_dict = dict(zip(arg_names, cnn_exec.arg_arrays))

    # TODO Does that randomize the initial values? What's the 0.1 value for here?
    initializer = mx.initializer.Uniform(0.1)

    # There's an error when initialization occurs with initializer below. Trying a couple things to get around it.
    #new_initializer = mx.sym.Variable(init=mx.initializer.Uniform(0.1))

    for i, name in enumerate(arg_names):
        #print('i:{}, name:{}'.format(i, name))
        #print('i:{}, arg_dict[{}]:{}'.format(i, name, arg_dict[name]))
        # Input, output.
        if name in ['softmax_label', 'data']:
            continue

        #pprint(arg_dict)

        #initializer.set_verbosity(verbose=True)
        #print(initializer.dumps())

        # Initialization fails here.
        initializer(name, arg_dict[name])

        # TODO what are the capabilities of an append statement like this?
        param_blocks.append(
            (i, arg_dict[name], args_grad[name], name)
        )

    # This does not appear to get used.
    out_dict = dict(zip(cnn.list_outputs(), cnn_exec.outputs))

    data = cnn_exec.arg_dict['data']
    label = cnn_exec.arg_dict['softmax_label']

    cnn_model = CNNModel(
        cnn_exec=cnn_exec,
        symbol=cnn,
        data=data,
        label=label,
        param_blocks=param_blocks
    )


    # Train the CNN model using backpropagation.

    # TODO What are my optimizer options in MXNet?
    optimizer = 'rmsprop'
    max_grad_norm = 5.0
    learning_rate = 0.0005
    epoch = 50

    # Build optimizer.
    opt = mx.optimizer.create(optimizer)
    opt.lr = learning_rate

    updater = mx.optimizer.get_updater(opt)

    # Create logging output. Unnecessary if a proper logging facility is created otherwise.
    logs = sys.stderr

    # For each training epoch
    for iteration in range(epoch):
        tick = time.time()
        num_correct = 0
        num_total = 0

        # Over each batch of the training data
        for begin in range(0, prepro_data['x_train_shape'][0], batch_size):
            batchX = prepro_data['x_train'][begin:begin+batch_size]
            batchY = prepro_data['y_train'][begin:begin+batch_size]
            if batchX.shape[0] != batch_size:
                continue

            cnn_model.data[:] = batchX
            cnn_model.label[:] = batchY

            # Forward propagation.
            # TODO What is the significance of is_train?
            cnn_model.cnn_exec.forward(is_train=True)

            # Backpropagation.
            cnn_model.cnn_exec.backward()

            # Evaluate on the training data.
            num_correct += sum(
                batchY == np.argmax(
                    cnn_model.cnn_exec.outputs[0].asnumpy(),
                    axis=1
                )
            )
            num_total += len(batchY)

            # Update the weights.
            norm = 0
            for idx, weight, grad, name in cnn_model.param_blocks:
                grad /= batch_size
                l2_norm = mx.nd.norm(grad).asscalar()
                norm += l2_norm * l2_norm

            norm = math.sqrt(norm)
            for idx, weight, grad, name in cnn_model.param_blocks:
                if norm > max_grad_norm:
                    grad *= (max_grad_norm / norm)

                updater(idx, grad, weight)

                # Reset the gradient to zero.
                grad[:] = 0.0

        # Decay the learning rate for this epoch to ensure we're not overshooting the optimum.
        if iteration % 50 == 0 and iteration > 0:
            opt.lr *= 0.5
            print('Reset learning rate to {}'.format(opt.lr))

        # End the training loop for this epoch.
        tock = time.time()
        train_time = tock - tick
        if num_correct == 0 or num_total == 0:
            print('ERROR num_correct:{}, num_total:{}'.format(num_correct, num_total))
            train_acc = 0
        else:
            train_acc = num_correct * 100 / float(num_total)

        # Save the checkpoint to disk.
        if (iteration + 1) % 10 == 0:
            prefix = 'cnn'
            cnn_model.symbol.save('{}{}-symbol.json'.format(outdir, prefix))
            save_dict = {
                ('arg:{}'.format(k)): v for k, v in cnn_model.cnn_exec.arg_dict.items()
            }
            save_dict.update(
                {
                    ('aux:{}'.format(k)): v for k, v in cnn_model.cnn_exec.aux_dict.items()
                }
            )
            # Tutorial gives different syntax for this, could influence format in the filename here.
            param_name = '{}{}-{}.params'.format(outdir, prefix, iteration)
            mx.nd.save(param_name, save_dict)
            print('Saved checkpoint to {}'.format(param_name))

        # Evaluate the model after this epoch on the eval set.
        num_correct = 0
        num_total = 0

        # For each test batch.
        for begin in range(0, prepro_data['x_eval'].shape[0], batch_size):
            batchX = prepro_data['x_eval'][begin:begin+batch_size]
            batchY = prepro_data['y_eval'][begin:begin+batch_size]

            if batchX.shape[0] != batch_size:
                continue

            cnn_model.data[:] = batchX

            # Forward propagation again!
            cnn_model.cnn_exec.forward(is_train=False)

            num_correct += sum(
                batchY == np.argmax(
                    cnn_model.cnn_exec.outputs[0].asnumpy(),
                    axis=1
                )
            )
            num_total += len(batchY)


        if num_correct == 0 or num_total == 0:
            print('ERROR Evaluation: num_correct:{}, num_total:{}'.format(num_correct, num_total))
            evaluation_accuracy = 0
        else:
            evaluation_accuracy = num_correct * 100 / float(num_total)
        print('Iteration [%d] Train: Time: %.3fs, Training Accuracy: %.3f ' \
                       'Evaluation Accuracy thus far: %.3f' % (
            iteration,
            train_time,
            train_acc,
            evaluation_accuracy))

    return

####
# From here down are the unit test functions.
####


def _test_build_and_train():
    try:
        # TODO fabricate this dictionary for the unit test to use.
        prepro_data = {
            'THIS DICT NEEDS TO BE POPULATED.': ''
        }
        build_and_train(
            vocab_size=5,
            sentence_size=10,
            prepro_data=prepro_data
        )
        return [True, '']
    except Exception as e:
        e_extended = traceback.format_exc()
        return [False, '_test_train: {}\n{}'.format(e, e_extended)]


if __name__ == '__main__':
    # Run unit tests and raise an Exception if any fail.
    testresult = []
    # Disabling this until I can fabricate the prepro_data dictionary.
    #testresult.append(_test_build_and_train())
    for testresult, testmsg in testresult:
        if not testresult:
            raise Exception('Tests did not pass: {}'.format(testmsg))

    # At this point all unit tests have completed, run main script.
    prepro_data = load_and_preprocess()
    # TODO Should just get the vocab/sentence_size from prepro_data within build_and_train.
    build_and_train(
        vocab_size=prepro_data['vocab_size'],
        sentence_size=prepro_data['sentence_size'],
        prepro_data=prepro_data
    )
