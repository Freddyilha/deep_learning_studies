import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from regression import LogisticRegression, load_data
from mlp import HiddenLayer

#LOAD PARAMETERS
n_train = 40000                     #Number of training images
n_valid = 5000                      #Number of validation images
n_test  = 5000                     #Number of testing images
img_size = 1024                     #Size of each image channels*heigth*width
dataset_path = "dataset_cifar-10_full.txt"

#FILTER PARAMETERS
feature_maps = 1                    #Image channels
img_h = 32                          #Image height
img_w = 32                          #Image width
n_kerns = [5, 5]                  #Number of kernels in each filter: filter_i --> n_kerns[i]
kern_size = [(5,5), (5,5)]          #Same number as the number of filters
pool_size = [(2,2), (2,2)]          #Same number as the number of filters
n_outputs = 10                      #Number of classifications

#TRAINING PARAMETERS
learning_rate = 0.1
n_epochs_     = 2000
batch_size_   = 50

class LeNetConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):

        assert image_shape[1] == filter_shape[1]

        self.input = input

        fan_in = numpy.prod(filter_shape[1:])

        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(poolsize))

        W_bound = numpy.sqrt(6. / (fan_in + fan_out))

        self.W = theano.shared(numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX), borrow=True)

        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)

        self.b = theano.shared(value=b_values, borrow=True)

        conv_out = conv.conv2d(input=input, filters=self.W, filter_shape=filter_shape, image_shape=image_shape)

        pooled_out = downsample.max_pool_2d(input=conv_out, ds=poolsize, ignore_border=True)

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]

def evaluate_lenet5(learning_rate=learning_rate, n_epochs=n_epochs_, batch_size=batch_size_):
    datasets = load_data(dataset_path, n_train, n_valid, n_test, img_size)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()

    x = T.matrix('x')
    y = T.ivector('y')

    rng = numpy.random.RandomState(23455)

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    layer0_input = x.reshape((batch_size, feature_maps, img_h, img_w))

    layer0 = LeNetConvPoolLayer(rng, input=layer0_input, image_shape=(batch_size, feature_maps, img_h, img_w), filter_shape=(n_kerns[0], feature_maps, kern_size[0][0], kern_size[0][1]), poolsize=pool_size[0])

    #Modify shape from layer 0
    shapeA = (img_h-kern_size[0][0]+1)/pool_size[0][0]
    shapeB = (img_w-kern_size[0][1]+1)/pool_size[0][1]
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output, image_shape=(batch_size, n_kerns[0], shapeA, shapeB), filter_shape=(n_kerns[1], n_kerns[0], kern_size[1][0], kern_size[1][1]), poolsize=pool_size[1])

    layer2_input = layer1.output.flatten(2)             

    #Modify shape from layer 1
    shapeA = (shapeA-kern_size[1][0]+1)/pool_size[1][0]
    shapeB = (shapeB-kern_size[1][1]+1)/pool_size[1][1]
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=n_kerns[1]*shapeA*shapeB, n_out=batch_size, activation=T.tanh)

    layer3 = LogisticRegression(input=layer2.output, n_in=batch_size, n_out=n_outputs)

    cost = layer3.negative_log_likelihood(y)

    test_model = theano.function([index], layer3.errors(y), givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function([index], layer3.errors(y), givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    params = layer3.params + layer2.params + layer0.params

    grads = T.grad(cost, params)

    updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]

    train_model = theano.function([index], cost, updates=updates, givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    patience = 10000
    patience_increase = 2

    improvement_threshold = 0.995

    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                if this_validation_loss < best_validation_loss:

                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
