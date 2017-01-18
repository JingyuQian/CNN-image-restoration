import numpy
import timeit
import time
import inspect
import sys
import os
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.shared_randomstreams import RandomStreams
from scipy.ndimage.interpolation import rotate

from hw3_utils import shared_dataset, load_data
from hw3_nn import LogisticRegression, HiddenLayer, LeNetConvPoolLayer, DropoutHiddenLayer, drop, LeNetConvLayer, LeNetPoolLayer, bn_layer, upsampling, train_nn

#Problem 1
#Implement the convolutional neural network architecture depicted in HW3 problem 1
#Reference code can be found in http://deeplearning.net/tutorial/code/convolutional_mlp.py
def test_lenet(learning_rate=0.1, n_epochs=200,
               nkerns=[32,64], batch_size=500):
    rng = numpy.random.RandomState(12345)

#    ds_rate=2
    datasets = load_data(ds_rate=None,theano_shared=True)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    time.sleep(2)
    print('... building the model')
    
    layer0_input = x.reshape((batch_size,3,32,32))
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (32-3+1 , 32-3+1) = (30, 30)
    # maxpooling reduces this further to (30/2, 30/2) = (15, 15)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 15, 15)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size,3,32,32),
        filter_shape=(nkerns[0],3,3,3),
        poolsize=(2,2)
    )
    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (15-3+1, 15-3+1) = (13, 13)
    # maxpooling reduces this further to (13/2, 13/2) = (6, 6)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 6, 6)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 15, 15),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(2, 2)
    )

    layer2_input = layer1.output.flatten(2)

    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=2304,
        n_out=4096,
        activation=T.tanh)
    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=4096,
        n_out=512,
        activation=T.tanh)

    layer4 = LogisticRegression(input=layer3.output,n_in=512,n_out=10)

    cost = layer4.negative_log_likelihood(y)

    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # create a list of all model prameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    train_nn(train_model, validate_model, test_model,
             n_train_batches, n_valid_batches, n_test_batches, n_epochs)

#Problem 2.1
#Write a function to add translations
def translate_image(rng, batch_x):
    tx = rng.uniform(low=-2,high=2,dtype='int32')
    ty = rng.uniform(low=-2,high=2,dtype='int32')
    trans = T.roll(batch_x,tx,axis=2)
    trans = T.roll(trans,ty,axis=3)
    trans = T.switch(T.ge(tx,0),T.set_subtensor(trans[:,:,:tx,:],0),T.set_subtensor(trans[:,:,tx:,:],0))
    trans = T.switch(T.ge(ty,0),T.set_subtensor(trans[:,:,:,:ty],0),T.set_subtensor(trans[:,:,:,ty:],0))
    return trans
#Implement a convolutional neural network with the translation method for augmentation
def test_lenet_translation(learning_rate=0.1, n_epochs=200,
                   nkerns=[32,64], batch_size=500):
    rng = numpy.random.RandomState(23455)
    srng = RandomStreams(seed=23455)
    datasets = load_data(ds_rate=None,theano_shared=True)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    time.sleep(2)
    print('... building the model')
    

    layer0_input = translate_image(srng,x.reshape((batch_size,3,32,32)))
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (32-3+1 , 32-3+1) = (30, 30)
    # maxpooling reduces this further to (30/2, 30/2) = (15, 15)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 15, 15)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size,3,32,32),
        filter_shape=(nkerns[0],3,3,3),
        poolsize=(2,2)
    )
    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (15-3+1, 15-3+1) = (13, 13)
    # maxpooling reduces this further to (13/2, 13/2) = (6, 6)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 6, 6)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 15, 15),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(2, 2)
    )

    layer2_input = layer1.output.flatten(2)

    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=2304,
        n_out=4096,
        activation=T.tanh)
    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=4096,
        n_out=512,
        activation=T.tanh)

    layer4 = LogisticRegression(input=layer3.output,n_in=512,n_out=10)

    cost = layer4.negative_log_likelihood(y)

    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # create a list of all model prameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    train_nn(train_model, validate_model, test_model,
             n_train_batches, n_valid_batches, n_test_batches, n_epochs)

#Problem 2.2
#Write a function to add roatations
def rotate_image(batch_x):
    temp = batch_x.reshape(500,3,32,32)
    deg=numpy.random.uniform(low=-7,high=7)
    rot_x=rotate(temp,deg,axes=(2,3),reshape=False)
    rot_x=rot_x.reshape(500,3072)
    return rot_x
#Implement a convolutional neural network with the rotation method for augmentation
def test_lenet_rotation(learning_rate=0.1, n_epochs=200,
                   nkerns=[32,64], batch_size=500, verbose=True):
    rng = numpy.random.RandomState(23455)
    datasets = load_data(ds_rate=None,theano_shared=False)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    n_train_batches = train_set_x.shape[0]
    n_valid_batches = valid_set_x.shape[0]
    n_test_batches = test_set_x.shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    time.sleep(2)
    print('... building the model')
    

    layer0_input = x.reshape((batch_size,3,32,32))
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (32-3+1 , 32-3+1) = (30, 30)
    # maxpooling reduces this further to (30/2, 30/2) = (15, 15)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 15, 15)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size,3,32,32),
        filter_shape=(nkerns[0],3,3,3),
        poolsize=(2,2)
    )
    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (15-3+1, 15-3+1) = (13, 13)
    # maxpooling reduces this further to (13/2, 13/2) = (6, 6)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 6, 6)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 15, 15),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(2, 2)
    )

    layer2_input = layer1.output.flatten(2)

    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=2304,
        n_out=4096,
        activation=T.tanh)
    
    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=4096,
        n_out=512,
        activation=T.tanh)

    layer4 = LogisticRegression(input=layer3.output,n_in=512,n_out=10)

    cost = layer4.negative_log_likelihood(y)

    test_model = theano.function(
        [x,y],
        layer4.errors(y),
    )

    validate_model = theano.function(
        [x,y],
        layer4.errors(y),
    )
    # create a list of all model prameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [x,y],
        cost,
        updates=updates
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
            cost_ij = train_model(rotate_image(train_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]).astype(numpy.float32),
                                   train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size].astype(numpy.int32))
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(
                              valid_set_x[i * batch_size: (i + 1) * batch_size].astype(numpy.float32),
                              valid_set_y[i * batch_size: (i + 1) * batch_size].astype(numpy.int32))
                              for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                if verbose:
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                        (epoch,minibatch_index + 1,n_train_batches,this_validation_loss * 100.))
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(test_set_x[i * batch_size: (i + 1) * batch_size].astype(numpy.float32),
                               test_set_y[i * batch_size: (i + 1) * batch_size].astype(numpy.int32))
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)

                    if verbose:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1,
                               n_train_batches,
                               test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    # Print out summary
    print('Optimization complete.')
    print('Best validation error of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The training process for function ran for %.2fm' % ((end_time - start_time) / 60.)))

#Problem 2.3
#Write a function to flip images
def flip_image(srng,batch_x):
    test=srng.uniform(low=-1,high=1,dtype='float32')
    output = T.switch(T.ge(test,0),batch_x,batch_x[:,:,:,::-1])
    return output
#Implement a convolutional neural network with the flip method for augmentation
def test_lenet_flip(learning_rate=0.1, n_epochs=200,
              nkerns=[32,64], batch_size=500):
    rng = numpy.random.RandomState(23455)
    srng = RandomStreams(seed=23455)
    datasets = load_data(ds_rate=None,theano_shared=True)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    time.sleep(2)
    print('... building the model')
    

    layer0_input = flip_image(srng,x.reshape((batch_size,3,32,32)))
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (32-3+1 , 32-3+1) = (30, 30)
    # maxpooling reduces this further to (30/2, 30/2) = (15, 15)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 15, 15)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size,3,32,32),
        filter_shape=(nkerns[0],3,3,3),
        poolsize=(2,2)
    )
    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (15-3+1, 15-3+1) = (13, 13)
    # maxpooling reduces this further to (13/2, 13/2) = (6, 6)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 6, 6)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 15, 15),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(2, 2)
    )

    layer2_input = layer1.output.flatten(2)

    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=2304,
        n_out=4096,
        activation=T.tanh)
    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=4096,
        n_out=512,
        activation=T.tanh)

    layer4 = LogisticRegression(input=layer3.output,n_in=512,n_out=10)

    cost = layer4.negative_log_likelihood(y)

    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # create a list of all model prameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    train_nn(train_model, validate_model, test_model,
             n_train_batches, n_valid_batches, n_test_batches, n_epochs)
    
#Problem 2.4
#Write a function to add noise, it should at least provide Gaussian-distributed and uniform-distributed noise with zero mean
def noise_injection(srng,batch_x):
    noise = srng.normal(size=(500,3,32,32),std=0.01)*0.25
    output = noise+batch_x
    return output
#Implement a convolutional neural network with the augmentation of injecting noise into input
def test_lenet_inject_noise_input(learning_rate=0.05, n_epochs=200,
                   nkerns=[32,64], batch_size=500):
    rng = numpy.random.RandomState(64724)
    srng = RandomStreams(seed=23455)
    datasets = load_data(ds_rate=None,theano_shared=True)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    time.sleep(2)
    print('... building the model')

    layer0_input = noise_injection(srng,x.reshape((batch_size,3,32,32)))
#    layer0_input = x.reshape((batch_size,3,32,32))
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (32-3+1 , 32-3+1) = (30, 30)
    # maxpooling reduces this further to (30/2, 30/2) = (15, 15)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 15, 15)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size,3,32,32),
        filter_shape=(nkerns[0],3,3,3),
        poolsize=(2,2)
    )
    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (15-3+1, 15-3+1) = (13, 13)
    # maxpooling reduces this further to (13/2, 13/2) = (6, 6)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 6, 6)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 15, 15),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(2, 2)
    )

    layer2_input = layer1.output.flatten(2)

    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=2304,
        n_out=4096,
        activation=T.tanh)
    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=4096,
        n_out=512,
        activation=T.tanh)

    layer4 = LogisticRegression(input=layer3.output,n_in=512,n_out=10)

    cost = layer4.negative_log_likelihood(y)

    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # create a list of all model prameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    train_nn(train_model, validate_model, test_model,
             n_train_batches, n_valid_batches, n_test_batches, n_epochs)
    
#===================================Problem 3=========================================#
def transform(srng,batch_x):
    temp = flip_image(srng,batch_x)
    return temp
def RMSprop(cost, params, lr=0.0005, rho=0.85, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
#        v = theano.shared(numpy.random.rand(1))
#        p = p + alpha * v
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
#        updates.append((v, v - lr * g))
        updates.append((p, p - lr * g))
    return updates
def MY_lenet(n_epochs=400,
         nkerns=[64,128],batch_size=500,
         p=0.7, lr=0.001, rho=0.9,verbose=True):
    rng = numpy.random.RandomState(23455)
    srng = RandomStreams(23455)
    datasets = load_data(ds_rate=None,theano_shared=False)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    train_set_x = numpy.concatenate((train_set_x, valid_set_x[:-5 * batch_size]))
    train_set_y = numpy.concatenate((train_set_y, valid_set_y[:-5 * batch_size]))
    valid_set_x = valid_set_x[-5 * batch_size:]
    valid_set_y = valid_set_y[-5 * batch_size:]
#    valid_set_x = valid_set_x[-10*batch_size:]
#    valid_set_y = valid_set_y[-10*batch_size:]

    train_set_x, train_set_y = shared_dataset([train_set_x, train_set_y])
    valid_set_x, valid_set_y = shared_dataset([valid_set_x, valid_set_y])
    test_set_x, test_set_y = shared_dataset([test_set_x, test_set_y])
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_valid_batches //= batch_size
    n_train_batches //= batch_size
    n_test_batches //= batch_size
    
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    training_enabled = T.iscalar('training_enabled')
    print('... building the model')

    layer0_input = transform(srng,x.reshape((batch_size,3,32,32)))
    
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size,3,32,32),
        filter_shape=(nkerns[0],3,3,3),
        poolsize = (3,3),
        activation=T.nnet.sigmoid
    )
# 500,64,10,10
    layer_bn0 = bn_layer(layer0.output,[500,64,10,10])
# 500,64,10,10    
    layer1 = LeNetConvLayer(
        rng,
        input=layer_bn0.output,
        image_shape=(batch_size, nkerns[0], 10, 10),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        activation=T.nnet.sigmoid
    )
#500,128,8,8
    layer_bn1 = bn_layer(layer1.output,[500,128,8,8])
#500,128,8,8
    layer_pool1 = LeNetPoolLayer(
        input = layer_bn1.output,
        poolsize = (2,2)
    )
#500,128,4,4   
    layer2_input = layer_pool1.output.flatten(2)

    layer2 = DropoutHiddenLayer(
        rng,
        is_train=training_enabled,
        input=layer2_input,
        n_in=2048,
        n_out=4096,
        activation=T.nnet.sigmoid,
        p=p
        )
    layer3 = DropoutHiddenLayer(
        rng,
        is_train=training_enabled,
        input=layer2.output,
        n_in=4096,
        n_out=512,
        activation=T.nnet.sigmoid,
        p=p
    )

    layer4 = LogisticRegression(input=layer3.output,n_in=512,n_out=10)

    cost = layer4.negative_log_likelihood(y)

    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](0)}
    )
    
    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](0)}
    )
    # create a list of all model prameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer_bn1.params + layer1.params + layer_bn0.params + layer0.params
    grads = T.grad(cost,params)
    
    updates = RMSprop(cost, params,lr=lr,rho=rho)
    
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](1)}
    )
    print('... training')
    train_nn(train_model, validate_model, test_model,
             n_train_batches, n_valid_batches, n_test_batches, n_epochs)
#================Problem 4==================================#
def MY_CNN(n_epochs=128, batch_size=500, verbose=True):
    rng = numpy.random.RandomState(23455)
    srng = RandomStreams(23455)
    datasets = load_data(ds_rate=None,theano_shared=False)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
# Use first 50000 for training and 10000 for validation. No test.
    train_set_x = numpy.concatenate((train_set_x, valid_set_x))
    train_set_y = numpy.concatenate((train_set_y, valid_set_y))
    valid_set_x = test_set_x
    valid_set_y = test_set_y
# Pre-store the ground truth for plotting
    ground_truth = valid_set_x[0:8]
    
    train_set_x, train_set_y = shared_dataset([train_set_x, train_set_y])
    valid_set_x, valid_set_y = shared_dataset([valid_set_x, valid_set_y])
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
#    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_valid_batches //= batch_size
    n_train_batches //= batch_size
#    n_test_batches //= batch_size

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    print('... building the model')

    layer_corrupt_input = x.reshape((batch_size,3,32,32))
# corrupt input
    layer0_input = drop(layer_corrupt_input, p=0.7)

    corruption = theano.function(
        [index],
        layer0_input,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size]}
    )
    
    layer0 = LeNetConvLayer(
        rng,
        input = layer0_input,
        filter_shape = (64, 3, 3, 3),
        image_shape = (batch_size, 3, 32, 32),
    )
#500,64,32,32
    layer1 = LeNetConvLayer(
        rng,
        input = layer0.output,
        filter_shape = (64, 64, 3, 3),
        image_shape = (batch_size, 64, 32, 32),
    )
#500,64,32,32
    layer2 = LeNetPoolLayer(layer1.output, poolsize = (2,2))
#500,64,16,16
    layer3 = LeNetConvLayer(
        rng,
        input = layer2.output,
        filter_shape = (128, 64, 3, 3),
        image_shape = (batch_size, 64, 16, 16)
   )
#500,128,16,16
    layer4 = LeNetConvLayer(
        rng,
        input = layer3.output,
        filter_shape = (128, 128, 3, 3),
        image_shape = (batch_size, 128, 16, 16)
   )
#500,128,16,16
    layer5 = LeNetPoolLayer(layer4.output, poolsize = (2,2))
#500,128,8,8
    layer6 = LeNetConvLayer(
        rng,
        input = layer5.output,
        filter_shape = (256,128,3,3),
        image_shape = (batch_size, 128,8,8)
    )
#500,256,8,8
    layer7_input = upsampling(layer6.output)
#500,256,16,16
    layer7 = LeNetConvLayer(
        rng,
        input = layer7_input,
        filter_shape = (128,256,3,3),
        image_shape = (batch_size,256,16,16)
    )
#500,128,16,16
    layer8 = LeNetConvLayer(
        rng,
        input = layer7.output,
        filter_shape = (128,128,3,3),
        image_shape = (batch_size, 128,16,16)
    )
#500,128,16,16
    layer9_input =upsampling(layer8.output + layer4.output)
#500,128,32,32
    layer9 = LeNetConvLayer(
        rng,
        input = layer9_input,
        filter_shape = (64,128,3,3),
        image_shape = (batch_size, 128, 32, 32)
    )
#500,64,32,32
    layer10 = LeNetConvLayer(
        rng,
        input = layer9.output,
        filter_shape = (64,64,3,3),
        image_shape = (batch_size,64,32,32)
    )
    layer11_input = layer1.output+layer10.output
#500,64,32,32
    layer11 = LeNetConvLayer(
        rng,
        input =layer11_input,
        filter_shape = (3,64,3,3),
        image_shape = (batch_size,64,32,32)
    )
#500,3,32,32
    cost = T.mean(T.sqr(layer11.output - layer_corrupt_input))
    params = layer11.params + layer10.params + \
    layer9.params + layer8.params + \
    layer7.params + layer6.params + \
    layer4.params + layer3.params + \
    layer1.params + layer0.params

    updates = RMSprop(cost,params, lr = 0.001, rho=0.9)

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]}
    )

    validate_model = theano.function(
        [index],
        cost,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size]}
    )
    
    output_model = theano.function(
        [index],
        layer11.output,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size]}
    )
        

    print('... training')

    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is                   # found
    improvement_threshold = 0.98  # a relative improvement of this much is
                                   # considered significant
    corrupted = corruption(0)
    corrupted_8 = corrupted[0:8]
    validation_frequency = min(n_train_batches, patience // 2)
    best_validation_loss = numpy.inf
    best_iter = 0
    start_time = timeit.default_timer()
    epoch = 0
    done_looping = False
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                if verbose:
                    print('epoch %i, minibatch %i/%i, current MSE %f ' %
                        (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_validation_loss))
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    recovered = output_model(0)
                    recovered_8 = recovered[0:8]
                    plt.figure(figsize = (9,14))   
                    for i in range(24):
                        plt.subplot(8,3,i+1)
                        if(i%3==0):
                            plt.imshow(numpy.reshape(ground_truth[i//3,:],(3,32,32)).transpose(1,2,0))
                            plt.title('Original%d'%(i//3+1))
                        elif(i%3==1):
                            plt.imshow(numpy.reshape(corrupted_8[i//3,:],(3,32,32)).transpose(1,2,0))
                            plt.title('Corrupted%d'%(i//3+1))
                        else:
                            plt.imshow(numpy.reshape(recovered_8[i//3,:],(3,32,32)).transpose(1,2,0))
                            plt.title('Recovered%d'%(i//3+1))
                        plt.xticks([])
                        plt.yticks([])
                        plt.savefig('hw3_4_%d.png'%(best_iter))
                    plt.close()
                if verbose:
                    print('            Best MSE %f'%(best_validation_loss))
                    
            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    # Print out summary
    print('Optimization complete.')
    print('Best MSE of %f %% obtained at iteration %i,'
          %(best_validation_loss, best_iter + 1))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)