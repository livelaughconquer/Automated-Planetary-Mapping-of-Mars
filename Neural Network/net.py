"""
The Convolutional Neural Network is modelled after the PyData2015 example
"""

from PIL import Image
import sys
import os
import gzip
import pickle
import theano
import theano.tensor as T
import lasagne
import load_extension
import matplotlib.pyplot as plt
import numpy as np

# Seed for reproduciblity
np.random.seed(42)

# load training and test aplits as np arrays
train, val, test = pickle.load(gzip.open('mnist.pkl.gz'))
X_train, y_train = train
X_val, y_val = val

# for training, we want to sample examples at random in small batches for efficiency
def batch_gen(X, y, N):
    while True:
        idx = np.random.choice(len(y), N)
        yield X[idx].astype('float32'), y[idx].astype('int32')


# reshape from a 1D feature vector to a 1 channel 2D image
# then apply 3 convolutional filters with 3x3 kernal size
l_in = lasagne.layers.InputLayer((None, 768))  # "None" for a more general batch size
l_shape = lasagne.layers.ReshapeLayer(l_in, (-1, 1, 28, 28))  # batch dim, channel dim, image dims
l_conv = lasagne.layers.Conv2DLayer(l_shape, num_filters=3, filter_size=3, pad=1)
# 10 classes for each number (0-9)
l_out = lasagne.layers.DenseLayer(l_conv, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)

# define symbolic variables for matrix
X_sym = T.matrix()
y_sym = T.ivector() # target (vector of ints)

# theano expressions for the output distribution and predicted class
output = lasagne.layers.get_output(l_out, X_sym)
pred = output.argmax(-1)  # get predicted class by taking argmax over final dimension

# the loss function is cross-entropy averaged over a minibatch
loss = T.mean(lasagne.objectives.categorical_crossentropy(output, y_sym))
acc = T.mean(T.eq(pred, y_sym))  # compute accuracy

# retrieve all the trainable parameters in the network
params = lasagne.layers.get_all_params(l_out)

# compute the gradient of the loss function with respect to the parameter
# the stochastic gradient descent produces updates for each parameter
grad = T.grad(loss, params)
updates = lasagne.updates.adam(grad, params, learning_rate=0.005)

# define a training function that will compute the loss and accuracy then apply the updates
f_train = theano.function([X_sym, y_sym], [loss, acc], updates=updates)
# same thing as training function with update (we don't want to train with validation data)
f_val = theano.function([X_sym, y_sym], [loss, acc])
# give predicted class
f_predict = theano.function([X_sym], pred)

# choose a batch size and calculate the number of batches in an "epoch" (one pass through the data)
BATCH_SIZE = 64
N_BATCHES = len(X_train) // BATCH_SIZE
N_VAL_BATCHES = len(X_val) // BATCH_SIZE

# minibatch generators for the training and validation sets
train_batches = batch_gen(X_train, y_train, BATCH_SIZE)
val_batches = batch_gen(X_val, y_val, BATCH_SIZE)

# sample the batch generator here
# plot an image and corresponding label to verify they match
"""
X, y = next(train_batches)
plt.imshow(X[0].reshape((28, 28)), cmap='gray', interpolation='nearest')
print(y[0])
plt.show()
exit()
"""

# for each epoch, we call the training function N_BATCHES times,
# accumulating an estimate of the training loss and accuracy
# then we do the same thing for the validation set
# plotting the ratio of val to train loss can help recognize overfitting.
for epoch in range(5):
    train_loss = 0
    train_acc = 0
    for _ in range(N_BATCHES):
        X, y = next(train_batches)
        loss, acc = f_train(X, y)
        train_loss += loss
        train_acc += acc
    train_loss /= N_BATCHES
    train_acc /= N_BATCHES

    val_loss = 0
    val_acc = 0
    for _ in range(N_VAL_BATCHES):
        X, y = next(val_batches)
        loss, acc = f_val(X, y)
        val_loss += loss
        val_acc += acc
    val_loss /= N_VAL_BATCHES
    val_acc /= N_VAL_BATCHES

    print('Epoch {}, Train (val) loss {:.03f} ({:.03f}) ratio {:.03f}'.format(
            epoch, train_loss, val_loss, val_loss/train_loss))
    print('Train (val) accuracy {:.03f} ({:.03f})'.format(train_acc, val_acc))

# we can look at the output after the convolutional layer
filtered = lasagne.layers.get_output(l_conv, X_sym)
f_filter = theano.function([X_sym], filtered)

# filter the first few training examples
im = f_filter(X_train[:10])
print(im.shape)

# rearrange dimension so we can plot the result as RGB images
im = np.rollaxis(np.rollaxis(im, 3, 1), 3, 1)

# we can see that each filter seems different features in the images
# ie horizontal / diagonal / vertical segments
plt.figure(figsize=(16,8))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(im[i], interpolation='nearest')
    plt.axis('off')
    plt.show()
