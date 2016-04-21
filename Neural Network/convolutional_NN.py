__author__ = 'Team Strata'
"""Convolutional Neural Network prototype - The prototype creates test and training data for a Neurial Network in order to detect sand dunes"""

import numpy
import glob, os
import load_extension
import numpy as np
import matplotlib.pyplot as plt
import datetime
from PIL import Image
from StringIO import StringIO
from math import sqrt
import matplotlib.cm as cm
import pylab
import cPickle as pickle

#Lasagne Imports
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

#Function:Separate image into image blocks to use for training data
#       Parameters:
#           row_pixel - number of pixels in a row of the new image blocks
#           col_pixel - number of pixels in a column of a new image block
#           blocks - number of desired image blocks
#           og_pixel - number of pixels in the original image Note: Original row pixels first, then original column pixels

def create_image_blocks(blocks, row_pixel, col_pixel,og_row_pixel, og_col_pixel,image):
    """Create a Numpy array to hold each image block"""

    image_blocks = np.ones((blocks,row_pixel,col_pixel),'uint8')
    #Variables to keep track of image slicing
    block = 0
    row = row_pixel
    #Nested sloop that creates the image splicing and writes to Numpy array
    for i in range(0, (og_row_pixel/row_pixel)):
        col = col_pixel
        for j in range(0, (og_col_pixel/col_pixel)):
            image_blocks[block] = image[row-row_pixel:row,col-col_pixel:col]
            block += 1
            col += col_pixel
        row += row_pixel

    #print image_blocks
    #Return numpy array with Image blocks arrays
    return image_blocks, blocks



def get_labeled_data(filename, training_file, block_size=32):
    """Read input-array (image) and label-images and return it as list of tuples. """

    rows,cols =  load_extension.getDims(filename)
    print rows,cols

    image = np.ones((rows,cols),'uint8')
    label_image = np.ones((rows,cols),'uint8')
    # x is a dummy to use as a form of error checking will return false on error
    x = load_extension.getImage(image ,filename)
    x = load_extension.getTraining(label_image,filename, training_file)
    X = []
    y = []
    for i in xrange(0,rows,block_size):
        for j in xrange(0,cols,block_size):
            try:
                X.append(image[i:i + block_size, j:j + block_size].reshape(1, block_size * block_size))
                y.append(int(load_extension.getLabel(label_image[i:i + block_size, j:j + block_size], "1", "0", 0.75)))
            except ValueError:
                continue

    X = np.array(X).astype(np.float32)
    label_blocks = np.array(y).astype(np.int32)
    test_blocks = X.reshape(-1, 1, block_size, block_size)
#Seperate Image and Label into blocks
    #test_blocks,blocks = create_image_blocks(768, 393,11543,rows,cols,image)
    #label_blocks, blocks = create_image_blocks(768, 393,11543,rows,cols,label_image)
    # test_blocks,blocks = load4d(4096, 8, 8,rows,cols,image)
    # label_blocks, blocks = load4d(4096, 8,8,rows,cols,label_image)
    #Used to Write image blocks to folder
    #or i in range(blocks):
         #im = Image.fromarray(test_blocks[i][i])
         #im.save(str(i) +"label.tif")
    return test_blocks, label_blocks

def view_data(block_number):
    """View Image with labeled image"""
    figure_1 = 'test/' + str(block_number-1) + '.tif'
    figure_2 = 'labels/' + str(block_number-1) + 'label.tif'
    print figure_1
    print figure_2
    f = pylab.figure()
    for i, fname in enumerate((figure_1, figure_2)):
        image = Image.open(fname).convert("L")
        arr = np.asarray(image)
        f.add_subplot(2, 1, i)
        pylab.imshow(arr, cmap=cm.Greys_r)
    pylab.show()

# Load stored model

#with open('net.pickle', 'rb') as f:
 #   net_pretrain = pickle.load(f)

#net_pretrain.max_epochs = 25  # Train the previous model over more epochs

def convolutionalNeuralNetwork(epochs):
    net = NeuralNet(
        layers=[ #three layers: Input, hidden, and output
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            #('conv3', layers.Conv2DLayer),
            #('pool3', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('dense', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
        ],

        #Parameters for the layers
        #input layer
        input_shape = (None, 1, 32, 32), #input pixels per image  block
        #convolutional 1
        conv1_num_filters=32,
        conv1_filter_size=(5, 5),
        conv1_nonlinearity=lasagne.nonlinearities.rectify,
        conv1_W=lasagne.init.GlorotUniform(),
        #maxpool 1
        pool1_pool_size=(2, 2),
        #convolutional 2
        conv2_num_filters=32,
        conv2_filter_size=(5, 5),
        conv2_nonlinearity=lasagne.nonlinearities.rectify,
        conv2_W=lasagne.init.GlorotUniform(),
        #maxpool 2
        pool2_pool_size=(2, 2),
        #dropout 1
        dropout1_p=0.5,
        # dense
        dense_num_units=256,
        dense_nonlinearity=lasagne.nonlinearities.rectify,
        # dropout2
        dropout2_p=0.5,
        # output
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=2,

        #optimization method:
        update = nesterov_momentum,
        update_learning_rate = 0.01,
        update_momentum = 0.9,
        #eval_size = .2,
        #regression = True,
        max_epochs = epochs,
        verbose = 1,
    )
    return net

"""Creates image blocks in shape (batch_size, channels, num_rows, num_columns). For test input in a 2D convolutional layer """
def load4d(blocks, row_pixel, col_pixel,og_row_pixel, og_col_pixel,image):
    image_blocks = np.ones((blocks,1,row_pixel,col_pixel),'uint8')
    #Variables to keep track of image slicing
    block = 0
    row = row_pixel
    #Nested sloop that creates the image splicing and writes to Numpy array
    for i in range(0, (og_row_pixel/row_pixel)):
        col = col_pixel
        for j in range(0, (og_col_pixel/col_pixel)):
            image_blocks[block] = image[row-row_pixel:row,col-col_pixel:col]
            block += 1
            col += col_pixel
        row += row_pixel

    #print image_blocks
    #Return numpy array with Image blocks arrays
    return image_blocks, blocks



#######################################################################################################################
#######################################################################################################################
"""Below is the implementation of the convolutional neural network using the Lasagne library for python"""

#Step1:Load Data
#assumes you have Ryans images in the same folder as this script
filename ="test.tif"
training_file = "train.tif"
test_blocks, label_blocks = get_labeled_data(filename, training_file)
X = test_blocks
print label_blocks
print test_blocks.shape, label_blocks.shape

ones = 0
zeroes = 0
for i in range(label_blocks.shape[0]):
    if label_blocks[i] == 1:
        ones+=1
    elif label_blocks[i] == 0:
        zeroes += 1

print ones, zeroes

#labels = getLabel(training_file)
#print labels
#print test_blocks
# print label_blocks.shape

#Reshape data into 2D
#test_blocks = test_blocks.reshape(-1, 4096, 8, 8)
#label_blocks = label_blocks.reshape(-1, 4096, 8, 8)

#print test_blocks[40]

#Step 2 Create Neural Network with 2 Hidden Layers
net = convolutionalNeuralNetwork(25)

#Step 3 Train Neural Net

train = net.fit(test_blocks, label_blocks)
#import pickle to store neural net training
#train = net_pretrain.fit(test_blocks, label_blocks) #Train pre-trained model more

#Store the trained model
#with open('net.pickle', 'wb') as f:
    #pickle.dump(net, f, -1)



#Step 4 Look at Predictions from neural network

y_pred = net.predict(X)

#Checking to see if predictions detect any sand dunes. Outputs the indice where a 1 is found.
ones = 0
zeroes = 0
array_dunes = []
for i in range(y_pred.shape[0]):
    if y_pred[i] == 1:
        ones+=1
        array_dunes.append(i)
    elif y_pred[i] == 0:
        zeroes += 1

print ones, zeroes

for j in range(len(array_dunes)):
    print array_dunes[j]
    #print y_pred[i]

#Plot
#im = Image.fromarray(y_pred)
#im.save("label.tif")