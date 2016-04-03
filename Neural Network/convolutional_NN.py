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

#Lasagne Imports
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

#assumes you have Ryans images in the same folder as this script
filename ="test.tif"
training_file = "train.tif"

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
    return image_blocks


def get_labeled_data(filename, training_file):
    """Read input-array (image) and label-images and return it as list of tuples. """

    rows,cols =  load_extension.getDims(filename)
    print rows,cols

    image = np.ones((rows,cols),'uint8')
    label_image = np.ones((rows,cols),'uint8')
    # x is a dummy to use as a form of error checking will return false on error
    x = load_extension.getImage(image ,filename)
    x = load_extension.getTraining(label_image,filename, training_file)

    #Seperate Image and Label into blocks
    test_blocks = create_image_blocks(768, 393,11543,rows,cols,image)
    label_blocks = create_image_blocks(768, 393,11543,rows,cols,label_image)
    #Used to Write image blocks to folder
    #for i in range(blocks):
     #    im = Image.fromarray(test_blocks[i])
      #   im.save(str(i) +"label.tif")
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

#######################################################################################################################
#######################################################################################################################
"""Below is the implementation of the convolutional neural network using the Lasagne library for python"""

#Step1:Load Data
test_blocks, label_blocks = get_labeled_data(filename, training_file)
print test_blocks.shape
print label_blocks.shape





#Step 2 Create Neural Network with 2 Hidden Layers
def convolutionalNeuralNetwork(epochs):
    net = NeuralNet(
        layers=[ #three layers: Input, hidden, and output
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
        ],

        #Parameters for the layers
        input_shape = (None, 2,393,11543), #input pixels per image  block
        conv1_num_filters=32,
        conv1_filter_size=(3, 3),
        pool1_pool_size=(2, 2),
        conv2_num_filters=64,
        conv2_filter_size=(2, 2),
        pool2_pool_size=(2, 2),
        conv3_num_filters=128,
        conv3_filter_size=(2, 2),
        pool3_pool_size=(2, 2),
        hidden4_num_units=50,
        hidden5_num_units=50,
        output_num_units=1,
        output_nonlinearity=None,


        #optimization method:
        update = nesterov_momentum,
        update_learning_rate = 0.01,
        update_momentum = 0.9,

        regression = True,
        max_epochs = epochs,
        verbose = 1,
    )
    return net

#Step 3 Train Neural Net

net = convolutionalNeuralNetwork(40)
net.fit(test_blocks, label_blocks)


#Step 4 Test Neural Network