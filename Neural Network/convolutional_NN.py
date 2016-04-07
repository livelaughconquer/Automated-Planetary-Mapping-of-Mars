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
    #test_blocks,blocks = create_image_blocks(768, 393,11543,rows,cols,image)
    #label_blocks, blocks = create_image_blocks(768, 393,11543,rows,cols,label_image)
    test_blocks,blocks = load4d(4096, 8, 8,rows,cols,image)
    label_blocks, blocks = load4d(4096, 8,8,rows,cols,label_image)
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
        input_shape = (None, 1, 8, 8), #input pixels per image  block
        conv1_num_filters=1,
        conv1_filter_size=(4, 4),
        pool1_pool_size=(4, 4),
        conv2_num_filters=4,
        conv2_filter_size=(1, 1),
        pool2_pool_size=(1, 1),
        conv3_num_filters=4,
        conv3_filter_size=(1, 1),
        pool3_pool_size=(1, 1),
        hidden4_num_units=4,
        hidden5_num_units=4,
        output_num_units=1,
        output_nonlinearity=None,


        #optimization method:
        update = nesterov_momentum,
        update_learning_rate = 0.01,
        update_momentum = 0.9,
        eval_size = .2,
        regression = True,
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
filename ="circles.tif"
training_file = "circles_train.tif"
test_blocks, label_blocks = get_labeled_data(filename, training_file)
print test_blocks.shape

#print test_blocks
print label_blocks.shape

#Reshape data into 2D
#test_blocks = test_blocks.reshape(-1, 4096, 8, 8)
#label_blocks = label_blocks.reshape(-1, 4096, 8, 8)

#print test_blocks

#Step 2 Create Neural Network with 2 Hidden Layers
net = convolutionalNeuralNetwork(40)

#Step 3 Train Neural Net

#net.summary()
train = net.fit(test_blocks, label_blocks)

#Step 4 Look at Predictions from neural network
