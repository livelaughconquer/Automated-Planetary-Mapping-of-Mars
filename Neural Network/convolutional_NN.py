__author__ = 'Team Strata'
"""Convolutional Neural Network prototype - The prototype creates test and training data for a Neural Network in order to detect sand dunes"""


import click
import glob, os
import load_extension
import numpy as np

from PIL import Image
import matplotlib.cm as cm
import pylab
import cPickle as pickle
import os.path
import urllib, urllib2

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

    rows,cols = load_extension.getDims(filename)
    print rows,cols

    image = np.ones((rows, cols), 'uint8')
    label_image = np.ones((rows, cols), 'uint8')
    # x is a dummy to use as a form of error checking will return false on error
    x = load_extension.getImage(image, filename)
    x = load_extension.getTraining(label_image, filename, training_file)
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

# Simple function that downloads a hirise image when filename follows naming convention i.e. PSP_009650_1755_RED.JP2
def download_image(filename):
    url = "http://hirise-pds.lpl.arizona.edu/PDS/RDR/"
    filename =filename.upper()
    splitfile =filename.split("_")
    url += splitfile[0] + "/ORB_" + splitfile[1][:-2] + "00_"+splitfile[1][:-2]+"99/"
    url += splitfile[0] + "_" + splitfile[1] + "_" + splitfile[2] + "/" +filename
    try:
        ret = urllib2.urlopen(url)
        urllib.urlretrieve(url,filename)
    except:
        print "File not found online"
    return


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

def getPredictionData(inputFile, block_size=32):
     #Load image using extension
    rows, cols = load_extension.getDims(inputFile)
    print rows, cols
    image = np.ones((rows, cols), 'uint8')
    # x is a dummy to use as a form of error checking will return false on error
    x = load_extension.getImage(image, inputFile)
    X = []
    blocklist = []

    block_size = 32
    for i in xrange(0,rows,block_size):
        for j in xrange(0,cols,block_size):
            try:
                X.append(image[i:i + block_size, j:j + block_size].reshape(1, block_size * block_size))
                blocklist.append(image[i:i + block_size, j:j + block_size])
            except ValueError:
                continue

    X = np.array(X).astype(np.float32)
    X = X.reshape(-1, 1, block_size, block_size)
    load_extension.getImage(image, inputFile)
    return X, image, blocklist

#######################################################################################################################
#######################################################################################################################
"""Below is the implementation of the convolutional neural network using the Lasagne library for python"""

def loadDataset(test, train):
    """  Python Function that loads the testing and training images into numpy arrays. """
    #Step1:Load Data
    filename = test
    print test
    training_file = train
    print train
    click.echo('Loading images....')
    click.echo(' ')
    click.echo('Image dimensions: ')
    test_blocks, label_blocks = get_labeled_data(filename, training_file)

    #click.echo(label_blocks)
    click.echo('')
    click.echo('Shape of test followed by train: ')
    click.echo(test_blocks.shape)
    click.echo(label_blocks.shape)

    ones = 0
    zeroes = 0
    for i in range(label_blocks.shape[0]):
        if label_blocks[i] == 1:
            ones+=1
        elif label_blocks[i] == 0:
            zeroes += 1

    click.echo(' ')
    click.echo('Number of success sand dune blocks followed by failure labels: ')
    click.echo( ones )
    click.echo(zeroes )

    click.echo('Images have been successfully loaded')

    return test_blocks, label_blocks

def trainNetwork(epochs, testFile, trainFile):
    """Function that trains implemented network"""

    test_blocks, label_blocks = loadDataset(testFile, trainFile)

    #Check to see if an existing pickle file exists
    if os.path.isfile('net.pickle'):
        # Load stored model
        with open('net.pickle', 'rb') as f:
            net_pretrain = pickle.load(f)
        # Train the previous model over more epochs
        net_pretrain.max_epochs = epochs

        #Train pre-trained network
        train = net_pretrain.fit(test_blocks, label_blocks)

        #Store the trained model
        with open('net.pickle', 'wb') as f:
            pickle.dump(net_pretrain, f, -1)
        return net_pretrain

    #If pickle does not exist then train network for the first time.
    else:
        #Create Neural Network
        net = convolutionalNeuralNetwork(epochs)
        #Train Neural Net
        train = net.fit(test_blocks, label_blocks)

        #Store the trained model
        with open('net.pickle', 'wb') as f:
            pickle.dump(net, f, -1)
        return net



#Step 4 Look at Predictions from neural network
def makePredictions(inputFile):

    X, image, blocklist = getPredictionData(inputFile)

    click.echo('Loading trained network data....')
    #Load stored data from network
    try:
        with open('net.pickle', 'rb') as f:
            net_pretrain = pickle.load(f)
        net_pretrain.max_epochs = 25  # Train the previous model over more epochs
    except IOError as e:
        print "No trained network is available. Use train command to train first. "

    click.echo('')
    click.echo('Making predictions....')
    #Make predictions
    y_pred = net_pretrain.predict(X)


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

    for x, y in enumerate(y_pred):
        if y == 1:
            blocklist[x][:] = 255

    click.echo('')
    click.echo('Dune blocks detected followed by negative blocks.')
    print ones, zeroes

    click.echo('')
    click.echo('Adding predictions to input image....')
    #Adding predictions to image data
    """
    white_pixels = np.array([255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255])
    for block in array_dunes:
        for i in range(32):
            X[block][0][i] = white_pixels


    #Writing image
    rows, cols = load_extension.getDims(inputFile)
    #nh = rows / 32  # New height
    #nw = X.shape[0] // nh  # New width
    #predictimg = X.reshape(nh, nw, X.shape[2], X.shape[3]).swapaxes(1, 2).reshape(nh*X.shape[2], nw*X.shape[3])
    """
    load_extension.writeImage(image, 'prediction.tif')
    click.echo('')
    click.echo('Writing image to directory....')



##############################################################################################################################
##############################################################################################################################
################################################# Main and UI ################################################################

@click.group()
def userInterface():
    """This program is designed to allow the user to load image data, train a neural network on the image data,
    or make predictions based on the stored neural network data.

    Commands:

    load: Loads image data. Input test and train file as an argument.

    Example: python convolutional_NN.py load testFile.tif trainFile.tif


    train: Input testFile trainFile, and number of epochs to train data.
    Loads image data and trains the convolutional neural network to
    detect sand dunes. Saves trained network on a pickle file.

    Example: python convolutional_NN.py train testFile.tif trainFile.tif --epochs=10


    predict:Using existing trained network pickled data, make predictions
    on pickle data. Input image as an argument.

    Example: python convolutional_NN.py predict inputFile.tif


    """
    pass

@userInterface.command()
#@click.option('--load', is_flag=True,help='Loads image data. Assumes files are in same directory as file and named test.tif and train.tif. ')
@click.argument('testfile')
@click.argument('trainfile')
def load(testfile, trainfile):
    """ Loads and prints image specs. """
    #Convert Unicode to string
    test = str(testfile)
    train = str(trainfile)
    test_blocks, label_blocks = loadDataset(test, train)

@userInterface.command()
@click.argument('testfile')
@click.argument('trainfile')
@click.option('--epochs', default=1 ,help=' Input number of Epochs you would like to train network on.')
def train(testfile, trainfile, epochs):
    """Train convolutional neural network. """
    #Convert Unicode to string
    test = str(testfile)
    train = str(trainfile)
    click.echo('Training network....')
    net = trainNetwork(epochs, test, train)
    click.echo('Training done.')

@userInterface.command()
#@click.option('--predict', is_flag=True, help='Using existing trained network pickled data, make predictions on pickle data.')
@click.argument('input')
def predict(input):
    """Make predictions on input image."""
    input = str(input)
    #click.echo('Making Predictions....')
    makePredictions(input)
    click.echo('Predictions done.')

if __name__ == '__main__':
    """Main Function"""
    userInterface()
    #net = trainNetwork()
    #makePredictions("test.tif")
    #loadDataset("test.tif", "train.tif")