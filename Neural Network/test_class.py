__author__ = 'Jorge Felix from TEAM STRATA'

#Small Prototype of a Deep Neural Network that classifies images with white pixels, Not all the code was written by TEAM STRATA. A tutorial was used found at https://martin-thoma.com/classify-mnist-with-pybrain/

#Import of Libraries
import numpy
import glob, os
from sklearn.metrics import accuracy_score
import load_extension
import numpy as np
import matplotlib.pyplot as plt
import datetime
from PIL import Image
from StringIO import StringIO
from math import sqrt
import matplotlib.cm as cm
import pylab
from numpy import ravel
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader

#assumes you have Ryans images in the same folder as this script
train_file ="train_file.tif"
test_file = "test_file.tif"

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
            col += row_pixel
        row += row_pixel

    #print image_blocks
    #Return numpy array with Image blocks arrays
    return image_blocks,blocks


def get_labeled_data(filename, training_file):
    """Read input-array (image) and label-images and return it as list of tuples. """

    rows,cols =  load_extension.get_dims(filename)
    print rows,cols

    image = np.ones((rows,cols),'uint8')
    label_image = np.ones((rows,cols),'uint8')
    # x is a dummy to use as a form of error checking will return false on error
    x = load_extension.getImage(image ,filename)
    x = load_extension.getTraining(label_image,filename, training_file)

    #Seperate Image and Label into blocks
    test_blocks, blocks = create_image_blocks(64,64,64,rows,cols,image)
    label_blocks, blocks = create_image_blocks(64,64,64,rows,cols,label_image)
    #Used to Write image blocks to folder
    # for i in range(blocks):
    #     im = Image.fromarray(test_blocks[i])
    #     im.save(str(i) +"label.tif")
    return test_blocks, label_blocks

def view_data(block_number):
    """View Image with label"""
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

def neural_network(input, hidden, output):
    #Creates Network object
    net = FeedForwardNetwork()
    #Create Input, hidden, and output layers
    inLayer = LinearLayer(input)
    hiddenLayer = SigmoidLayer(hidden)
    outLayer = LinearLayer(output)
    #Add to Network
    net.addInputModule(inLayer)
    net.addInputModule(hiddenLayer)
    net.addInputModule(outLayer)
    #Produce full connectivity
    in_to_hidden = FullConnection(inLayer, hiddenLayer)
    hidden_to_out = FullConnection(hiddenLayer, outLayer)
    #Add to network
    net.addConnection(in_to_hidden)
    net.addConnection(hidden_to_out)
    #Function call to make Network usable
    net.sortModules()

def create_label_array(images, blocks):
    #Create array to holde labels, 1:Present and 0:Not present
    labels_array = np.zeros((blocks,1),'uint8')
    for i in range(blocks):
        if images[i].sum() > 0:
            labels_array[i] = 1
    return labels_array

def classify(training, testing, HIDDEN_NEURONS, MOMENTUM, WEIGHTDECAY,
             LEARNING_RATE, LEARNING_RATE_DECAY, EPOCHS):

    INPUT_FEATURES = testing['rows'] * testing['cols']

    trndata = ClassificationDataSet(INPUT_FEATURES, 1, nb_classes = 2)
    tstdata = ClassificationDataSet(INPUT_FEATURES, 1, nb_classes = 2)

    #Add testing and training data to Classification dataset
    for i in range(len(testing['x'])):
        tstdata.addSample(ravel(testing['x'][i]), [testing['y'][i]])

    for i in range(len(training['x'])):
        trndata.addSample(ravel(training['x'][i]), [training['y'][i]])

    trndata._convertToOneOfMany()
    tstdata._convertToOneOfMany()
    #print trndata
    #print tstdata

    #Build Network
    net = buildNetwork(trndata.indim, HIDDEN_NEURONS, trndata.outdim, outclass=SoftmaxLayer)
    print net
    #Create Trainer
    #print trndata
    #print tstdata
    trainer = BackpropTrainer(net, dataset=trndata)

    #train network
    for i in range(EPOCHS):
        #error = trainer.train()
        #print "Epoch: %d, Error: %7.4f" % (i, error)
        trainer.trainEpochs(1)
        trnresult = percentError(trainer.testOnClassData(),
                                 trndata['class'])
        tstresult = percentError(trainer.testOnClassData(
                                 dataset=tstdata), tstdata['class'])

        print("epoch: %4d" % trainer.totalepochs,
                     "  train error: %5.2f%%" % trnresult,
                     "  test error: %5.2f%%" % tstresult)
    return net

#Test
if __name__ == '__main__':
    #Split image into image blocks
    train_blocks, label_blocks = get_labeled_data(train_file, train_file)
    test_blocks, label_blocks = get_labeled_data(test_file, test_file)

    #print test_blocks[0][0]
    #print test_blocks[18].sum()
    #view_data(20)
    labels_test = create_label_array(test_blocks,64)
    labels_train = create_label_array(train_blocks, 64)

    #print labels
    print 'Getting Test and Training data '
    test_data = {'y': labels_test,'x': test_blocks, 'rows':64, 'cols':64 }
    train_data = {'y': labels_train,'x': train_blocks, 'rows':64, 'cols':64 }
    #print train_data
    #print test_data
    #img = Image.open(filename)
    #im = numpy.array(img)
    #print
    #Run Classification
    classify(train_data, test_data, 25, 0.1,
             0.01, 0.01, 1, 50)
    '''print test_blocks[3]
    print image
    im2= Image.fromarray(label_image)
    im2.save('train_test.tif')'''
