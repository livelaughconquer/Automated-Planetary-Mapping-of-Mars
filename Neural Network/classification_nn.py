__author__ = 'JOrge'
#Neural Network for Olivetti Face Classification
#The code below was not all written by Team Strata and instead comes from a tutorial found at corpocat.com


#Pybrain Libraries
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader


 #Library for graphical representation
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from sklearn import datasets
from numpy import ravel

#Dataset for visualization (Olivetti Faces from AT&T dataset
olivetti_set = datasets.fetch_olivetti_faces()
images, pixels = olivetti_set.data, olivetti_set.target
print "Dataset Length and pixels of each image... "
images.shape

#Feed Forward Neural Network with 4096 input neurons, 1 hidden layer with 64 neurons and 1 output neuron
inputData = ClassificationDataSet(4096, 1, nb_classes=40)
for i in range(len(images)):
    inputData.addSample(ravel(images[i]), pixels[i])

#Split data into 75% training and 25% test data
testdata, traindata = inputData.splitWithProportion( 0.25 )

#Convert one output to 40 binary outputs and convert supervised to classification dataset
def _convert_supervised_to_classification(supervised_dataset):
    classification_dataset = inputData

    for j in xrange(0,supervised_dataset.getLength()):
        classification_dataset.addSample(supervised_dataset.getSample(j)[0], supervised_dataset.getSample(j)[1])
    return classification_dataset

traindata = _convert_supervised_to_classification(traindata)
testdata = _convert_supervised_to_classification(testdata )
#Print data inside Neural Network
#print "Printing Neural Network Data...."
#print traindata['input'], traindata['target'], testdata.indim, testdata.outdim

#Build Network and BackPropagation trainer

feed_nn = buildNetwork(traindata.indim, 64, traindata.outdim, outclass=SoftmaxLayer)
trainer = BackpropTrainer( feed_nn, dataset=traindata, momentum=0.1, learningrate=0.01, verbose=True, weightdecay=0.01)

trainer.trainEpochs(10)
print 'Percent Error on Test dataset: ', percentError(trainer.testOnClassData(
    dataset=testdata), testdata['class'])