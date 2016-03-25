"""
This simple ConvNet was modeled after oduerr's minimal Lasagne CNN tutorial.
"""

import cPickle as pickle
import gzip
import matplotlib.pyplot as plt
from lasagne import layers
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet

# Uncomment this block to reload previously stored model for more training
"""
# Load stored model
with open('net1.pickle', 'rb') as f:
    net_pretrain = pickle.load(f)

net_pretrain.max_epochs = 25  # Train the previous model over more epochs
"""

# Load data
with gzip.open('mnist_4000.pkl.gz', 'rb') as f:
    (X,y) = pickle.load(f)  # X contains the images and y contains the labels
PIXELS = len(X[0,0,0,:])

# Build the net
net1 = NeuralNet(
    # Geometry of the network
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, PIXELS, PIXELS), # None in the first axis indicates that the batch size can be set later
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    hidden4_num_units=500,
    output_num_units=10, output_nonlinearity=nonlinearities.softmax,

    # Learning rate parameters
    update_learning_rate=0.01,
    update_momentum=0.9,
    regression=False,
    max_epochs=100,
    verbose=1,

    # Training test-set split
    eval_size = 0.2
    )
net = net1.fit(X[0:1000,:,:,:],y[0:1000])  # Train the network for the first 1000 images and labels
#net = net_pretrain.fit(X[1000:2000,:,:,:],y[1000:2000])  # Train the previous model over the next range of images

# Choose a range of predictions
toTest = range(3025,3050)
preds = net.predict(X[toTest,:,:,:])
print preds

# Compare predictions and images
plt.interactive(False)
fig = plt.figure(figsize=(10,10))
for i,num in enumerate(toTest):
    a=fig.add_subplot(5,5,(i+1))
    plt.axis('off')
    a.set_title(str(preds[i]) + " (" + str(y[num]) + ")")
    plt.imshow(-X[num,0,:,:], interpolation='none',cmap=plt.get_cmap('gray'))
    plt.show()

# Store the trained model
with open('net1.pickle', 'wb') as f:
    pickle.dump(net, f, -1)
