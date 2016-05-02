import unittest
from PIL import Image
import load_extension
import os
import numpy as np
import cPickle as pickle
import convolutional_NN as cn
import makeTest as mt


class Test_convolutional_nn(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # pass
        mt.make()
        self.testfile = 'test.tif'
        self.trainfile = 'train.tif'
        self.downloadfile = 'TRA_000823_1720_COLOR.JP2'

        temp = np.ones((256,256),'uint8')*125
        self.rows, self.cols = load_extension.getDims(self.testfile)

        test_image = Image.fromarray(temp)
        test_image.save(self.testfile)
        self.image_8bit = np.array(Image.open(self.testfile))

        train_image = Image.fromarray(temp)
        train_image.save(self.trainfile)
        self.image_8bit = np.array(Image.open(self.trainfile))
        self.dims = self.image_8bit.shape

    @classmethod
    def tearDownClass(self):
        os.remove(self.downloadfile)
        os.remove(self.testfile)
        os.remove(self.trainfile)
        os.remove('prediction.tif')
        os.remove('net.pickle')

    def test_get_labeled_data(self):
        x = cn.get_labeled_data(self.testfile, self.trainfile, block_size=32)
        self.assertEqual(len(x), 2)
        self.assertEqual(x[1].size, 64)

    def test_trainNetwork(self):
        cn.trainNetwork(1, self.testfile, self.trainfile)
        with open('net.pickle', 'rb') as f:
            net_pretrain = pickle.load(f)
        self.assertIsNotNone(net_pretrain)

    def test_makePredictions(self):
        cn.trainNetwork(5, self.testfile, self.trainfile)
        cn.makePredictions(self.testfile)
        rows, cols = load_extension.getDims(self.testfile)
        predimage = 'prediction.tif'
        testrows, testcols = load_extension.getDims(predimage)
        self.assertEqual((rows, cols), (testrows, testcols))
        self.assertIs(type(testrows), int)
        self.assertIs(type(testcols), int)

    def test_getPredictionData(self):
        x = cn.getPredictionData(self.testfile, block_size=32)
        rows, cols = load_extension.getDims(self.testfile)
        testrows, testcols = x[1].shape
        self.assertEqual(np.ndim(x[0]), 4)
        self.assertEqual((testrows, testcols), (rows, cols))

    def test_loadDataset(self):
        x = cn.loadDataset(self.testfile, self.trainfile)
        self.assertEqual(np.ndim(x[0]), 4)
        self.assertEqual(np.ndim(x[1]), 1)

    def test_convolutionalNeuralNetwork(self):
        x = cn.convolutionalNeuralNetwork(0)
        self.assertIsNotNone(x.layers[0][1])

    def test_download_image(self):
        cn.download_image(self.downloadfile)

if __name__ == '__main__':
    unittest.main()
