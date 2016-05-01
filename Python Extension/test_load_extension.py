import unittest
from PIL import Image
import load_extension as le
import numpy as np
import os
Image.MAX_IMAGE_PIXELS = None


class Test_load_extension(unittest.TestCase):
    def setUp(self):
        # pass
        self.filename = "load_extension_test_image.tif"
        temp = np.ones((256,256),'uint8')*125
        test_image = Image.fromarray(temp)
        test_image.save(self.filename)
        self.image_8bit = np.array(Image.open(self.filename))
        self.dims = self.image_8bit.shape


    def tearDown(self):
       os.remove(self.filename)


    def test_getDims(self):
        # pil_dims = Image.open()
        assert(self.dims==le.getDims(self.filename), True)


    def test_writeImage(self):
        img_dims = (25600,25600)
        img_1 = np.ones(img_dims,'uint8') * 123
        img_2 = img_1.copy()
        le.writeImage(img_1, self.filename)
        read_img = np.zeros(img_dims,'uint8')
        le.getImage(read_img,self.filename)

        assert((read_img==img_2).all(), True )


    def test_getImage(self):
        x = np.ones(le.getDims(self.filename), 'uint8')
        le.getImage(x, self.filename)
        self.image_8bit = np.array(Image.open(self.filename))
        assert ((self.image_8bit == x).all(), True)


    def test_getTraining(self):
        test = np.array(Image.open(self.filename), 'uint8')
        train = np.array(Image.open(self.filename), 'uint8')
        np.bitwise_xor(test,train,test)
        image = np.ones(test.shape,test.dtype)
        le.getTraining(image,self.filename, self.filename)
        assert((test == image).all(), True)

    def test_getLabel(self):
        rows, cols = (25600, 25600)
        diag = np.zeros((rows, cols), 'uint8')
        np.fill_diagonal(diag, 1)
        block_size = 256
        for i in xrange(0, rows, block_size):
            for j in xrange(0, cols, block_size):
                if diag[i, j] == 1:
                    diag[i:i + block_size, j:j + block_size] = 255

        thresholds = [0, 0.25, 0.5, 0.75, 1.0]
        block_sizes = [2 ** i for i in range(8, 11)]
        success = "1"
        failure = "0"

        for block_size in block_sizes:
            for threshold in thresholds:
                for i in xrange(0, rows, block_size):
                    for j in xrange(0, cols, block_size):
                        temp = diag[i:i + block_size, j:j + block_size]
                        test_percent = np.count_nonzero(temp) / temp.size
                        if test_percent >= threshold:
                            test_label = success
                        else:
                            test_label = failure
                        assert(test_label==le.getLabel(temp, success, success, threshold),True)

if __name__ == '__main__':
    unittest.main()