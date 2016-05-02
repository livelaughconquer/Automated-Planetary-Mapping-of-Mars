#import write_extension
import numpy as np
import datetime
from PIL import Image

def make():
    filename ="test.tif"
    training_file = "train.tif"
    rows= 256
    cols = 256

    image = np.ones((rows,cols),'uint8')
    for i in range(rows):
        for j in range(cols):
            image[i][j] = j
    img = Image.fromarray(image)
    img1 = Image.fromarray(np.fliplr(image))
    img.save(filename)
    img1.save(training_file)