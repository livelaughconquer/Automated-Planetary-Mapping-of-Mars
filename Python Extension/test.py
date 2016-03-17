import load_extension
import numpy as np
import matplotlib.pyplot as plt

import datetime
from PIL import Image
#assumes you have Ryans images in the same folder as this script
filename ="PSP_009650_1755_RED.tif"
training_file = "PSP_009650_1755_RED_dunes.tif"
rows,cols =  load_extension.get_dims(filename)
print rows,cols

start = datetime.datetime.now()
print start
image = np.ones((rows,cols),'uint8')
label_image = np.ones((rows,cols),'uint8')

# x is a dummy to use as a form of error checking will return false on error
x = load_extension.getImage(image ,filename)
end = datetime.datetime.now()
print end
print end-start
start = datetime.datetime.now()
print start

x = load_extension.getTraining(label_image,filename, training_file)

end = datetime.datetime.now()
print end
print end-start
