import load_extension
import numpy as np
import matplotlib.pyplot as plt

import datetime
# from PIL import Image
filename = "example.tif"
# filename ="PSP_009650_1755_RED.TIF"

row,cols =  load_extension.get_dims(filename)
print row,cols

start = datetime.datetime.now()
print start
# x = plt.imread(filename)
# x = Image.open(filename)

what = np.zeros((row,cols),'uint8')
x = load_extension.getImage(what ,filename)

end = datetime.datetime.now()
print end
print end-start
print what
# image = Image.fromarray(what)
# image.save("TEST.tif")
# image.show()