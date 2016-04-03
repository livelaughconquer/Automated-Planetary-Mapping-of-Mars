import load_extension
import numpy as np
import matplotlib.pyplot as plt
import Tkinter
import datetime
from PIL import Image

root = Tkinter.Tk()

start = datetime.datetime.now()

#assumes you have Ryans images in the same folder as this script
filename ="PSP_009650_1755_RED.tif"

#filename = "example.tif"
#filename = "training_image.tif"
training_file = "PSP_009650_1755_RED_dunes.tif"
rows,cols =  load_extension.getDims(filename)

train_image = np.zeros((rows,cols),'uint8')
image = np.zeros((rows,cols),'uint8')
load_extension.getImage(image,filename)
newfile = filename.split(".")[0] +"_flipped."+filename.split(".")[1]
start = datetime.datetime.now()
image = image[::-1]
ones = np.ones((rows,cols),'uint8')
image = np.multiply(ones, image)
load_extension.writeImage(image,newfile)
print datetime.datetime.now() -start