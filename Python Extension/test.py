import load_extension
import numpy as np
import matplotlib.pyplot as plt
import Tkinter
import datetime
from PIL import Image

root = Tkinter.Tk()


#assumes you have Ryans images in the same folder as this script
filename ="PSP_009650_1755_RED.tif"
#filename = "example.tif"
training_file = "PSP_009650_1755_RED_dunes.tif"
rows,cols =  load_extension.getDims(filename)
print rows,cols

start = datetime.datetime.now()
print start
image = np.ones((rows,cols),'uint8')
label_image = np.ones((rows,cols),'uint8')

# x is a dummy to use as a form of error checking will return false on error
x = load_extension.getImage(image ,filename)
end = datetime.datetime.now()
#print end
print "Total image load time: ",end-start

start = datetime.datetime.now()
#print start
x = load_extension.getTraining(label_image,filename, training_file)
end = datetime.datetime.now()
#print end
print "Total training image load time: ",end-start

start = datetime.datetime.now()
#print start
x = load_extension.writeImage(image,"TEST.tif")
end = datetime.datetime.now()
#print end
print "Total image write time: ", end-start
#start = datetime.datetime.now()
#print start
#x = load_extension.writeImage(image,"TEST.tif")
#end = datetime.datetime.now()
#print end
#print end-start
ratio = max((cols)/root.winfo_screenwidth(),(rows)/root.winfo_screenheight()) 
size = (cols /ratio , rows / ratio)

start = datetime.datetime.now()
#print start
im = Image.fromarray(image)
im.save("TEST2.tif")
end = datetime.datetime.now()
#print end
print "Total PIL write time: ", end-start

start = datetime.datetime.now()
#print start
im.thumbnail(size, Image.ANTIALIAS)
im.show()
end = datetime.datetime.now()
#print end
print "Total PIL downsample time: ",end-start
