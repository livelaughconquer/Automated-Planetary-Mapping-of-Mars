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
rows,cols =  load_extension.getDims(training_file)
#print load_extension.getDims(testfile)

train_image = np.zeros((rows,cols),'uint8')
load_extension.getTraining(train_image, filename,training_file)
#image = np.zeros((rows,cols),'uint8')

print "Training image load time:", datetime.datetime.now() - start
print
label = []
#test = np.ones((4,4),'uint8')
ones = np.ones((8,8),'uint8')
zeros = np.zeros((8,8),'uint8')
temp = []
mat = np.ones((8,8),'uint8')

for i in range(4):
    if i ==0:
        mat = np.ones((8,8),'uint8')
    else:
        mat = np.zeros((8,8),'uint8')    
    for j in range(3):
        if i==j+1:
            mat = np.hstack((mat,ones))
        else:
            mat = np.hstack((mat,zeros))
    temp.append(mat)
#    mat = np.zeros((8,8),'uint8')
mat = np.vstack((i for i in temp))
#ascii visualization of the image
for i in temp:
    print i

    

block_steps = [2,4,8,16,32] 
rows,cols = mat.shape
print mat.shape
for block in block_steps:
    cur =0
    label=[]
#    print block
    start = datetime.datetime.now()
    for i in xrange(0,rows,block):
        for j in xrange(0,cols,block):
#            print i,j
            label.append(load_extension.getLabel(mat[i:i+block,j:j+block], "Win","Lose", 1))
            cur+=1    
    print "Block: {0}x{1}".format(block,block),"Total time: ",datetime.datetime.now() -start,"Success label:",label.count("Win"),"Failure label:",label.count("Lose"), "Total labels:",len(label)

start = datetime.datetime.now()
block = 32
rows,cols = train_image.shape
print
print rows, cols
for i in xrange(0,rows,block):
    for j in xrange(0,cols,block):
#            print i,j
        label.append(load_extension.getLabel(train_image[i:i+block,j:j+block], "Win","Lose", 0.75))
print "Block: {0}x{1}".format(block,block),"Total time: ",datetime.datetime.now() -start,"Success label:",label.count("Win"),"Failure label:",label.count("Lose"), "Total labels:",len(label)

"""
ratio = max((cols)/root.winfo_screenwidth(),(rows)/root.winfo_screenheight()) 
size = (cols /ratio , rows / ratio)
block = 32
block_rows = rows/block
block_cols = cols/block
cur = 0
image = np.ones((rows,cols),'uint8')
label_image = np.ones((rows,cols),'uint8')*255
x = load_extension.getImage(image, nfilename)
x = load_extension.getImage(label_image, filename)
#x = load_extension.getTraining(label_image,filename, training_file)
end = datetime.datetime.now()
#print end
print "Image load time:",end - start

im = Image.fromarray(label_image[:rows/4][cols/2:])
im.thumbnail(size, Image.ANTIALIAS)
im.show()
im = Image.fromarray(image[:rows/4][cols/2:])
im.thumbnail(size, Image.ANTIALIAS)
im.show()
load_extension.writeImage(image[:rows/4][cols/2:],"TEST.tif")
load_extension.writeImage(label_image[:rows/4][cols/2:],"TEST_TRAIN.tif")

label = []
start = datetime.datetime.now()

for i in xrange(0,rows,block):
    for j in xrange(0,cols,block):
        label.append(load_extension.getLabel(label_image[i:i+block][j:j+block], "Win","Lose", 0.05))
        #cur += 1
"""
#print start
#image = np.zeros((rows,cols),'uint8')
#x = load_extension.getImage(image ,filename)
# x is a dummy to use as a form of error checking will return false on error

"""
print "Total image load time: ",end-start

start = datetime.datetime.now()
#print start
x = load_extension.getTraining(label_image,filename, training_file)
end = datetime.datetime.now()
#print end
print "Total training image load time: ",end-start
"""
""""
#print start
x = load_extension.writeImage(image[:rows/2][:],"TEST.tif")
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
im = Image.fromarray(image)
im.thumbnail(size, Image.ANTIALIAS)
im.show()
end = datetime.datetime.now()
#print end
print "Total PIL downsample time: ",end-start
"""