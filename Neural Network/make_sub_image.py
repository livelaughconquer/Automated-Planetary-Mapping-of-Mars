import load_extension
import numpy as np
from PIL import Image
import Tkinter

root = Tkinter.Tk()



image_file = "PSP_009650_1755_RED"
train_file = image_file+"_dunes.tif"
image_file += ".tif"

rows, cols = load_extension.getDims(image_file)
ratio = max((cols)/root.winfo_screenwidth(),(rows)/root.winfo_screenheight())
size = (cols /ratio , rows / ratio)

sub_rows = rows/8
sub_cols = cols/4
print sub_rows, sub_cols


image = np.zeros((rows,cols),"uint8")
load_extension.getImage(image,image_file)

# im = Image.fromarray(image[:sub_rows, sub_cols:])
#im.thumbnail(size,Image.ANTIALIAS)
#im.show()
# ones = np.ones((sub_rows,sub_cols),'uint8')
# image = np.multiply(ones, image[:sub_rows, :sub_cols])

load_extension.writeImage( image[:sub_rows][sub_cols:], "test.tif")
load_extension.getImage(image,train_file)
load_extension.writeImage(image[:sub_rows][sub_cols:], "train.tif")