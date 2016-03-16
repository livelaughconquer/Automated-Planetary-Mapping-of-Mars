import os, sys
from osgeo import gdal
# argv[1] is input tif, argv[2] is output file name
dset = gdal.Open(sys.argv[1])

width = dset.RasterXSize
height = dset.RasterYSize

print width, 'x', height

tilesize = 5000 # Change to desired tile size

for i in range(0, width, tilesize):
    for j in range(0, height, tilesize):
        w = min(i+tilesize, width) - i
        h = min(j+tilesize, height) - j
        gdaltranString = "gdal_translate -of GTIFF -srcwin "+str(i)+", "+str(j)+", "+str(w)+", " \
            +str(h)+" " + sys.argv[1] + " " + sys.argv[2] + "_"+str(i)+"_"+str(j)+".tif"
        os.system(gdaltranString)