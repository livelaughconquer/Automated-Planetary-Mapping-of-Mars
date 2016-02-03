#include "gdal_priv.h"
#include "cpl_conv.h" // for CPLMalloc()


#include "opencv2/core/core.hpp"
//#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/photo/photo.hpp"
//#include "opencv2/video/video.hpp"
//#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/objdetect/objdetect.hpp"
//#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/contrib/contrib.hpp"

//#include "opencv2/core/core_c.h"
//#include "opencv2/highgui/highgui_c.h"
//#include "opencv2/imgproc/imgproc_c.h"
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	GDALDataset  *dataset;
	GDALRasterBand *band, *overview;

	//registers drivers
	GDALAllRegister();
	//GDALOpen() function to open a datset, in this case a JP2 file
	dataset = (GDALDataset *)GDALOpen("example2.jp2", GA_ReadOnly);

	band = dataset->GetRasterBand(1);
	overview = band->GetOverview(3);

	//	int block_x, block_y;

	//	overview->GetBlockSize(&block_x, &block_y);

	uint image_width = overview->GetXSize();
	uint image_height = overview->GetYSize();
	//print out image_width and image_height to console
	cout << image_width << " " << image_height << " " << dataset << endl;

	ushort * arr = (ushort *)VSIMalloc(sizeof(ushort)*image_width*image_height);//new uint16_t[image_width];//(uint16_t *)VSIMalloc(image_length);
	overview->RasterIO(GF_Read, 0, 0, image_width, image_height, arr, image_width, image_height, GDT_UInt16, 0, 0);
	Mat image = Mat(image_height, image_width, CV_16U, arr);
	cv::imwrite("C:\\warmerda\\bld\\bin\\downsampled.jp2", image);
	VSIFree(arr);


	GDALClose(dataset);

	return 0;
}