#define MS_NO_COREDLL

#include "gdal_priv.h"
////#include "cpl_conv.h" // for CPLMalloc()
#include "Python.h"
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
#include <string.h>
//using namespace cv;
//using namespace std;

GDALDataset* open_dataset(std::string file_name) {
	return (GDALDataset *)GDALOpen(file_name.c_str(), GA_Update);
}


int main(){
		//Hardcoded strings for training datasets. In future would be parameters for extension.
		std::string src = "PSP_009650_1755_RED.tif";
		std::string src_lbl = "PSP_009650_1755_RED_dunes.tif";


		//Create pointers to gdal datasets original, and the one marked by sponsor
		GDALDataset  *dataset, *mask; 

		//Create pointers to raster bands in dataset
		GDALRasterBand *data_band, *mask_band;

		//Pointer that will be used to pass data to python interpreter currently only used to make sure library/header was found.
		PyObject *toPy;

		//OpenCV matrices will be used for image processing
		cv::Mat image, mask_layer;

		//registers drivers for gdal to use
		GDALAllRegister();

		//Attempts to load previously built training data if no file found a empty matrix is returned 
		image = cv::imread(src_lbl + "training_image.tif");
		
		//Stop gap solution for merging of training data images housed inside conditional as the processing takes 6 to 7mins
		//while a load of previously written data is about 2 mins
		//if (image.empty()) {

			// Gets pointers to the two images provided by sponsor to use as training data
			dataset = open_dataset(src);
			mask = open_dataset(src_lbl);

			// Acquires pointers to raster data inside dataset structure
			data_band = dataset->GetRasterBand(1);
			mask_band = mask->GetRasterBand(1);

			// Get size of images. In future the sizes should be be acquired from both images and compared to prevent merge errors
			uint32_t image_width = data_band->GetXSize();
			uint32_t image_height = data_band->GetYSize();
			
			//Allocates arrays to store image information for original and labelled images
			//Initial trainng data is known to be 8-bit but future should check the data-type of image and process accordingly
			uint8_t * data_arr = (uint8_t *)VSIMalloc(sizeof(uint8_t)*image_width*image_height);
			uint8_t * mask_arr = (uint8_t *)VSIMalloc(sizeof(uint8_t)*image_width*image_height);


			//Read entire images into their respective arrays
			data_band->RasterIO(GF_Read, 0, 0, image_width, image_height, data_arr, image_width, image_height, GDT_Byte, 0, 0);
			mask_band->RasterIO(GF_Read, 0, 0, image_width, image_height, mask_arr, image_width, image_height, GDT_Byte, 0, 0);


			//Builds a opencv matrix out of the image data housed in the allocated arrays
			image = cv::Mat(image_height, image_width, CV_8U, data_arr);
			mask_layer = cv::Mat(image_height, image_width, CV_8U, mask_arr);

			//xor the matrices to find differences between images provided by sponsor
			cv::bitwise_xor(image, mask_layer, mask_layer);

			//Set up vector to use as a list of matrices
			std::vector<cv::Mat> matrix_list;

			//adds matrices to list to represent channels in RGB image
			//A matrix of zeroes added as padding as 2 channelled image disallowed
			matrix_list.push_back(image);
			matrix_list.push_back(mask_layer);
			matrix_list.push_back(cv::Mat(image_height, image_width, CV_8U, cvScalar(0)));
			
			//merges the matrices into one matrix 
			cv::merge(matrix_list, image);

			//write the composite training image as stop gap to load up training data.
			cv::imwrite(src_lbl + "training_image1.tif", image);

			//deallocates the array for labelled image and closes the gdal dataset
			VSIFree(mask_arr);
			GDALClose(mask);

			//deallocates the array for labelled image and closes the gdal dataset
			VSIFree(data_arr);
			GDALClose(dataset);
		//}
	//	else {
		//	std::cout << "Place Holder\n";
	//	}
	return 0;
}