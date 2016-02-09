#define MS_NO_COREDLL

#include "gdal_priv.h"
#include "cpl_conv.h" // for CPLMalloc()
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

uint8_t* get_mask(uint8_t* origin, uint8_t* labelled, uint32_t length) {

	for (uint32_t i = 0; i < length; i++)
		if (origin[i] == labelled[i])
			labelled[i] = 0;
		else
			labelled[i] = 255;

	return labelled;
}

bool get_north(uint8_t* image, uint32_t position, int64_t width) {
	int64_t test = (position - width);
	if ( test< 0)
		return false;
	else if (image[position] == image[position - width])
		return true;
	else
		return false;
}
bool get_south(uint8_t* image, uint32_t position, uint32_t width,uint32_t height) {
	if (position + width >= width*height)
		return false;
	else if (image[position] == image[position + width])
		return true;
	else
		return false;
}
bool get_west(uint8_t* image, uint32_t position) {
	if (position - 1 < 0)
		return false;
	else if (image[position] == image[position - 1])
		return true;
	else
		return false;
}
bool get_east(uint8_t* image, uint32_t position, uint32_t width, uint32_t height) {
	if (position + 1 <= width*height)
		return false;
	else if (image[position] == image[position + 1])
		return true;
	else
		return false;
}


uint8_t* remove_noise(uint8_t* mask_band, uint32_t width, uint32_t height) {
	for (uint32_t i = 0; i < width*height; i++)
		if (!(get_north(mask_band, i, width) || get_south(mask_band, i, width, height) || get_east(mask_band, i, width, height) || get_west(mask_band, i)))
			mask_band[i] = 255;
	return mask_band;
}

void shift_left(uint16_t* input, uint32_t length) {
	for (uint32_t i = 0; i < length; i++)
		input[i] = input[i] << 8;
	return;
}

int main(){
	std::string src = "PSP_009650_1755_RED.tif";
	std::string src_lbl = "PSP_009650_1755_RED_dunes.tif";

		GDALDataset  *dataset, *mask;
		GDALRasterBand *data_band, *mask_band;
		PyObject *toPy;
		cv::Mat image, mask_layer;

		//registers drivers
		GDALAllRegister();
		//GDALOpen() function to open a datset, in this case a JP2 file
		image = cv::imread(src_lbl + "training_image.tif");
		if (image.empty()) {
			dataset = open_dataset(src);
			mask = open_dataset(src_lbl);

			data_band = dataset->GetRasterBand(1);
			mask_band = mask->GetRasterBand(1);


			//int block_x, block_y;
			//data_band->GetBlockSize(&block_x, &block_y);

			uint32_t image_width = data_band->GetXSize();
			uint32_t image_height = data_band->GetYSize();

			//print out image_width and image_height to console

			uint8_t * data_arr = (uint8_t *)VSIMalloc(sizeof(uint8_t)*image_width*image_height);
			uint8_t * mask_arr = (uint8_t *)VSIMalloc(sizeof(uint8_t)*image_width*image_height);

			data_band->RasterIO(GF_Read, 0, 0, image_width, image_height, data_arr, image_width, image_height, GDT_Byte, 0, 0);
			mask_band->RasterIO(GF_Read, 0, 0, image_width, image_height, mask_arr, image_width, image_height, GDT_Byte, 0, 0);


			image = cv::Mat(image_height, image_width, CV_8U, data_arr);
			mask_layer = cv::Mat(image_height, image_width, CV_8U, mask_arr);

			cv::bitwise_xor(image, mask_layer, mask_layer);


			std::vector<cv::Mat> matrix_list;

			matrix_list.push_back(image);
			matrix_list.push_back(mask_layer);
			matrix_list.push_back(cv::Mat(image_height, image_width, CV_8U, cvScalar(0)));
			cv::merge(matrix_list, image);


			cv::imwrite(src_lbl + "training_image.tif", image);
			
			VSIFree(mask_arr);
			GDALClose(mask);

			VSIFree(data_arr);

			GDALClose(dataset);
		}
		else {
			std::cout << "Place Holder\n";
		}
	return 0;
}