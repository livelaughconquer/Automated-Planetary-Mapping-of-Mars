#define BOOST_PYTHON_STATIC_LIB
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
//#define BOOST_PYTHON_MAX_ARITY 30

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "gdal_priv.h"

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/tuple.hpp>

#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/numeric.hpp>

#include <numpy/ndarrayobject.h>

#include <string>
#include <iostream>

#include <thread>
#include <vector>
namespace py = boost::python;

/*
This function returns a Classification Label string
	@ PyObject* np: This is an numpy array to be inspected 
	@ string success: This is the Label that will be used to indicate the desired condition is met
	@ string failure: This is the Label that will be used to indicate the desired condition is not met
	@ double threshold: This a 0 to 1.0 percentage representation used for classification of the image 
*/
std::string getLabel(PyObject* np, std::string success, std::string failure, double threshold) {
	std::string label;
	Py_BEGIN_ALLOW_THREADS;
	int min = 2, max = 2;
	PyArray_Descr* dtype = NULL;
	PyArrayObject* ret = (PyArrayObject*)PyArray_FromAny(np, dtype, min, max, NPY_ARRAY_WRITEABLE, NULL);

	int dim = PyArray_NDIM(ret);
	npy_intp* test = PyArray_SHAPE(ret);
	uint8_t* tester = (uint8_t*)PyArray_DATA(ret);

	int counter = 0;
	
	for (int i = 0; i<test[0]; i++) {
		counter += cv::countNonZero(cv::Mat(1,test[1], CV_8UC1, PyArray_GETPTR2(ret, i, 0)) ); //<< &temp << " ";
	}


	double percentage = ((double)counter) / ((double)(test[0] * test[1]));

	if (percentage >= threshold) {
		label = success;

	}
	else {
		label = failure;
	}

	Py_END_ALLOW_THREADS;
	return label;
}

/*Helper function to open a Gdal dataset*/
GDALDataset* open_dataset(std::string file_name) {
		return (GDALDataset *)GDALOpen(file_name.c_str(), GA_Update);
	}

/*Helper function to close a Gdal dataset*/
void close_dataset(GDALDataset* data) {
		GDALClose(data);
	}

/*
This function returns a tuple with the rows and columns of the given image
	@ string filename: File system name that will be inspected
*/
py::tuple getDims(std::string file_name) {
		//python::list dimensions;
		GDALDataset* data = open_dataset(file_name);
		GDALRasterBand *data_band;
		// Acquires pointers to raster data inside dataset structure
		data_band = data->GetRasterBand(1);
		//mask_band = mask->GetRasterBand(1);

		// Get size of images. In future the sizes should be be acquired from both images and compared to prevent merge errors
		uint32_t image_width = data_band->GetXSize();
		uint32_t image_height = data_band->GetYSize();

//		dimensions.append(image_width, image_height);
		close_dataset(data);
		return py::make_tuple(image_height,image_width);

	}

/*Intialization function*/
void init_extent() {
	Py_Initialize();
	GDALAllRegister();
    import_array();
}

/*Threading helper function that populates numpy array with pixel data*/
void filler( void* start, uint8_t * arr,int stride ){
   uint8_t* temp =  (uint8_t *) start;
//   uint8_t
   for(int i =0; i<stride; i++){
        temp[i] = arr[i];
   }
//   return;
}

/*This returns a bool to indicate that the operation was completed successfully
	@ PyObject* np: This is an numpy array to be filled with the label information at the pixel level
	@ string file_name: This is the file system name of image that the training image will be compared too
	@ string training_file: This is the file system name of training image that the training pixel data will be extracted from
*/
bool getTraining(PyObject* np, std::string file_name, std::string training_file) {
	Py_BEGIN_ALLOW_THREADS;

	int min = 2, max = 2;
	PyArray_Descr* dtype = NULL;

	PyArrayObject* ret = (PyArrayObject*)PyArray_FromAny(np, dtype, min, max, NPY_ARRAY_WRITEABLE, NULL);

	int dim = PyArray_NDIM(ret);
	npy_intp* test = PyArray_DIMS(ret);
	uint8_t* tester = (uint8_t*)PyArray_DATA(ret);
	npy_intp s = PyArray_STRIDE(ret, 0);

	GDALDataset* data = open_dataset(file_name);
	GDALDataset* mask = open_dataset(training_file);

	// Acquires pointers to raster data inside dataset structure
	GDALRasterBand* data_band = data->GetRasterBand(1);
	GDALRasterBand* mask_band = mask->GetRasterBand(1);

//	data_band = data->GetRasterBand(1);
	// Get size of images. In future the sizes should be be acquired from both images and compared to prevent merge errors
	
	uint32_t image_columns = data_band->GetXSize();
	uint32_t image_rows = data_band->GetYSize();

	//Allocates arrays to store image information for original and labelled images
	//Initial trainng data is known to be 8-bit but future should check the data-type of image and process accordingly
	uint8_t * data_arr = (uint8_t *)VSIMalloc(sizeof(uint8_t)*image_columns*image_rows);
	uint8_t * mask_arr = (uint8_t *)VSIMalloc(sizeof(uint8_t)*image_columns*image_rows);


	//Read entire images into their respective arrays
	data_band->RasterIO(GF_Read, 0, 0, image_columns, image_rows, data_arr, image_columns, image_rows, GDT_Byte, 0, 0);
	mask_band->RasterIO(GF_Read, 0, 0, image_columns, image_rows, mask_arr, image_columns, image_rows, GDT_Byte, 0, 0);


	//Builds a opencv matrix out of the image data housed in the allocated arrays
	cv::Mat image = cv::Mat(image_rows, image_columns, CV_8U, data_arr);
	cv::Mat mask_layer = cv::Mat(image_rows, image_columns, CV_8U, mask_arr);

	//xor the matrices to find differences between images provided by sponsor
	cv::bitwise_xor(image, mask_layer, mask_layer);

	VSIFree(data_arr);
	close_dataset(data);

	uint64_t position = 0;
	
	std::vector<std::thread> threadder;
	uint8_t * temp;
	// }
	for (uint64_t i = 0; i<image_rows; i++) {
		temp = (uint8_t*)PyArray_GETPTR2(ret, i, 0);
		threadder.push_back(std::thread(filler, temp, &mask_arr[position], image_columns));
		position += image_columns;
	}

	for (auto& thread : threadder) {
		thread.join();
	}
//	cv::imwrite("training_image.tif", mask_layer);

	// PyEval_ReleaseLock();
	if (position >= (image_rows*image_columns)) {

		//    delete threadder;
		VSIFree(mask_arr);
		close_dataset(mask);

		Py_END_ALLOW_THREADS;
		//    Py_Finalize();
		return true;
	}



	//    std::cout << s;//test[0]<<" " <<test[1] <<std::endl;
	//  tester[2]=99;
	//   std::cout << (short)tester[4];
}

/*This returns a bool to indicate that the operation was completed successfully
	@ PyObject* np: This is an numpy array to be filled with the image data pixel information
	@ string file_name: This is the file system name of image that is to be loaded in the memory
*/
bool getImage(PyObject* np, std::string file_name){
    Py_BEGIN_ALLOW_THREADS;

    int min = 2, max =2;
    PyArray_Descr* dtype = NULL;

   PyArrayObject* ret = (PyArrayObject*)PyArray_FromAny( np, dtype, min, max, NPY_ARRAY_WRITEABLE, NULL);

    int dim = PyArray_NDIM(ret);
    npy_intp* test = PyArray_DIMS(ret);
    uint8_t* tester = (uint8_t*)PyArray_DATA(ret);
    npy_intp s = PyArray_STRIDE (ret, 0);
    GDALDataset* data = open_dataset(file_name);
    GDALRasterBand *data_band = data->GetRasterBand(1);



		// Get size of images. In future the sizes should be be acquired from both images and compared to prevent merge errors
    uint32_t image_columns = data_band->GetXSize();
    uint32_t image_rows = data_band->GetYSize();

    uint8_t * data_arr = (uint8_t *)VSIMalloc(sizeof(uint8_t)*image_columns*image_rows);
    data_band->RasterIO(GF_Read, 0, 0, image_columns, image_rows, data_arr, image_columns, image_rows, GDT_Byte, 0, 0);
//
    close_dataset(data);

//
    uint64_t position = 0;
//    uint8_t * temp;

std::vector<std::thread> threadder;
uint8_t * temp;
// }
    for(uint64_t i=0; i<image_rows;i++){
           temp =(uint8_t*)PyArray_GETPTR2(ret,i,0);
            threadder.push_back(std::thread(filler, temp ,&data_arr[position],image_columns));
            position += image_columns;
        }

for(auto& thread : threadder){
    thread.join();
}
// PyEval_ReleaseLock();
 if(position>=(image_rows*image_columns)){

//    delete threadder;
    VSIFree(data_arr);
    Py_END_ALLOW_THREADS;
//    Py_Finalize();
    return true;
 }

}

/*This returns a bool to indicate that the operation was completed successfully
	@ PyObject* np: This is an numpy array containing pixel information to be written to disk
	@ string file_name: This is the file system name of image that is to be written to disk
*/
bool writeImage(PyObject* np, std::string file_name) {
	Py_BEGIN_ALLOW_THREADS;
	int min = 2, max = 2;
	PyArray_Descr* dtype = NULL;

	PyArrayObject* ret = (PyArrayObject*)PyArray_FromAny(np, dtype, min, max, NPY_ARRAY_WRITEABLE, NULL);
	//if()
	const char *format = "GTiff";
	//   const char *format = "JP2OpenJPEG";
	GDALDriver* driver, *vrt_driver;
	int dim = PyArray_NDIM(ret);
	npy_intp* test = PyArray_DIMS(ret);
	uint16_t* tester = (uint16_t*)PyArray_DATA(ret);
	npy_intp s = PyArray_STRIDE(ret, 0);
	//GDALDataset* data = open_dataset(file_name);
	//GDALRasterBand *data_band = data->GetRasterBand(1);
	//std::cout << test[0] << " " << test[1] << " "  << file_name <<" " << s <<std::endl;

	//	cv::Mat image = cv::Mat(test[0], test[1], CV_8U, ret);
	vrt_driver = (GDALDriver *)GDALGetDriverByName("MEM");
	driver = (GDALDriver *)GDALGetDriverByName(format);
	char ** options = NULL;

	GDALDataset* src = vrt_driver->Create("", test[1], test[0], 1, GDT_Byte, options);
	src->RasterIO(GF_Write, 0, 0, test[1], test[0], PyArray_GETPTR2(ret, 0, 0), test[1], test[0], GDT_Byte, 1, NULL, 0, 0, 0, NULL);

	GDALDataset* image = driver->CreateCopy(file_name.c_str(), src, 0, options, NULL, NULL);
	close_dataset(src);
	GDALRasterBand *image_band = image->GetRasterBand(1);

	int cols = image_band->GetXSize();
	int rows = image_band->GetYSize();

//	std::cout << rows << " " << cols << std::endl;

	//	image_band->RasterIO(GF_Write, 0, 0, cols, rows,(uint16_t **)PyArray_GETPTR2(ret, 0, 0) , cols, rows, GDT_UInt16, 0, 0);

	close_dataset(image);
	//std::cout << driver->GetDescription();

	Py_END_ALLOW_THREADS;
	return true;

}

//using namespace boost::python;
BOOST_PYTHON_MODULE(load_extension)
{
	init_extent();
	// Add regular functions to the module.
	py::def("getTraining", getTraining);
	py::def("getImage", getImage);
	py::def("getDims", getDims);
	py::def("writeImage", writeImage);
	py::def("getLabel", getLabel);
}

