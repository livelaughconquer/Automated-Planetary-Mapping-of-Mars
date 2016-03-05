#define BOOST_PYTHON_STATIC_LIB
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

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

void simp(){std::cout<< "poop\n";}
GDALDataset* open_dataset(std::string file_name) {
		return (GDALDataset *)GDALOpen(file_name.c_str(), GA_Update);
	}

void close_dataset(GDALDataset* data) {
		GDALClose(data);
	}

py::tuple get_dims(std::string file_name) {
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
void init() {
	Py_Initialize();
	GDALAllRegister();
    import_array();
}

void filler( void* start, uint8_t * arr,int stride ){
   uint8_t* temp =  (uint8_t *) start;
//   uint8_t
   for(int i =0; i<stride; i++){
        temp[i] = arr[i];
   }
//   return;
}

bool getImage(PyObject* np, std::string file_name){
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

Py_BEGIN_ALLOW_THREADS;
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



//    std::cout << s;//test[0]<<" " <<test[1] <<std::endl;
  //  tester[2]=99;
 //   std::cout << (short)tester[4];
}
//using namespace boost::python;
BOOST_PYTHON_MODULE(load_extension)
{
	init();
	// Add regular functions to the module.
//	py::
//    py::numeric::array::set_module_and_type("numpy","ndarray");
	py::def("getImage", getImage);
	py::def("get_dims", get_dims);

//	py::
//	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
}

