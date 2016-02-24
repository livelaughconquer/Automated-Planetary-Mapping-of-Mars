#define BOOST_PYTHON_STATIC_LIB
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "gdal_priv.h"

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/tuple.hpp>

#include <numpy/ndarrayobject.h>

#include <string>
#include <iostream>

namespace python = boost::python;

GDALDataset* open_dataset(std::string file_name) {
		return (GDALDataset *)GDALOpen(file_name.c_str(), GA_Update);
	}

void close_dataset(GDALDataset* data) {
		GDALClose(data);
	}

python::tuple get_dims(std::string file_name) {
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
		return python::make_tuple(image_width, image_height);

	}

void init() {
	Py_Initialize();
	GDALAllRegister();
}
BOOST_PYTHON_MODULE(load_extension)
{
	init();
	// Add regular functions to the module.
	python::def("get_dims", get_dims);
//	python::def("square", square);
}

