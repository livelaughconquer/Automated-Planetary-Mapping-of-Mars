#include <Python.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//static PyObject* testmoduleError;

static PyObject* load(PyObject *self, PyObject *args)
{
	int i;
	Mat image = imread("C:\\Users\\SeanB\\Pictures\\apple.jpg" , CV_LOAD_IMAGE_COLOR);   // Read the file
	if (!PyArg_ParseTuple(args, "i", &i)) // Changing the parse type to a string with the path instead crashes python
	{
		goto error;
	}
	
	//return image;  // This line doesn't work because proper conversion between cv::mat and numpy is needed
	
	return 0;
	error:
	return 0;
}

PyMethodDef testmoduleMethods[] = 
{
	{"load", (PyCFunction)load, METH_VARARGS,0},
	{0,0,0,0}
};

PyMODINIT_FUNC
inittestmodule(void)
{
	PyObject *m;

	m = Py_InitModule("testmodule", testmoduleMethods);
	if (m == NULL)
		return;
}

int main(int argc, char *argv[])
{
	// Pass argv[0] to the Python interpreter 
	Py_SetProgramName(argv[0]);

	// Initialize the Python interpreter.  Required. 
	Py_Initialize();

	// Add a static module 
	inittestmodule();
}