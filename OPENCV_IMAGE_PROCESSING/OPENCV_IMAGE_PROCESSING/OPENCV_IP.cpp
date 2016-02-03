////#include "gdal_priv.h"
////#include "cpl_conv.h" // for CPLMalloc()
//
//#include "opencv2/opencv.hpp"
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//
//int main(int argc, const char** argv)
//{
//	Mat img = imread("downsampled.JP2", CV_LOAD_IMAGE_UNCHANGED); //Function imread(), reads in an image
//
//	if (img.empty()) //check to see if image was empty
//	{
//		cout << "Error :Image was not loaded" << endl;  //If empty, display error message
//		return -1;
//	}
//
//	namedWindow("JP2 File", CV_WINDOW_AUTOSIZE); //create a window with the name "JP2 File
//	imshow("JP2 File", img); //display the image which is stored in the 'img' in the window created
//
//	waitKey(0); //wait infinite time for a keypress
//
//	destroyWindow("JP2 File"); //destroy window created
//
//	return 0;
//}