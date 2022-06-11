#include "LLF.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
using namespace std;
using namespace cv;

#pragma region lib_link
#define CV_LIB_PREFIX "opencv_"

#define CV_LIB_VERSION CVAUX_STR(CV_MAJOR_VERSION)\
    CVAUX_STR(CV_MINOR_VERSION)\
    CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define CV_LIB_SUFFIX CV_LIB_VERSION "d.lib"
#else
#define CV_LIB_SUFFIX CV_LIB_VERSION ".lib"
#endif

#define CV_LIBRARY(lib_name) CV_LIB_PREFIX CVAUX_STR(lib_name) CV_LIB_SUFFIX
#pragma comment(lib, CV_LIBRARY(core))
#pragma comment(lib, CV_LIBRARY(highgui))
#pragma comment(lib, CV_LIBRARY(imgcodecs))
#pragma comment(lib, CV_LIBRARY(imgproc))
#pragma endregion

//alpha blending comparison between src1 and src2 by GUI
void compare(const string wname, const Mat& src1, const Mat& src2)
{
	namedWindow(wname);
	int a = 0; createTrackbar("alpha", wname, &a, 100);
	int key = 0;
	Mat show;
	while (key != 'q')
	{
		addWeighted(src1, a * 0.01, src2, (100 - a) * 0.01, 0.0, show);
		imshow(wname, show);
		key = waitKey(1);
	}
}

int main()
{
	Mat src = imread("flower.png");
	Mat destFastLLF, destFourierLLF, destFastLLFAaptive, destFourierLLFAaptive;
	
	FastLLF llf;
	GaussianFourierLLF gfllf;

	const float sigma = 30.f;
	const float boost = 2.f;
	const int level = 2;
	const int order = 4;

	//parameter fix filter
	llf.filter(src, destFastLLF, order * 2, sigma, boost, level);
	gfllf.filter(src, destFourierLLF, order, sigma, boost, level);
	
	//parameter adaptive filter
	//generate parameter maps
	Mat sigmaMap(src.size(), CV_32F);
	sigmaMap.setTo(sigma);
	circle(sigmaMap, Point(src.size()) / 2, src.cols / 4, Scalar::all(sigma * 2.f), cv::FILLED);

	Mat boostMap(src.size(), CV_32F);
	boostMap.setTo(boost);
	circle(boostMap, Point(src.size()) / 2, src.cols / 4, Scalar::all(boost * 2.0), cv::FILLED);

	llf.filter(src, destFastLLFAaptive, order * 2, sigmaMap, boostMap, level);
	gfllf.filter(src, destFourierLLFAaptive, order, sigmaMap, boostMap, level);

	//compare("LLF", dest1, dest2);

	imshow("src", src);
	imshow("Fast LLF dest", destFastLLF);
	imshow("Fourier LLF dest", destFourierLLF);
	imshow("Fast LLF Adaptive dest", destFastLLFAaptive);
	imshow("Fourier LLF Adaptive dest", destFourierLLFAaptive);	
	waitKey();
	return 0;
}