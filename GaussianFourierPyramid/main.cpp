#include <opencv2/opencv.hpp>
//#include <E:/Github/OpenCP/include/opencp.hpp>
//#pragma comment(lib, "opencp.lib")
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

using namespace std;
using namespace cv;

#pragma region pyramidUtility
void buildLaplacianPyramid(const Mat& src, vector<Mat>& destPyramid, const int level)
{
	if (destPyramid.size() != level + 1) destPyramid.resize(level + 1);

	buildPyramid(src, destPyramid, level);
	for (int i = 0; i < level; i++)
	{
		Mat temp;
		pyrUp(destPyramid[i + 1], temp, destPyramid[i].size());
		subtract(destPyramid[i], temp, destPyramid[i]);
	}
}

void collapseLaplacianPyramid(vector<Mat>& LaplacianPyramid, Mat& dest)
{
	const int level = (int)LaplacianPyramid.size();
	Mat ret;
	cv::pyrUp(LaplacianPyramid[level - 1], ret, LaplacianPyramid[level - 2].size());
	for (int i = level - 2; i != 0; i--)
	{
		cv::add(ret, LaplacianPyramid[i], ret);
		cv::pyrUp(ret, ret, LaplacianPyramid[i - 1].size());
	}
	cv::add(ret, LaplacianPyramid[0], dest);
}
#pragma endregion

//fast Local Laplacian Filter(2014)
class FastLLF
{
private:
	float intensityMin = 0.f;
	float intensityMax = 255.f;
	float intensityRange = 255.f;
	const int rangeMax = 256;

	//double intensityMin, intensityMax;
	std::vector<cv::Mat> GaussianPyramid;
	std::vector<std::vector<cv::Mat>> LaplacianPyramidOrder;
	std::vector<cv::Mat> LaplacianPyramid;

	float getTau(const int k, const int order)
	{
		const float delta = intensityRange / (order - 1);
		return float(k * delta + intensityMin);
	}
	//last level is not blended; thus, inplace operation for input Gaussian Pyramid is required.

	inline void getLinearIndex(float v, int& index_l, int& index_h, float& alpha, const int order, const float intensityMin, const float intensityMax)
	{
		const float intensityRange = intensityMax - intensityMin;
		const float delta = intensityRange / (order - 1);
		const int i = (int)(v / delta);
		/*cout << step << endl;
		cout << i * step << ":" << v << endl;*/
		//cout << "check:" << (v - (i * step)) / step << endl;

		if (i < 0)
		{
			//cout << "-sign" << endl;
			index_l = 0;
			index_h = 0;
			alpha = 1.f;
		}
		else if (i + 1 > order - 1)
		{
			//cout << "-sign" << endl;
			index_l = order - 1;
			index_h = order - 1;
			alpha = 0.f;
		}
		else
		{
			index_l = i;
			index_h = i + 1;
			alpha = 1.f - (v - (i * delta)) / (delta);
			//const float vv = (i * step) * alpha + ((i + 1) * step) * (1.0 - alpha);
			//print_debug5(v, i * step, step, alpha,vv);
		}
	}

	//do not handle last level
	void blendLaplacianLinear(const std::vector<std::vector<cv::Mat>>& LaplacianPyramid, std::vector<cv::Mat>& GaussianPyramid, std::vector<cv::Mat>& destPyramid, const int order)
	{
		const int level = (int)GaussianPyramid.size();
		destPyramid.resize(level);
		AutoBuffer<const float*> lptr(order);
		for (int l = 0; l < level - 1; l++)
		{
			destPyramid[l].create(GaussianPyramid[l].size(), CV_32F);
			float* g = GaussianPyramid[l].ptr<float>();
			float* d = destPyramid[l].ptr<float>();
			for (int k = 0; k < order; k++)
			{
				lptr[k] = LaplacianPyramid[k][l].ptr<float>();
			}

			for (int i = 0; i < GaussianPyramid[l].size().area(); i++)
			{
				float alpha;
				int high, low;
				getLinearIndex(g[i], low, high, alpha, order, intensityMin, intensityMax);
				d[i] = alpha * lptr[low][i] + (1.f - alpha) * lptr[high][i];
			}
		}
	}

	void remap(const Mat& src, Mat& dest, const float g, const float sigma_range, const float boost)
	{
		if (src.data != dest.data) dest.create(src.size(), CV_32F);

		const int size = src.size().area();
		const float* s = src.ptr<float>();
		float* d = dest.ptr<float>();
		const float coeff = float(1.0 / (-2.0 * sigma_range * sigma_range));
		for (int i = 0; i < size; i++)
		{
			const float x = s[i] - g;
			d[i] = x * boost * exp(x * x * coeff) + s[i];
		}
	}

public:
	//grayscale processing
	void gray(const cv::Mat& src, cv::Mat& dest, const int order, const float sigma_range, const float boost, const int level)
	{
		//(0) alloc
		if (GaussianPyramid.size() != level + 1)GaussianPyramid.resize(level + 1);
		LaplacianPyramidOrder.resize(order);
		for (int n = 0; n < order; n++)
		{
			LaplacianPyramidOrder[n].resize(level + 1);
		}
		if (src.depth() == CV_32F) src.copyTo(GaussianPyramid[0]);
		else src.convertTo(GaussianPyramid[0], CV_32F);

		//(1) build Gaussian Pyramid
		cv::buildPyramid(GaussianPyramid[0], GaussianPyramid, level);

		//(2) build remapped Laplacian Pyramids
#pragma omp parallel for
		for (int n = 0; n < order; n++)
		{
			//(2)-1 Remap Input Image
			if (true)
			{
				remap(GaussianPyramid[0], LaplacianPyramidOrder[n][0], getTau(n, order), sigma_range, boost);
			}
			else
			{
				//remapAdaptive(GaussianPyramid[0], LaplacianPyramidOrder[n][0], getTau(n), adaptiveSigmaMap[0], adaptiveBoostMap[0]);
			}

			//(2)-2 Build Remapped Laplacian Pyramids
			buildLaplacianPyramid(LaplacianPyramidOrder[n][0], LaplacianPyramidOrder[n], level);
		}

		//(3) interpolate Laplacian Pyramid from Remapped Laplacian Pyramids
		blendLaplacianLinear(LaplacianPyramidOrder, GaussianPyramid, LaplacianPyramid, order);
		//set last level
		LaplacianPyramid[level] = GaussianPyramid[level];

		//(4) collapse Laplacian Pyramid to the last level
		collapseLaplacianPyramid(LaplacianPyramid, LaplacianPyramid[0]);

		LaplacianPyramid[0].convertTo(dest, src.depth());//convert 32F to output type
	}

	void filter(const cv::Mat& src, cv::Mat& dest, const int order, const float sigma_range, const float boost = 1.f, const int level = 2)
	{
		dest.create(src.size(), src.type());
		if (src.channels() == 1)
		{
			gray(src, dest, order, sigma_range, boost, level);
		}
		else
		{
			const bool onlyY = true;
			if (onlyY)
			{
				Mat gim;
				cv::cvtColor(src, gim, COLOR_BGR2YUV);
				vector<Mat> vsrc;
				split(gim, vsrc);
				cout << vsrc[0].depth() << endl;
				gray(vsrc[0], vsrc[0], order, sigma_range, boost, level);
				merge(vsrc, dest);	
				cv::cvtColor(dest, dest, COLOR_YUV2BGR);
			}
			else
			{
				vector<Mat> vsrc;
				vector<Mat> vdst(3);
				split(src, vsrc);
				gray(vsrc[0], vdst[0], order, sigma_range, boost, level);
				gray(vsrc[1], vdst[1], order, sigma_range, boost, level);
				gray(vsrc[2], vdst[2], order, sigma_range, boost, level);
				merge(vdst, dest);
			}
		}
	}
};

#pragma endregion

int main()
{
	Mat src = imread("flower.png");
	Mat dest;
	FastLLF llf;
	llf.filter(src, dest, 8, 30, 3, 2);

	imshow("LLF src", src);
	imshow("LLF dest", dest);
	waitKey();
	return 0;
}