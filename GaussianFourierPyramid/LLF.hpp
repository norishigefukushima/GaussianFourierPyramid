#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#pragma region pyramidUtility
//image input version
static void buildLaplacianPyramid(const cv::Mat& src, std::vector<cv::Mat>& destPyramid, const int level)
{
	if (destPyramid.size() != level + 1) destPyramid.resize(level + 1);

	cv::buildPyramid(src, destPyramid, level);
	for (int i = 0; i < level; i++)
	{
		cv::Mat temp;
		cv::pyrUp(destPyramid[i + 1], temp, destPyramid[i].size());
		cv::subtract(destPyramid[i], temp, destPyramid[i]);
	}
}
//pyramid input version
static void buildLaplacianPyramid(const std::vector<cv::Mat>& GaussianPyramid, std::vector<cv::Mat>& destPyramid, const int level)
{
	if (destPyramid.size() != level + 1) destPyramid.resize(level + 1);

	for (int l = 0; l < level; l++)
	{
		cv::Mat temp;
		cv::pyrUp(GaussianPyramid[l + 1], temp, GaussianPyramid[l].size());
		cv::subtract(GaussianPyramid[l], temp, destPyramid[l]);
	}
}

static void collapseLaplacianPyramid(const std::vector<cv::Mat>& LaplacianPyramid, cv::Mat& dest)
{
	const int level = (int)LaplacianPyramid.size();
	cv::Mat ret;
	cv::pyrUp(LaplacianPyramid[level - 1], ret, LaplacianPyramid[level - 2].size());
	for (int i = level - 2; i != 0; i--)
	{
		cv::add(ret, LaplacianPyramid[i], ret);
		cv::pyrUp(ret, ret, LaplacianPyramid[i - 1].size());
	}
	cv::add(ret, LaplacianPyramid[0], dest);
}
#pragma endregion

#pragma region LLF
// Fast Local Laplacian Filter
// M. Aubry, S. Paris, J. Kautz, and F. Durand, ÅgFast local laplacian filters: Theory and applications,ÅhACM Transactionson Graphics, vol. 33, no. 5, 2014.
// Z. Qtang, L. He, Y. Chen, X. Chen, and D. Xu, ÅgAdaptive fast local laplacian filtersand its edge - aware application,ÅhMultimedia Toolsand Applications, vol. 78, pp. 5, 2019.
class FastLLF
{
private:
	const float intensityMin = 0.f;
	const float intensityMax = 255.f;
	const float intensityRange = 255.f;
	const int rangeMax = 256;

	std::vector<cv::Mat> GaussianPyramid;
	std::vector<std::vector<cv::Mat>> LaplacianPyramidOrder;
	std::vector<cv::Mat> LaplacianPyramid;

	bool isAdaptive = false;
	float sigma_range = 0.f;
	float boost = 1.f;
	cv::Mat adaptiveSigmaMap;
	cv::Mat adaptiveBoostMap;

	//compute interval parameter in linear interpolation
	float getTau(const int k, const int order)
	{
		const float delta = intensityRange / (order - 1);
		return float(k * delta + intensityMin);
	}

	void remap(const cv::Mat& src, cv::Mat& dest, const float g, const float sigma_range, const float boost)
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

	void remapAdaptive(const cv::Mat& src, cv::Mat& dest, const float g, const cv::Mat& sigma_range, const cv::Mat& boost)
	{
		if (src.data != dest.data) dest.create(src.size(), CV_32F);

		const int size = src.size().area();
		const float* s = src.ptr<float>();
		float* d = dest.ptr<float>();
		const float* sigmaptr = sigma_range.ptr<float>();
		const float* boostptr = boost.ptr<float>();

		for (int i = 0; i < size; i++)
		{
			const float coeff = 1.f / (-2.f * sigmaptr[i] * sigmaptr[i]);
			const float boost = boostptr[i];
			const float x = s[i] - g;
			d[i] = x * boost * exp(x * x * coeff) + s[i];
		}
	}

	inline void getLinearIndex(float v, int& index_l, int& index_h, float& alpha, const int order, const float intensityMin, const float intensityMax)
	{
		const float intensityRange = intensityMax - intensityMin;
		const float delta = intensityRange / (order - 1);
		const int i = (int)(v / delta);

		if (i < 0)
		{
			index_l = 0;
			index_h = 0;
			alpha = 1.f;
		}
		else if (i + 1 > order - 1)
		{
			index_l = order - 1;
			index_h = order - 1;
			alpha = 0.f;
		}
		else
		{
			index_l = i;
			index_h = i + 1;
			alpha = 1.f - (v - (i * delta)) / (delta);
		}
	}

	//last level is not blended; thus, inplace operation for input Gaussian Pyramid is required.
	void blendLaplacianLinear(const std::vector<std::vector<cv::Mat>>& LaplacianPyramid, std::vector<cv::Mat>& GaussianPyramid, std::vector<cv::Mat>& destPyramid, const int order)
	{
		const int level = (int)GaussianPyramid.size();
		destPyramid.resize(level);
		std::vector<const float*> lptr(order);
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

	void setAdaptive(const cv::Mat& sigmaMap, const cv::Mat& boostMap)
	{
		isAdaptive = true;
		this->adaptiveSigmaMap = sigmaMap;
		this->adaptiveBoostMap = boostMap;
	}

	void setFix(const float sigma_range, const float boost)
	{
		isAdaptive = false;
		this->sigma_range = sigma_range;
		this->boost = boost;
	}

	//grayscale processing
	void gray(const cv::Mat& src, cv::Mat& dest, const int order, const int level)
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

		//(1) Build Gaussian Pyramid
		cv::buildPyramid(GaussianPyramid[0], GaussianPyramid, level);

		//(2) Build remapped Laplacian Pyramids
		for (int n = 0; n < order; n++)
		{
			//(2-1) Remap Input Image
			if (isAdaptive)
			{
				remapAdaptive(GaussianPyramid[0], LaplacianPyramidOrder[n][0], getTau(n, order), adaptiveSigmaMap, adaptiveBoostMap);
			}
			else
			{
				remap(GaussianPyramid[0], LaplacianPyramidOrder[n][0], getTau(n, order), sigma_range, boost);
			}

			//(2-2) Build Remapped Laplacian Pyramids
			buildLaplacianPyramid(LaplacianPyramidOrder[n][0], LaplacianPyramidOrder[n], level);
		}

		//(3) interpolate Laplacian Pyramid from Remapped Laplacian Pyramids
		blendLaplacianLinear(LaplacianPyramidOrder, GaussianPyramid, LaplacianPyramid, order);
		//set last level
		LaplacianPyramid[level] = GaussianPyramid[level];

		//(4) Collapse Laplacian Pyramid to the last level
		collapseLaplacianPyramid(LaplacianPyramid, LaplacianPyramid[0]);

		LaplacianPyramid[0].convertTo(dest, src.depth());//convert 32F to output type
	}

	//main processing (same methods: Fast LLF and Fourier LLF)
	void body(const cv::Mat& src, cv::Mat& dest, const int order, const int level)
	{
		dest.create(src.size(), src.type());
		if (src.channels() == 1)
		{
			gray(src, dest, order, level);
		}
		else
		{
			const bool onlyY = true;
			if (onlyY)
			{
				cv::Mat gim;
				cv::cvtColor(src, gim, cv::COLOR_BGR2YUV);
				std::vector<cv::Mat> vsrc;
				cv::split(gim, vsrc);
				gray(vsrc[0], vsrc[0], order, level);
				merge(vsrc, dest);
				cv::cvtColor(dest, dest, cv::COLOR_YUV2BGR);
			}
			else
			{
				std::vector<cv::Mat> vsrc;
				std::vector<cv::Mat> vdst(3);
				cv::split(src, vsrc);
				gray(vsrc[0], vdst[0], order, level);
				gray(vsrc[1], vdst[1], order, level);
				gray(vsrc[2], vdst[2], order, level);
				cv::merge(vdst, dest);
			}
		}
	}
public:
	//fix parameter (sigma_range and boost)
	void filter(const cv::Mat& src, cv::Mat& dest, const int order, const float sigma_range, const float boost, const int level = 2)
	{
		setFix(sigma_range, boost);
		body(src, dest, order, level);
	}
	//adaptive parameter (sigma_range and boost)
	void filter(const cv::Mat& src, cv::Mat& dest, const int order, const cv::Mat& sigma_range, const cv::Mat& boost, const int level = 2)
	{
		setAdaptive(sigma_range, boost);
		body(src, dest, order, level);
	}
};

// Fourier Local Laplacian Filter
// Y. Sumiya, T. Otsuka, Y. Maedaand N. Fukushima, "Gaussian Fourier Pyramid for Local Laplacian Filter," IEEE Signal Processing Letters, vol. 29, pp. 11 - 15, 2022.
class GaussianFourierLLF
{
private:
	const float intensityMin = 0.f;
	const float intensityMax = 255.f;
	const float intensityRange = 255.f;
	const int rangeMax = 256;

	float T = 0.f;
	std::vector<float> alpha, beta;
	std::vector<float> omega;//(CV_2PI*(k + 1)/T)

	std::vector<cv::Mat> FourierPyramidSin; //level+1
	std::vector<cv::Mat> FourierPyramidCos; //level+1
	std::vector<cv::Mat> LaplacianPyramid; //level+1
	std::vector<cv::Mat> GaussianPyramid; //level+1

	bool isAdaptive = false;
	float sigma_range = 0.f;
	float boost = 1.f;
	int level;
	std::vector<cv::Mat> adaptiveSigmaMap;
	std::vector<cv::Mat> adaptiveBoostMap;

	double df(double x, const int K, const double Irange, const double sigma_range)
	{
		const double s = sigma_range / Irange;
		const double kappa = (2 * K + 1) * CV_PI;
		const double psi = kappa * s / x;
		const double phi = (x - 1.0) / s;
		return (-kappa * exp(-phi * phi) + psi * psi * exp(-psi * psi));
	}

	double computeT_ClosedForm(int order, double sigma_range, const double intensityRange)
	{
		double x, diff;

		double x1 = 1.0, x2 = 15.0;
		int loop = 20;
		for (int i = 0; i < loop; ++i)
		{
			x = (x1 + x2) / 2.0;
			diff = df(x, order, intensityRange, sigma_range);
			((0.0 <= diff) ? x2 : x1) = x;
		}
		return x;
	}

	void initRangeFourier(const int order, const float sigma_range, const float boost)
	{
		if (alpha.size() != order)
		{
			alpha.resize(order);
			beta.resize(order);
		}

		if (omega.size() != order) omega.resize(order);

		T = float(intensityRange * computeT_ClosedForm(order, sigma_range, intensityRange));

		for (int k = 0; k < order; k++)
		{
			omega[k] = float(CV_2PI / (double)T * (double)(k + 1));
			const double coeff_kT = omega[k] * sigma_range;
			alpha[k] = float(2.0 * exp(-0.5 * coeff_kT * coeff_kT) * sqrt(CV_2PI) * sigma_range / T);
		}
	}

	void remapCos(const cv::Mat& src, cv::Mat& dest, const float omega)
	{
		dest.create(src.size(), CV_32F);
		const float* s = src.ptr<float>();
		float* d = dest.ptr<float>();
		const int size = src.size().area();

		for (int i = 0; i < size; i++)
		{
			d[i] = cos(omega * s[i]);
		}
	}

	void remapSin(const cv::Mat& src, cv::Mat& dest, const float omega)
	{
		dest.create(src.size(), CV_32F);
		const float* s = src.ptr<float>();
		float* d = dest.ptr<float>();
		const int size = src.size().area();

		for (int i = 0; i < size; i++)
		{
			d[i] = sin(omega * s[i]);
		}
	}

	void productSumPyramidLayer(const cv::Mat& srccos, const cv::Mat& srcsin, const cv::Mat gauss, cv::Mat& dest, const float omega, const float alpha, const float sigma, const float boost)
	{
		dest.create(srccos.size(), CV_32F);

		const int size = srccos.size().area();

		const float* c = srccos.ptr<float>();
		const float* s = srcsin.ptr<float>();
		const float* g = gauss.ptr<float>();
		float* d = dest.ptr<float>();

		const float lalpha = -sigma * sigma * omega * alpha * boost;

		for (int i = 0; i < size; i++)
		{
			const float ms = omega * g[i];
			d[i] += lalpha * (sin(ms) * c[i] - cos(ms) * (s[i]));
		}
	}

	inline float getAdaptiveAlpha(float coeff, float base, float sigma, float boost)
	{
		const float a = coeff * sigma;
		return sigma * sigma * sigma * boost * base * exp(-0.5f * a * a);
	}

	void productSumAdaptivePyramidLayer(const cv::Mat& srccos, const cv::Mat& srcsin, const cv::Mat gauss, cv::Mat& dest, const float omega, const float alpha, const cv::Mat& sigma, const cv::Mat& boost)
	{
		dest.create(srccos.size(), CV_32F);

		const int size = srccos.size().area();

		const float* c = srccos.ptr<float>();
		const float* s = srcsin.ptr<float>();
		const float* g = gauss.ptr<float>();
		float* d = dest.ptr<float>();

		const float base = -float(2.0 * sqrt(CV_2PI) * omega / T);
		const float* adaptiveSigma = sigma.ptr<float>();
		const float* adaptiveBoost = boost.ptr<float>();

		for (int i = 0; i < size; i++)
		{
			const float lalpha = getAdaptiveAlpha(omega, base, adaptiveSigma[i], adaptiveBoost[i]);
			const float ms = omega * g[i];
			d[i] += lalpha * (sin(ms) * c[i] - cos(ms) * (s[i]));
		}
	}

	void setAdaptive(const cv::Mat& sigmaMap, const cv::Mat& boostMap, const int level)
	{
		isAdaptive = true;
		cv::buildPyramid(sigmaMap, adaptiveSigmaMap, level);
		cv::buildPyramid(boostMap, adaptiveBoostMap, level);
	}

	void setFix(const float sigma_range, const float boost)
	{
		isAdaptive = false;
		this->sigma_range = sigma_range;
		this->boost = boost;
	}

	//grayscale processing
	void gray(const cv::Mat& src, cv::Mat& dest, const int order, const int level)
	{
		//(0) alloc
		GaussianPyramid.resize(level + 1);
		FourierPyramidCos.resize(level + 1);
		FourierPyramidSin.resize(level + 1);
		if (src.depth() == CV_8U) src.convertTo(GaussianPyramid[0], CV_32F);
		else src.copyTo(GaussianPyramid[0]);

		//compute alpha, omega, T
		initRangeFourier(order, sigma_range, boost);


		//(1) Build Gaussian Pyramid
		cv::buildPyramid(GaussianPyramid[0], GaussianPyramid, level);

		//(2) Build Laplacian Pyramid
		//(2-1) Build Laplacian Pyramid for DC
		buildLaplacianPyramid(GaussianPyramid, LaplacianPyramid, level);

		for (int k = 0; k < order; k++)
		{
			// (2-2) Build Remapped Laplacian Pyramid for Cos
			remapCos(GaussianPyramid[0], FourierPyramidCos[0], omega[k]);
			buildLaplacianPyramid(FourierPyramidCos[0], FourierPyramidCos, level);
			// (2-3) Build Remapped Laplacian Pyramid for Sin
			remapSin(GaussianPyramid[0], FourierPyramidSin[0], omega[k]);
			buildLaplacianPyramid(FourierPyramidSin[0], FourierPyramidSin, level);

			// (3) product-sum Gaussian Fourier pyramid
			if (isAdaptive)
			{
				for (int l = 0; l < level; l++)
				{
					productSumAdaptivePyramidLayer(FourierPyramidCos[l], FourierPyramidSin[l], GaussianPyramid[l], LaplacianPyramid[l], omega[k], alpha[k], adaptiveSigmaMap[l], adaptiveBoostMap[l]);
				}
			}
			else
			{
				for (int l = 0; l < level; l++)
				{
					productSumPyramidLayer(FourierPyramidCos[l], FourierPyramidSin[l], GaussianPyramid[l], LaplacianPyramid[l], omega[k], alpha[k], sigma_range, boost);
				}
			}
		}
		//set last level
		LaplacianPyramid[level] = GaussianPyramid[level];

		//(4) Collapse Laplacian Pyramid
		collapseLaplacianPyramid(LaplacianPyramid, LaplacianPyramid[0]);

		LaplacianPyramid[0].convertTo(dest, src.depth());//convert 32F to output type
	}

	//main processing (same methods: Fast LLF and Fourier LLF)
	void body(const cv::Mat& src, cv::Mat& dest, const int order, const int level)
	{
		dest.create(src.size(), src.type());
		if (src.channels() == 1)
		{
			gray(src, dest, order, level);
		}
		else
		{
			const bool onlyY = true;
			if (onlyY)
			{
				cv::Mat gim;
				cv::cvtColor(src, gim, cv::COLOR_BGR2YUV);
				std::vector<cv::Mat> vsrc;
				cv::split(gim, vsrc);
				gray(vsrc[0], vsrc[0], order, level);
				merge(vsrc, dest);
				cv::cvtColor(dest, dest, cv::COLOR_YUV2BGR);
			}
			else
			{
				std::vector<cv::Mat> vsrc;
				std::vector<cv::Mat> vdst(3);
				cv::split(src, vsrc);
				gray(vsrc[0], vdst[0], order, level);
				gray(vsrc[1], vdst[1], order, level);
				gray(vsrc[2], vdst[2], order, level);
				cv::merge(vdst, dest);
			}
		}
	}
public:	
	//fix parameter (sigma_range and boost)
	void filter(const cv::Mat& src, cv::Mat& dest, const int order, const float sigma_range, const float boost, const int level = 2)
	{
		setFix(sigma_range, boost);
		body(src, dest, order, level);
	}
	//adaptive parameter (sigma_range and boost)
	void filter(const cv::Mat& src, cv::Mat& dest, const int order, const cv::Mat& sigma_range, const cv::Mat& boost, const int level = 2)
	{
		setAdaptive(sigma_range, boost, level);
		body(src, dest, order, level);
	}
};
#pragma endregion