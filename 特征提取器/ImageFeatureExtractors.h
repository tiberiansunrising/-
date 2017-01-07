#pragma once
///标准库
#include<vector>
#include<array>
#include<memory>
#include<fstream>
///第三方库
#include<opencv2/opencv.hpp>

namespace Kagamine
{
	namespace cvex
	{
		class IImageFeatureExtractor//特征提取接口
		{
		public:
			int dims = 0;
			virtual cv::Mat Extract(const cv::Mat &src) = 0;
		protected:
			virtual cv::Mat ImagePreProcessing(const cv::Mat &src) = 0;
			virtual cv::Mat ExtractFeatures(const cv::Mat &src) = 0;
			virtual cv::Mat DimensionReduction(const cv::Mat &src) = 0;
		};

		class HJGaborFeatureExtractor :public IImageFeatureExtractor
		{
		public:
			HJGaborFeatureExtractor();
			//输入图像为单通道灰度图
			cv::Mat Extract(const cv::Mat &src);
			cv::Mat PreProcessing(const cv::Mat &src);
		protected:
			cv::Mat ImagePreProcessing(const cv::Mat &src);
			cv::Mat ExtractFeatures(const cv::Mat &src);
			cv::Mat DimensionReduction(const cv::Mat &src);
		private:
			///成员变量
			const unsigned int CellSize = 64;
			const int V_Max = 5, Mu_Max = 8;
			const double Kmax = CV_PI / 2;

			cv::Mat GabKRe[40], GabKIm[40];
			cv::Mat KernelDOG;//DOG核
			///成员函数
			///预载相关
			cv::Mat GuassBase(const cv::Size size, const int delta);
			cv::Mat GetKernel(const cv::Size &sz, const int delta_1, const int delta_2);
			void HJFeatureBase(int V, int Mu, int size, cv::Mat &Mat_Re, cv::Mat &Mat_Im);
			///预处理相关
			void Gamma(const cv::Mat &src, cv::Mat &dst, double g);
			void DOG(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernels);
			void ContEqu(const cv::Mat &src, cv::Mat &dst, double tao = 0.0);
			void Tangent(const cv::Mat &src, cv::Mat &dst);
			///特征提取相关
			void HJDS_v2f(cv::Mat mat, int n, std::vector<float> &ds_arr);
			double HJArrMeanf(std::vector<float> &arr);
			double HJArrDevf(std::vector<float> &arr, double arr_mean);
			void HJNormArrf(std::vector<float> &arr);
			void HJFeatureCalc(const cv::Mat &src, const cv::Mat &Mat_Re, const cv::Mat &Mat_Im, cv::Mat &Mat_Mod);
			cv::Mat HJGaborFeatExtra(const cv::Mat &src);
		};

		class HJPOEMFeatureExtractor :public IImageFeatureExtractor
		{
		public:
			HJPOEMFeatureExtractor();
			//输入图像为单通道灰度图
			cv::Mat Extract(const cv::Mat &src);
			cv::Mat ExtractWithOutDR(const cv::Mat &src);
			cv::Mat PreProcessing(const cv::Mat &src);
		protected:
			cv::Mat ImagePreProcessing(const cv::Mat &src);
			cv::Mat ExtractFeatures(const cv::Mat &src);
			cv::Mat DimensionReduction(const cv::Mat &src);
		private:
			///预处理相关
			cv::Mat GuassBase(const cv::Size size, const int delta);
			cv::Mat GetKernel(const cv::Size &sz, const int delta_1, const int delta_2);
			void Gamma(const cv::Mat &src, cv::Mat &dst, double g);
			void DOG(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernels);
			void ContEqu(const cv::Mat &src, cv::Mat &dst, double tao = 0.0);
			void Tangent(const cv::Mat &src, cv::Mat &dst);
			cv::Mat KernelDOG;
			//Initial variables
			int val_ang = 3;	//Orientation of Angles
			int val_neighbor = 6;
			int val_cellsize = 7;	//HOG cell size
			int val_blocksize = 10;	//LBP size
			int nW = 8;	//Uncovered patch num
			int nH = 8;
			//	int val_neighbor = 6;	//LBP neighbors
			//Initial LBP settings
			cv::Mat mMtx, prjMtx;
			int LBPnum;
			std::vector<int> LBPTable;
			cv::Mat LoadMtx2(std::wstring strPath, int nrows, int ncols);
			void GetLBPTable(int neighbor, std::vector<int> &LBPTable, int &LBPnum);
			int HJRotateLeft1(int Val, int nSize);
			int HJCalcJumps(int Val1, int Val2, int nSize);
			void HJHOG(const cv::Mat &src, cv::Mat &Mag, cv::Mat &Ang);
			void HJDecomp(cv::Mat &Mag, cv::Mat &Ang, int m, std::vector<cv::Mat> &Ms);
			void HJAccumCeil(cv::Mat &Im, int ceil_size, cv::Mat &Op);
			double vecMin(double *src, int Len);
			double vecMax(double *src, int Len);
			cv::Size iFastPreLBP(cv::Mat &ipl_src, int neighbors, int radius, int mode);
			void FastPreLBP(cv::Mat &ipl_src, cv::Mat &ipl_dst, int mode, int neighbors, int radius);
			//void matCMPGE(const cv::Mat &matl, const cv::Mat &matr, cv::Mat &mat_dst);
			void HJFastPreLBP(cv:: Mat &Im, cv::Mat &Op, int mode, int neighbor, int radius);
			void BlockIpl(cv::Mat &src, int nX, int nY, std::vector<cv::Rect> &mblock);
			cv::Mat LBP2Hist(const cv::Mat &src,cv::Rect r, std::vector<int> &LBPtable,int &num);
			cv::Mat LBP2Hist(const cv::Mat &src, cv::Rect r, cv::Mat &ulbp, std::vector<int> &LBPtable, int &num);
			cv::Mat HJFeatPOEM(const cv::Mat &src);
		};
	}
}