#include "ImageFeatureExtractors.h"

namespace Kagamine
{
	namespace cvex
	{
		HJGaborFeatureExtractor::HJGaborFeatureExtractor()
		{
			dims = 10240;
			KernelDOG = GetKernel(cv::Size(15, 15), 1, 2);
			int img_size = 32;
			for (int V = 0; V < V_Max; ++V)
			{
				for (int Mu = 0; Mu < Mu_Max; ++Mu)
				{
					// 1. Calculate the Gabor wavelet base
					GabKRe[V * 8 + Mu] = cv::Mat(img_size, img_size, CV_32FC1);
					GabKIm[V * 8 + Mu] = cv::Mat(img_size, img_size, CV_32FC1);
					HJFeatureBase(V, Mu, img_size, GabKRe[V * 8 + Mu], GabKIm[V * 8 + Mu]);
				}
			}
		}
		cv::Mat HJGaborFeatureExtractor::PreProcessing(const cv::Mat &src)
		{
			return ImagePreProcessing(src);
		}
		cv::Mat HJGaborFeatureExtractor::Extract(const cv::Mat &src)
		{
			return DimensionReduction(ExtractFeatures(ImagePreProcessing(src)));
		}

		cv::Mat HJGaborFeatureExtractor::DimensionReduction(const cv::Mat &src)
		{
			return src;
		}

		cv::Mat HJGaborFeatureExtractor::ImagePreProcessing(const cv::Mat &src)
		{
			double minval, maxval;
			cv::Mat mat, dst;
			src.convertTo(mat, CV_32FC1, 1.0 / 255.0);

			Gamma(mat, mat, 0.2);
			DOG(mat, mat, KernelDOG);
			ContEqu(mat, mat);
			ContEqu(mat, mat, 10);
			Tangent(mat, mat);
			cv::minMaxLoc(mat, &minval, &maxval);
			mat.convertTo(dst, CV_8UC1, 255.0f / (maxval - minval), -minval*255.f / (maxval - minval));
			cv::resize(dst, dst, cv::Size(CellSize, CellSize));
			//cv::imshow("lk", dst);
			//cv::waitKey(1);
			return dst;
		}

		cv::Mat HJGaborFeatureExtractor::ExtractFeatures(const cv::Mat &src)
		{
			return HJGaborFeatExtra(src);
		}

		void HJGaborFeatureExtractor::HJFeatureBase(int V, int Mu, int size, cv::Mat &Mat_Re, cv::Mat &Mat_Im)
		{

			struct {
				double x;
				double y;
			}Kuv;

			int x, y;			// зјБъ
			int sup_x = size / 2;
			int inf_x = -size / 2 + 1;
			int sup_y = sup_x;
			int inf_y = inf_x;
			int size_2 = size / 2;
			double Kv = Kmax / pow(sqrt(2.0), V);   // Kmax = CV_PI / 2, F = sqrt(2)

			double Phi_u = CV_PI * Mu / 8;

			Kuv.x = Kv * cos(Phi_u);
			Kuv.y = Kv * sin(Phi_u);

			double Module_Kuv = pow(Kuv.x, 2) + pow(Kuv.y, 2);
			double Sigma_2 = pow(2 * CV_PI, 2);
			// calculate
			for (y = sup_y; y >= inf_y; --y)
			{
				for (x = inf_x; x <= sup_x; ++x)
				{
					double Temp_Re, Temp_Im;

					double Module_Z = pow((double)x, 2.0) + pow((double)y, 2.0);
					double angle = x * Kuv.x + y * Kuv.y;
					double index = -Module_Kuv * Module_Z / (2 * Sigma_2);

					Temp_Re = (Module_Kuv / Sigma_2) * exp(index) * (cos(angle) - exp(-Sigma_2 / 2));
					Temp_Im = (Module_Kuv / Sigma_2) * exp(index) * sin(angle);

					((float*)Mat_Re.data)[((x + size_2 - 1) + (-y + size_2) * Mat_Re.cols)] = Temp_Re;
					((float*)Mat_Im.data)[((x + size_2 - 1) + (-y + size_2) * Mat_Im.cols)] = Temp_Im;
				}
			}
		}

		cv::Mat HJGaborFeatureExtractor::GuassBase(const cv::Size size, const int delta)
		{
			cv::Mat mat(size, CV_32FC1);
			for (int y = 0; y < size.height; ++y)
			{
				for (int x = 0; x < size.width; ++x)
				{
					double d = sqrt((double)(y - size.height / 2)*(y - size.height / 2) + (x - size.width / 2)*(x - size.width / 2));
					((float*)mat.data)[(mat.cols * y + x)] = exp(-d*d / (2 * delta*delta)) / (2 * CV_PI*delta*delta);
				}
			}
			return mat;
		}

		cv::Mat HJGaborFeatureExtractor::GetKernel(const cv::Size &sz, const int delta_1, const int delta_2)
		{
			cv::Mat kernel1 = GuassBase(sz, delta_1);
			cv::Mat kernel2 = GuassBase(sz, delta_2);
			cv::Mat kernel(sz, CV_32FC1);
			cv::subtract(kernel1, kernel2, kernel);
			return kernel;
		}

		void HJGaborFeatureExtractor::Gamma(const cv::Mat &src, cv::Mat &dst, double g)
		{
			for (int i = 0; i < dst.rows; ++i)
			{
				for (int j = 0; j < dst.cols; ++j)
				{
					((float*)dst.data)[(i*dst.cols + j)] = pow(((float*)dst.data)[i*dst.cols + j], g);
				}
			}
		}
		void HJGaborFeatureExtractor::DOG(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernels)
		{
			cv::filter2D(src, dst, dst.depth(), kernels, cv::Point(7, 7));
		}
		void HJGaborFeatureExtractor::ContEqu(const cv::Mat &src, cv::Mat &dst, double tao)
		{
			double alpha = 0.1;
			double sum = 0;
			for (int i = 0; i < src.rows; ++i)
			{
				for (int j = 0; j < src.cols; ++j)
				{
					double tmp = fabs(((float*)src.data)[src.cols * i + j]);

					if (tao > 0)
					{
						tmp = tmp < tao ? tmp : tao;
					}

					if (tmp > 0)
					{
						sum += pow(tmp, alpha);
					}
				}
			}
			sum /= (src.rows*src.cols);
			dst = src*pow(sum, -1.0 / alpha);
		}
		void HJGaborFeatureExtractor::Tangent(const cv::Mat &src, cv::Mat &dst)
		{
			for (int i = 0; i < src.rows; ++i)
			{
				for (int j = 0; j < src.cols; ++j)
				{
					double tmp = ((float*)src.data)[src.cols * i + j];
					double exp1 = exp(tmp / 10);
					double exp2 = exp(-tmp / 10);
					double res = (exp1 - exp2) / (exp1 + exp2);
					((float*)dst.data)[(src.cols * i + j)] = 10 * res;
				}
			}
		}

		void HJGaborFeatureExtractor::HJDS_v2f(cv::Mat mat, int n, std::vector<float> &ds_arr)
		{

			long imat_size = mat.cols * mat.rows;//64*64
			int imat_recols = (int)(mat.cols / pow(2.0, n));//16
			int imat_rerows = (int)(mat.rows / pow(2.0, n));//16
			long iarr_size = imat_rerows*imat_recols;//16*16

			cv::Mat _remat, _rematT;
			cv::resize(mat, _remat, cv::Size(imat_recols, imat_rerows), CV_INTER_AREA);
			cv::transpose(_remat, _rematT);
			int cont = 0;
			for (int r = 0; r < _rematT.rows; r++)
			{
				for (int c = 0; c < _rematT.cols; c++)
				{
					ds_arr[cont++] = ((float*)_rematT.data)[c + r*_rematT.cols];
				}
			}
		}

		double HJGaborFeatureExtractor::HJArrMeanf(std::vector<float> &arr)
		{
			double arr_mean = 0;
			for (auto &n : arr)
			{
				arr_mean += n;
			}
			return 	arr_mean /= arr.size();
		}
		double HJGaborFeatureExtractor::HJArrDevf(std::vector<float> &arr, double arr_mean)
		{
			double arr_dev = 0;// , arr_mean = 0;
							  //arr_mean = HJArrMeanf(arr, arr_size);
			for (auto &n : arr)
			{
				arr_dev += pow(n - arr_mean, 2);
			}

			return 	arr_dev /= arr.size();
		}
		void HJGaborFeatureExtractor::HJNormArrf(std::vector<float> &arr)
		{
			auto arr_mean = HJArrMeanf(arr);
			auto arr_dev = HJArrDevf(arr, arr_mean);

			for (auto &n : arr)
			{
				n = (n - arr_mean) / sqrt(arr_dev);
			}
		}
		void HJGaborFeatureExtractor::HJFeatureCalc(const cv::Mat &src, const cv::Mat &Mat_Re, const cv::Mat &Mat_Im, cv::Mat &Mat_Mod)
		{

			int x, y;
			int img_size_w = src.cols;//64
			int img_size_h = src.rows;
			cv::Mat dstmat_re(img_size_h, img_size_w, CV_32FC1);						// 1.
			cv::Mat dstmat_im(img_size_h, img_size_w, CV_32FC1);						// 2.

																						// convolute the image with the real part of the Gabor wavelet, 
																						// and the result is saved in CvMat *dstmat_re
																						// 	cvFilter2D( src, dstmat_re, Mat_Re, cvPoint( img_size / 2, img_size / 2 ) );
																						//cvFilter2D(src, dstmat_re, Mat_Re, cvPoint(Mat_Re->cols / 2, Mat_Re->rows / 2));
			cv::filter2D(src, dstmat_re, dstmat_re.depth(), Mat_Re, cv::Point(Mat_Re.cols / 2, Mat_Re.rows / 2));
			// convolute the image with the image part of the Gabor wavelet, 
			// and the result is saved in CvMat *dstmat_im
			// 	cvFilter2D( src, dstmat_im, Mat_Im, cvPoint( img_size / 2, img_size / 2 ) );
			cv::filter2D(src, dstmat_im, dstmat_im.depth(), Mat_Im, cv::Point(Mat_Im.cols / 2, Mat_Im.rows / 2));

			for (y = 0; y < img_size_w; ++y)
			{
				for (x = 0; x < img_size_h; ++x)
				{
					double ftmp1 = ((float*)dstmat_re.data)[x*dstmat_re.cols + y];
					double ftmp2 = ((float*)dstmat_im.data)[x*dstmat_im.cols + y];
					double ftmp = sqrt(ftmp1 * ftmp1 + ftmp2 * ftmp2);
					((float*)Mat_Mod.data)[(x*Mat_Mod.cols + y)] = ftmp;
				}
			}
		}

		cv::Mat HJGaborFeatureExtractor::HJGaborFeatExtra(const cv::Mat &src)
		{
			cv::imshow("face",src);
			cv::waitKey(1);
			//cv::imshow("kk", src);
			int ds_multi = 2;
			long arr_w = (long)(src.cols / pow(2.0, ds_multi));//64/4
			long arr_h = (long)(src.rows / pow(2.0, ds_multi));
			long arr_size = arr_w*arr_h;//256

			long gf_row = arr_size * V_Max * Mu_Max;//10240
			int ndims = gf_row;
			cv::Mat FF(ndims, 1, CV_32FC1);

			int img_size_w = src.cols;
			int img_size_h = src.rows;
			cv::Mat Gabor_Module(img_size_h, img_size_w, CV_32FC1);			// 64*64
			//float *ds_arr = new float[arr_size];	
			std::vector<float> ds_arr(arr_size);// 5.

			cv::Mat src_mat;
			src.convertTo(src_mat, CV_32FC1, 1 / 255.0f);
			//cv::imshow("gabor", src_mat);
			//cv::waitKey(1);
			// convert IplImage to CvMat															// 1.
			// calculate the dimension of the matrix after down sampling 
			long mat_size = img_size_h * img_size_w;
			int i = 0;
			// the scale and orientation of the Gabor wavelet
			int V = 0, Mu = 0;
			// V changes from 0 to 4
			auto cl = 64;
			cv::Mat kk(5* cl,8* cl,CV_32FC1,cv::Scalar(0));
			/*#pragma omp parallel for*/
			for (V = 0; V < 5; ++V)
			{
				for (Mu = 0; Mu < 8; ++Mu)
				{
					// 1. Calculate the Gabor wavelet base
					// 			HJFeatureBase( V, Mu, img_size, Gabor_Re, Gabor_Im );

					// 2. Convolute the source face image with the Gabor base
					HJFeatureCalc(src_mat, GabKRe[V * 8 + Mu], GabKIm[V * 8 + Mu], Gabor_Module);
					// 3. Downsampling
					// 			HJDS( Gabor_Module, ds_multi, ds_arr, arr_size );
					auto ar = cv::Rect(Mu * cl, V * cl, cl, cl);
					Gabor_Module.copyTo(kk(ar));
					HJDS_v2f(Gabor_Module, ds_multi, ds_arr);

					// 4. Normalize ds_arr to zero mean and unit variance
					HJNormArrf(ds_arr);
					// 5. save the data to CvMat *Gabor_Feature
					memcpy(&(FF.data[arr_size*(V * 8 + Mu) * sizeof(float)]), ds_arr.data(), arr_size * sizeof(float));
				}
			}	
			kk.convertTo(kk, CV_8UC1, 255.0f);
			cv::imshow("jpl",kk);// ~5.	
			cv::waitKey(1);// ~4.
			return FF;
		}
	}
}
