#include "ImageFeatureExtractors.h"

namespace Kagamine
{
	namespace cvex
	{
		HJPOEMFeatureExtractor::HJPOEMFeatureExtractor()
		{
			//dims = 6336;
			dims = 200;
			GetLBPTable(val_neighbor, LBPTable, LBPnum);
			KernelDOG = GetKernel(cv::Size(15, 15), 1, 2);
			try
			{
				mMtx = LoadMtx2(L"Modules/HJDataPOEM/m.dat", 6336, 1);
				prjMtx = LoadMtx2(L"Modules/HJDataPOEM/p.dat", 200, 6336);
			}
			catch (...)
			{
				std::wcout <<L"模型未载入" << std::endl;
			}
		}

		cv::Mat HJPOEMFeatureExtractor::LoadMtx2(std::wstring strPath, int nrows, int ncols)
		{
			cv::Mat Mtx(nrows, ncols, CV_32FC1);
			std::ifstream ifs(std::string(strPath.begin(), strPath.end()), std::ios::binary | std::ios::in);
			if (!ifs.is_open())throw std::exception(std::logic_error("POEM降维文件读取失败"));
			ifs.read((char*)Mtx.data, nrows*ncols*Mtx.elemSize());
			ifs.close();
			return Mtx;
		}

		cv::Mat HJPOEMFeatureExtractor::Extract(const cv::Mat &src)
		{
			return DimensionReduction(ExtractFeatures(ImagePreProcessing(src)));
		}
		cv::Mat HJPOEMFeatureExtractor::ExtractWithOutDR(const cv::Mat &src)
		{
			return ExtractFeatures(ImagePreProcessing(src));
		}

		cv::Mat HJPOEMFeatureExtractor::DimensionReduction(const cv::Mat &src)
		{
			//cv::sqrt(src, src);
			//return src;
			return prjMtx*(src - mMtx);
		}
		cv::Mat HJPOEMFeatureExtractor::GuassBase(const cv::Size size, const int delta)
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

		void  HJPOEMFeatureExtractor::Gamma(const cv::Mat &src, cv::Mat &dst, double g)
		{
			for (int i = 0; i < dst.rows; ++i)
			{
				for (int j = 0; j < dst.cols; ++j)
				{
					((float*)dst.data)[(i*dst.cols + j)] = pow(((float*)dst.data)[i*dst.cols + j], g);
				}
			}
		}
		void HJPOEMFeatureExtractor::DOG(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernels)
		{
			cv::filter2D(src, dst, dst.depth(), kernels, cv::Point(7, 7));
		}
		void  HJPOEMFeatureExtractor::ContEqu(const cv::Mat &src, cv::Mat &dst, double tao)
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
		void  HJPOEMFeatureExtractor::Tangent(const cv::Mat &src, cv::Mat &dst)
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
		cv::Mat  HJPOEMFeatureExtractor::GetKernel(const cv::Size &sz, const int delta_1, const int delta_2)
		{
			cv::Mat kernel1 = GuassBase(sz, delta_1);
			cv::Mat kernel2 = GuassBase(sz, delta_2);
			cv::Mat kernel(sz, CV_32FC1);
			cv::subtract(kernel1, kernel2, kernel);
			return kernel;
		}
		cv::Mat HJPOEMFeatureExtractor::PreProcessing(const cv::Mat &src)
		{
			return ImagePreProcessing(src);
		}
		cv::Mat HJPOEMFeatureExtractor::ImagePreProcessing(const cv::Mat &src)
		{
			///POEM不进行图像预处理
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
			cv::resize(dst, dst, cv::Size(128,128));
			cv::imshow("lk", dst);
			cv::waitKey(1);
			//cv::imwrite("Modules/Saves/img"+std::to_string(i++)+".bmp", res);
			///大小缩放
			return dst;
		}
		cv::Mat HJPOEMFeatureExtractor::ExtractFeatures(const cv::Mat &src)
		{
			return HJFeatPOEM(src);
		}

		int HJPOEMFeatureExtractor::HJRotateLeft1(int Val, int nSize)
		{
			//Val must less than 2^nSize-1
			int newVal;

			//Get high-bit number
			int Mask = 1 << (nSize - 1);
			int highbit = Val & Mask;
			highbit = (highbit == 0) ? 0 : 1;
			//Shift left by 1
			int shiftVal = Val << 1;
			//Cut to nSzie
			if (shiftVal > (int)(pow(2.f, nSize)) - 1)
			{
				newVal = shiftVal - (1 << nSize) + highbit;
			}
			else
				newVal = shiftVal + highbit;
			return newVal;
		}

		int  HJPOEMFeatureExtractor::HJCalcJumps(int Val1, int Val2, int nSize)
		{
			int ans = Val1^Val2;
			int jumps = 0;
			int cont = nSize;
			while (ans&&cont)
			{
				int ans1 = ans >> 1;
				int ans2 = ans1 << 1;
				int resi = ans - ans2;
				jumps += resi;
				ans >>= 1;
				cont--;
			}
			return jumps;
		}

		void  HJPOEMFeatureExtractor::GetLBPTable(int neighbor, std::vector<int> &LBPTable, int &LBPnum)
		{
			LBPTable = std::vector<int>((int)pow(2.f, neighbor));
			int newMax = neighbor*(neighbor - 1) + 3;	//59
			LBPnum = newMax;
			int idx = 0;
			for (int i = 0; i < LBPTable.size(); i++)
			{
				int j = HJRotateLeft1(i, neighbor);
				int jumps = HJCalcJumps(i, j, neighbor);
				LBPTable[i] = jumps <= 2 ? idx++ : newMax - 1;
			}
		}

		void HJPOEMFeatureExtractor::HJHOG(const cv::Mat &src, cv::Mat &Mag, cv::Mat &Ang)
		{
			Mag.create(src.size(), CV_32FC1);
			Ang.create(src.size(), CV_32FC1);

			cv::Mat fsrc;
			src.convertTo(fsrc, CV_32FC1);
			//cv::imshow("src", fsrc/255);

			cv::Mat hx = (cv::Mat_< float>(1, 3) << -1, 0, 1);
			cv::Mat hy = (cv::Mat_< float>(3, 1) << 1, 0, -1);

			//Acquire Gradient of X or Y axis.
			cv::Mat GradX;
			cv::Mat GradY;
			cv::filter2D(fsrc, GradX, fsrc.depth(), hx, cv::Point(1, 0), 0, cv::BORDER_CONSTANT);
			cv::filter2D(fsrc, GradY, fsrc.depth(), hy, cv::Point(0, 1), 0, cv::BORDER_CONSTANT);

			cv::Mat GradX2;
			cv::Mat GradY2;
			cv::pow(GradX, 2.0, GradX2);
			cv::pow(GradY, 2.0, GradY2);
			cv::sqrt(GradX2 + GradY2, Mag);

			//cv::imshow("mag", Mag / 255);
			//Calc the Orientations which can be accelerated
			/*
			for (int i = 0; i < Ang.rows; i++)
			{
				for (int j = 0; j < Ang.cols; j++)
				{
					float angles = atan2(((float*)GradY.data)[i][j], ((float*)GradX.data)[i, j));
					Ang.at<float>(i, j) = angles;
				}
			}*/

			for (int p = 0; p < Ang.rows* Ang.cols; ++p)
			{
				float angles = atan2(((float*)GradY.data)[p], ((float*)GradX.data)[p]);
				((float*)Ang.data)[p] = angles;
			}
			//Unsign the orientation
			cv::Mat Orient;
			cv::compare(Ang, cv::Mat::zeros(Ang.size(), Ang.type()), Orient, CV_CMP_LT);
			Orient /=  255;

			cv::Mat O32;
			Orient.convertTo(O32, CV_32FC1);
			//unsigned angles中心对称
			Ang = Ang + O32*CV_PI;

			//cv::imshow("ang", (Ang+CV_PI)/(2*CV_PI));

			//cv::waitKey(1);

			return;
		}

		void  HJPOEMFeatureExtractor::HJDecomp(cv::Mat &Mag, cv::Mat &Ang, int m, std::vector<cv::Mat> &Ms)
		{
			if (Mag.rows != Ang.rows || Mag.cols != Ang.cols)
			{
				return;
			}
			//1.Pre calc the edge-range of angles
			float TolAng = CV_PI;
			float StepAng = TolAng / m;

			std::vector<float> b(m + 1);
			for (int i = 0; i < m + 1; i++)
			{
				b[i] = StepAng*i;
			}

			for (int i = 0; i < m; i++)
			{
				cv::Mat lowborder = cv::Mat::ones(Ang.size(), CV_32FC1)*b[i];
				cv::Mat upborder = cv::Mat::ones(Ang.size(), CV_32FC1)*b[i + 1];
				// 		Mat Mask1;
				// 		Mat Mask2;
				cv::Mat Mask;
				// 		compare( Ang , lowborder , Mask1 , CV_CMP_GE );
				// 		compare( Ang , upborder , Mask2 , CV_CMP_LE );
				// 		bitwise_and( Mask1 , Mask2 , Mask );
				cv::inRange(Ang, lowborder, upborder, Mask);
				Mag.copyTo(Ms[i], Mask);
			}
		}

		void HJPOEMFeatureExtractor::HJAccumCeil(cv::Mat &Im, int ceil_size, cv::Mat &Op)
		{
			//1.Get size of Op
			int Rows = Im.rows;
			int Cols = Im.cols;
			int Radius = cvFloor((float)ceil_size / 2) + 1;
			//2. Calc Mask or Kernels
			cv::Mat Mask = cv::Mat::ones(ceil_size, ceil_size, CV_32FC1);
			cv::Point anchor(Radius - 1, Radius - 1);
			cv::Mat AccIm;
			cv::filter2D(Im, AccIm, CV_32FC1, Mask, anchor, 0, cv::BORDER_CONSTANT);
			//3. Chop the ROI
			cv::Mat tmp = AccIm(cv::Range(Radius - 1, Rows - Radius + 1), cv::Range(Radius - 1, Cols - Radius + 1));
			Op = tmp;
		}

		double HJPOEMFeatureExtractor::vecMin(double *src, int Len)
		{
			double temp;
			temp = src[0];
			for (int i = 1; i < Len; i++)
			{
				temp = (temp > src[i]) ? src[i] : temp;
			}
			return temp;
		}

		double HJPOEMFeatureExtractor::vecMax(double *src, int Len)
		{
			double temp;
			temp = src[0];
			for (int i = 1; i < Len; i++)
			{
				temp = (temp > src[i]) ? temp : src[i];
			}
			return temp;
		}

		cv::Size HJPOEMFeatureExtractor::iFastPreLBP(cv::Mat &ipl_src, int neighbors, int radius, int mode)
		{
			double *spoints1 = new double[neighbors];	//2  //eight sample number
			double *spoints2 = new double[neighbors];
			if (mode == 1)
			{
				//  		double spoints11[] = { -1 , -1 , -1 , 0 , 0 , 1 , 1 , 1 };
				//  		double spoints22[] = { -1 , 0 , 1 , -1 , 1, -1 , 0 , 1 };
				spoints1[0] = -1;		spoints1[1] = -1;		spoints1[2] = -1;		spoints1[3] = 0;
				spoints1[4] = 0;		spoints1[5] = 1;		spoints1[6] = 1;		spoints1[7] = 1;
				spoints2[0] = -1;	spoints2[1] = 0;		spoints2[2] = 1;	spoints2[3] = -1;
				spoints2[4] = 1;		spoints2[5] = -1;		spoints2[6] = 0;		spoints2[7] = 1;

			}
			else if (mode == 2)
			{
				//Angle step
				double a = 2 * CV_PI / neighbors;
				for (int i = 1; i < neighbors + 1; i++)
				{
					spoints1[i - 1] = -radius*sin((i - 1)*a);
					spoints2[i - 1] = radius*cos((i - 1)*a);
				}
			}
			else
				exit(-1);

			//Determine the dimensions of the input image.
			int ysize = ipl_src.rows;
			int xsize = ipl_src.cols;

			double miny = vecMin(spoints1, neighbors);
			double maxy = vecMax(spoints1, neighbors);
			double minx = vecMin(spoints2, neighbors);
			double maxx = vecMax(spoints2, neighbors);

			//Block size, each LBP code is computed within a block of size bsizey*bsizex
			int bsizey = cvCeil(MAX(maxy, 0)) - cvFloor(MIN(miny, 0)) + 1;
			int bsizex = cvCeil(MAX(maxx, 0)) - cvFloor(MIN(minx, 0)) + 1;

			//Calculate dx and dy;
			int dx = xsize - bsizex;
			int dy = ysize - bsizey;

			delete[] spoints1;
			delete[] spoints2;
			return cv::Size(dx + 1, dy + 1);
		}
		/*
		void HJPOEMFeatureExtractor::matCMPGE(const cv::Mat &matl, const cv::Mat &matr, cv::Mat &mat_dst)//dst为CV_8UC1
		{
			if ((matl.rows != matr.rows) || (matl.type()!= matr.type()))
			{
				exit(-1);
			}
			
			for (int x = 0; x < matl.cols; x++)
			{
				for (int y = 0; y < matr.rows; y++)
				{
					float leftval = ((float*)matl.data)[x + y*matl.cols];
					float rightval = ((float*)matr.data)[x + y*matr.cols];
					mat_dst.data[x + y*mat_dst.cols] = (((float*)matl.data)[x + y*matl.cols] >= ((float*)matr.data)[x + y*matr.cols]) ? 1 : 0;
				}
			}
			for (int p = 0; p < matl.cols*matl.rows;++p)
			{
				mat_dst.data[p] = (((float*)matl.data)[p] >= ((float*)matr.data)[p]) ? 1 : 0;
			}
		}*/

		void HJPOEMFeatureExtractor::FastPreLBP(cv::Mat &ipl_src, cv::Mat &ipl_dst, int mode, int neighbors, int radius)
		{
			cv::Mat mat_src= ipl_src.clone();	//32FC1
			// 	cvConvert( ipl_src , mat_src );
			// 	int neighbors = 8;
			//cv::imshow("m", mat_src);
			//cv::waitKey(1);
			// 	int radius = 2;
			double *spoints1 = new double[neighbors];	//2  //eight sample number
			double *spoints2 = new double[neighbors];
			if (mode == 1)
			{
				//  		double spoints11[] = { -1 , -1 , -1 , 0 , 0 , 1 , 1 , 1 }; dety 
				//  		double spoints22[] = { -1 , 0 , 1 , -1 , 1, -1 , 0 , 1 };  detx
				// 0 1 2         0 1 2
				//3     4 ---->7    3	-1 -1 -1 0 1 1 1 0
				//5 6 7			6  5 4	-1 0 1 1 1 0 -1 -1

				spoints1[0] = -1;	spoints1[1] = -1;		spoints1[2] = -1;		spoints1[3] = 0;	spoints1[4] = 1;		spoints1[5] = 1;		spoints1[6] = 1;		spoints1[7] = 0;
				spoints2[0] = -1;	spoints2[1] = 0;		spoints2[2] = 1;		spoints2[3] = 1;	spoints2[4] = 1;		spoints2[5] = 0;		spoints2[6] = -1;		spoints2[7] = -1;


			}
			else if (mode == 2)
			{
				//Angle step
				double a = 2 * CV_PI / neighbors;
				for (int i = 1; i < neighbors + 1; i++)
				{
					spoints1[i - 1] = -radius*sin((i - 1)*a);
					spoints2[i - 1] = radius*cos((i - 1)*a);
				}
			}
			else
				exit(-1);

			//Determine the dimensions of the input image.
			int ysize = mat_src.rows;
			int xsize = mat_src.cols;

			double miny = vecMin(spoints1, neighbors);
			double maxy = vecMax(spoints1, neighbors);
			double minx = vecMin(spoints2, neighbors);
			double maxx = vecMax(spoints2, neighbors);

			//Block size, each LBP code is computed within a block of size bsizey*bsizex
			int bsizey = cvCeil(MAX(maxy, 0)) - cvFloor(MIN(miny, 0)) + 1;
			int bsizex = cvCeil(MAX(maxx, 0)) - cvFloor(MIN(minx, 0)) + 1;

			//Coordinates of origin (0,0) in the block
			int origy = 1 - cvFloor(MIN(miny, 0));
			int origx = 1 - cvFloor(MIN(minx, 0));

			//Calculate dx and dy;
			int dx = xsize - bsizex;
			int dy = ysize - bsizey;

			// Fill the center pixel matrix C.
			//C = image(origy:origy+dy,origx:origx+dx);
			//CvMat *mat_C = cvCreateMat(dy + 1, dx + 1, CV_32FC1);			//3
			//CvMat mat_tmp;
			//cvGetSubRect(mat_src, &mat_tmp, cvRect(origx - 1, origy - 1, dx + 1, dy + 1));
			//cvCopy(&mat_tmp, mat_C);
			auto mat_C = mat_src(cv::Rect(origx - 1, origy - 1, dx + 1, dy + 1));

			int bins = pow(2.f, neighbors);
			// Initialize the result matrix with zeros.
			cv::Mat mat_result(dy + 1, dx + 1, CV_8UC1,cv::Scalar(0));	//4
			//Compute the LBP code image
			for (int i = 1; i < neighbors + 1; i++)
			{
				double y = spoints1[i - 1] + origy;
				double x = spoints2[i - 1] + origx;
				//Calculate floors, ceils and rounds for the x and y.
				int fy = cvFloor(y);	int cy = cvCeil(y);		int ry = cvRound(y);
				int fx = cvFloor(x);	int cx = cvCeil(x);		int rx = cvRound(x);
				//Check if interpolation is needed.
				cv::Mat mat_D(dy + 1, dx + 1, CV_8UC1), mat_N;		//5
				if ((abs(x - rx) < 1e-6) && (abs(y - ry) < 1e-6))
				{
					// N = image(ry:ry+dy,rx:rx+dx);
					//cvGetSubRect(mat_src, &mat_tmp, cvRect(rx - 1, ry - 1, dx + 1, dy + 1));
					//CvMat *mat_N = cvCreateMat(dy + 1, dx + 1, CV_32FC1);	//6
					//cvCopy(&mat_tmp, mat_N);
					mat_N = mat_src(cv::Rect(rx - 1, ry - 1, dx + 1, dy + 1));
					//D = N>=C;
					/*cvCmp( mat_N , mat_C , mat_D , CV_CMP_GE );*/
					//matCMPGE(mat_N, mat_C, mat_D);
					//mat_D = mat_N >= mat_C;
					//cvReleaseMat(&mat_N);										//~6
				}
				else
				{
					// Interpolation needed, use double type images 
					double ty = y - fy;
					double tx = x - fx;
					//Calculate the interpolation weights.
					double w1 = (1 - tx) * (1 - ty);
					double w2 = tx  * (1 - ty);
					double w3 = (1 - tx) *      ty;
					double w4 = tx  *      ty;
					//Compute interpolated pixel values
					/************************************************************************/
					/*     N = w1*d_image(fy:fy+dy,fx:fx+dx) + w2*d_image(fy:fy+dy,cx:cx+dx) + ...
					w3*d_image(cy:cy+dy,fx:fx+dx) + w4*d_image(cy:cy+dy,cx:cx+dx);
					D = N >= d_C;                                                                      */
					/************************************************************************/

					cv::Mat mat_w1 = mat_src(cv::Rect(fx - 1, fy - 1, dx + 1, dy + 1));
					cv::Mat mat_w2 = mat_src(cv::Rect(cx - 1, fy - 1, dx + 1, dy + 1));
					cv::Mat mat_w3 = mat_src(cv::Rect(fx - 1, cy - 1, dx + 1, dy + 1));
					cv::Mat mat_w4 = mat_src(cv::Rect(cx - 1, cy - 1, dx + 1, dy + 1));

					//CvMat *mat_N = cvCreateMat(dy + 1, dx + 1, CV_32FC1);	//11
					//CvMat *mat_w12 = cvCreateMat(dy + 1, dx + 1, CV_32FC1);	//12
					//CvMat *mat_w34 = cvCreateMat(dy + 1, dx + 1, CV_32FC1);	//13
					//cvAddWeighted(mat_w1, w1, mat_w2, w2, 0, mat_w12);
					//cv::Mat mat_w12 = mat_w1*w1 + mat_w2*w2;
					//cvAddWeighted(mat_w3, w3, mat_w4, w4, 0, mat_w34);
					//cv::Mat mat_w34 = mat_w3*w3 + mat_w4*w4;
					//cvAdd(mat_w12, mat_w34, mat_N);
					mat_N = mat_w1*w1 + mat_w2*w2 + mat_w3*w3 + mat_w4*w4;
													//D = N>=C;
													/*cvCmp( mat_N , mat_C , mat_D , CV_CMP_GE );*/
					//matCMPGE(mat_N, mat_C, mat_D);
					//mat_D = mat_N >= mat_C;//~11
				}
				mat_D = mat_N >= mat_C;

				//Update the result matrix.
				int v = pow(2.0f, (i - 1));
				//result = result + v*D;
				//Convert D 255 -- 1;
				mat_D /= 255;
				mat_result += v*mat_D;
				//mat_result = mat_result + v*((mat_N >= mat_C) / 255);			//~5
			}
			//cv::imshow("m2", mat_result);
			//cv::waitKey(1);
			ipl_dst = mat_result;
		}

		void HJPOEMFeatureExtractor::HJFastPreLBP(cv::Mat &Im, cv::Mat &Op, int mode, int neighbor, int radius)
		{
			cv::Mat ipl_src = Im.clone();
			cv::Size wh = iFastPreLBP(ipl_src, neighbor, radius, mode);

			cv::Mat ipl_LBP(wh, CV_8UC1);					//+
			FastPreLBP(ipl_src, ipl_LBP, mode, neighbor, radius);		//1.image to lbp255
			Op = ipl_LBP;
		}

		void HJPOEMFeatureExtractor::BlockIpl(cv::Mat &src, int nX, int nY, std::vector<cv::Rect> &mblock)
		{
			//Initial partition block of face
			int Xlength = src.cols, Ylength = src.rows;
			int Xresidual = Xlength%nX, Yresidual = Ylength%nY;
			int Xblock = (Xlength - Xresidual) / nX;
			int Yblock = (Ylength - Yresidual) / nY;

			for (int y = 0; y < nY; y++)
			{
				for (int x = 0; x < nX; x++)
				{
					mblock[x + y*nX] = cv::Rect(x*Xblock, y*Yblock, Xblock, Yblock);
				}
			}
			if (Xresidual != 0)
			{
				for (int y = 0; y < nY; y++)
				{
					mblock[(nX - 1) + y*nY] = cv::Rect((nX - 1)*Xblock, y*Yblock, Xblock + Xresidual, Yblock);
				}
			}
			if (Yresidual != 0)
			{
				for (int x = 0; x < nX; x++)
				{
					int xtemp = mblock[x + (nY - 1)*nY].width;
					mblock[x + (nY - 1)*nY] = cv::Rect(x*Xblock, (nY - 1)*Yblock, xtemp, Yblock + Yresidual);
				}
			}
		}

		cv::Mat HJPOEMFeatureExtractor::LBP2Hist(const cv::Mat &src, cv::Rect r,std::vector<int> &LBPtable, int &num)
		{
			//std::vector<float> vmhist(num);
			cv::Mat mhist = cv::Mat::zeros(num, 1, CV_32FC1);
			for (int i = 0; i < r.height; i++)
			{
				for (int j = 0; j <r.width; j++)
				{
					int fc = src.data[j + r.x + (i + r.y)*src.cols];
					int idx = LBPtable[fc];
					((float*)mhist.data)[idx] += 1;
				}
			}
			//mhist = cv::Mat(num, 1, CV_32FC1, vmhist.data()).clone();
			return mhist;
		}

		cv::Mat HJPOEMFeatureExtractor::LBP2Hist(const cv::Mat &src, cv::Rect r, cv::Mat &ulbp, std::vector<int> &LBPtable, int &num)
		{
			ulbp = cv::Mat::zeros(cv::Size(r.width,r.height), CV_8UC1);
			cv::Mat mhist = cv::Mat::zeros(num, 1, CV_32FC1);
			for (int i = 0; i <r.height; i++)
			{
				for (int j = 0; j <r.width; j++)
				{
					int fc = src.data[j + r.x + (i + r.y)*src.cols];
					int idx = LBPtable[fc];
					ulbp.data[j + i*ulbp.cols] = idx;
					((float*)mhist.data)[idx] += 1.0f;
				}
			}
			return mhist;
		}



		cv::Mat HJPOEMFeatureExtractor::HJFeatPOEM(const cv::Mat &src)
		{
			//1.Generate gradient images HOG
			cv::Mat Mag, Ang;
			HJHOG(src, Mag, Ang);

			//2.Decompose Gradient images to three(val_ang) different ones
			std::vector<cv::Mat> Ms(val_ang);
			HJDecomp(Mag, Ang, val_ang, Ms);

			//3.Accumulate and LBP Transform
			//Return Mat
			cv::Mat Otp = cv::Mat::zeros(val_ang*nW*nH*LBPnum, 1, CV_32FC1);

			int copyStartpos = 0;
			for (int v = 0; v < val_ang; v++)
			{
				//Accumulate
				cv::Mat AccMs;
				HJAccumCeil(Ms[v], val_cellsize, AccMs);
				//cv::imshow("fo" + std::to_string(v), AccMs/255);
				//cv::waitKey(1);
				//cv::Mat tmp;
				//cv::equalizeHist(Mag, tmp);
				//LBP Transformation
				cv::Mat LBPMs;//CV8UC1
				HJFastPreLBP(AccMs, LBPMs, 2, val_neighbor, cvFloor((float)val_blocksize / 2));
				//cv::imshow("LBPMs"+std::to_string(v), LBPMs);
				//cv::waitKey(1);
				//Block
				std::vector<cv::Rect> blocks(nW*nH);
				BlockIpl(LBPMs, nW, nH, blocks);
				//Extra Hist from each block

				cv::Mat ktmp(LBPMs.size(), CV_8UC3,cv::Scalar(255,255,255));

				for (int m = 0; m < nH*nW; ++m, copyStartpos += LBPnum)
				{
					//每格统计33*1的向量
					cv::Rect r = blocks[m];
					cv::Mat rtmp;
					cv::Mat mhist = LBP2Hist(LBPMs,r,rtmp,LBPTable, LBPnum);
					cv::cvtColor(rtmp, rtmp, cv::COLOR_GRAY2BGR);
					rtmp.copyTo(ktmp(r));
					//cv::Mat mhist = LBP2Hist(LBPMs, r, LBPTable, LBPnum);
					//std::cout << mhist.size().area()<<"|"<<mhist.elemSize()<<"|"<<LBPnum << std::endl;
					memcpy(Otp.data + copyStartpos * sizeof(float), mhist.data, mhist.size().area()*mhist.elemSize());
				}
				//cv::imshow("uo"+std::to_string(v), ktmp*7);
				//cv::waitKey(1);
			}
			return Otp;
		}
	}
}