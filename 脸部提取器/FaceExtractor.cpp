#include"FaceExtractor.h"

using namespace Kagamine::FaceAnalyse;

FaceExtractor::FaceExtractor(std::wstring posemodelpath)
{
	m_FaceDetector = dlib::get_frontal_face_detector();
	dlib::deserialize(std::string(posemodelpath.begin(), posemodelpath.end())) >> m_ShapePredictor;
}

cv::Mat FaceExtractor::GetFace(const cv::Mat &src)
{
	//cv::imshow("", src);
	//cv::waitKey(10);
	auto cimg = dlib::cv_image<unsigned char>(src);
	if (src.empty())
	{
		std::wcout << L"空图" << std::endl;
		throw std::exception("错误！");
	}
	auto rec = m_FaceDetector(cimg);//人脸区域
	if (rec.size() == 0) {
		std::wcout << L"无脸" << std::endl;
		throw std::exception("错误！");
	}
	auto shape = m_ShapePredictor(cimg,rec[0]);//初始特征点
	dlib::matrix<unsigned char> RotatedImg;
	auto chip = get_face_chip_details(shape);
	dlib::extract_image_chip(cimg, chip, RotatedImg);
	cv::Mat tmpRotatedFace = dlib::toMat(RotatedImg);//校正后人脸图像
	shape = dlib::map_det_to_chip(shape, chip);//校正后特征点
	std::vector<cv::Point> cvshape,cvconvex;
	for (int i = 0; i < shape.num_parts();++i)
	{
		cvshape.push_back(cv::Point(shape.part(i).x(), shape.part(i).y()));
	}
	cv::convexHull(cvshape,cvconvex);//获取凸包RotatedFace);
	cv::Mat mask(tmpRotatedFace.size(), CV_8UC1, cv::Scalar(0));
	//std::cout << (mask.empty()?1:0) << std::endl;
	cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{cvconvex}, cv::Scalar(255));//制作遮罩图像
	cv::Mat face(tmpRotatedFace.size(), tmpRotatedFace.type(),cv::Scalar(145));
	tmpRotatedFace.copyTo(face, mask);

	///指定截取策略
	auto tmpRec = cv::boundingRect(cvconvex);
	auto len = tmpRec.width;
	//std::cout << cv::Rect(tmpRec.x, tmpRec.y, len, len) << std::endl;
	auto border = cv::Rect(tmpRec.x, tmpRec.y, len, len) & cv::Rect(0,0,200,200);//包围矩形

	face = face(border).clone();
	return face;
}
