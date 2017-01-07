#pragma once
#include<opencv2\opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/threads.h>
#include <dlib/opencv.h>

namespace Kagamine
{
	namespace FaceAnalyse
	{
		class FaceExtractor
		{
		public:
			FaceExtractor(std::wstring posemodelpath = L"Modules/shape_predictor_68_face_landmarks.dat");
			cv::Mat GetFace(const cv::Mat &src);
		private:
			dlib::frontal_face_detector m_FaceDetector;
			dlib::shape_predictor m_ShapePredictor;
		};
	}
}

