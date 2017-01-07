#include<iostream>
#include <regex>
#include <map>
#include<thread>
#include<unordered_map>
#include <vector>
#include <string>
#include<chrono>
#include<codecvt>
#include<array>
#include <windows.h>
#include <locale>
#include<numeric>
#include<algorithm>
#include<FileManager.h>
#include<CodeConverter.hpp>
#include<ImageFeatureExtractors.h>
#include"FaceExtractor.h"
#include<opencv2/opencv.hpp>

int wmain(int argc, wchar_t argv[])
{
	try
	{
		std::wcout.imbue(std::locale("chs"));
		std::wstring_convert<Kagamine::Encode::CodeConverterForChinese> conv;
		auto rawimg = cv::imread(conv.to_bytes(L"00005_001.bmp"), cv::IMREAD_GRAYSCALE);
		Kagamine::cvex::HJGaborFeatureExtractor gfr;
		Kagamine::FaceAnalyse::FaceExtractor fe;
		gfr.Extract(fe.GetFace(rawimg));
	}
	catch (cv::Exception &e)
	{
		std::wcout << e.what() << std::endl;
	}
	system("pause");
}