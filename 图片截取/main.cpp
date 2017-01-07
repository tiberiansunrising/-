#include<iostream>
#include <regex>
#include <map>
#include<thread>
#include<unordered_map>
#include <vector>
#include <string>
#include<chrono>
#include<array>
#include <windows.h>
#include <locale>
#include<numeric>
#include<algorithm>
#include<opencv2/opencv.hpp>
#include"CodeConverter.hpp"
#include<filesystem>
#include<ctime>
#include <iomanip> 
#include<sstream>
#pragma warning (disable:4996) 
int wmain(int argc, wchar_t argv[])
{
	std::wstring_convert<Kagamine::Encode::CodeConverterForChinese> conv;
	cv::VideoCapture cp(1);
	bool run = true;
	cv::namedWindow("Show");
	auto storagePath = std::experimental::filesystem::path(L"Modules//Screensaves");
	if (!std::experimental::filesystem::exists(storagePath))std::experimental::filesystem::create_directories(storagePath);
	std::wstringstream ss;
	ss.imbue(std::locale());
	std::wcout.imbue(std::locale());
	while (run)
	{
		cv::Mat pic;
		cp >> pic;
		cv::imshow("Show",pic);
		switch (cv::waitKey(25))
		{
		case 27:
		case 'q':
		case 'Q':
			run = false;
			break;
		case 'w':
		case 'W':
		{
			ss.str(L"");
			std::time_t tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
			ss << storagePath << "\\" << std::put_time(localtime(&tt), L"%Y-%m-%d#%H.%M.%S.bmp");
			auto name = ss.str();
			std::wcout << name << std::endl;
			cv::imwrite(conv.to_bytes(name), pic);
			break;
		}
		default:
			break;
		}
	}
	cv::destroyAllWindows();
	system("pause");
}