#include<iostream>
#include<codecvt>
#include <regex>
#include <map>
#include <vector>
#include <string>
#include <windows.h>
#include <locale>
#include<FileManager.h>
#include"FaceExtractor.h"
#include"CodeConverter.hpp"

int wmain(int argc, wchar_t*argv[])
{
	try
	{
		std::wstring_convert<Kagamine::Encode::CodeConverterForChinese> conv;
		std::wcout.imbue(std::locale("chs"));
		std::wstring rootFolder = (argc == 1 ? L"." : argv[1]);
		rootFolder += L"\\";
		std::wcout << L"开始提取" << rootFolder << std::endl;

		Kagamine::File::FileManager fm;
		std::vector<std::wstring> exList = { L"jpg",L"png", L"bmp", L"jpeg" };
		std::wstring exs = L"";
		for (auto &ex : exList)
		{
			exs += ex + L";";
		}
		auto filelist = fm.FileFilter(rootFolder, exs);

		std::vector<std::wstring> imageList;
		std::wregex r(L"\\.(jpg|png|bmp|jpeg)$", std::regex::icase);
		for (auto &ex : exList)
		{
			for (auto &name : filelist[ex])
			{
				imageList.push_back(name);
			}
		}
		std::wcout << L"扫描到" << imageList.size() << L"个图像文件" << std::endl;

		std::wstring storageFolder = rootFolder + L"\\Faces\\";
		CreateDirectoryW(storageFolder.c_str(), NULL);
		std::wcout << L"保存到" << storageFolder << std::endl;

		Kagamine::FaceAnalyse::FaceExtractor fe;
		for (auto &bmpname : imageList)
		{
			try
			{
				auto fullStorageName = storageFolder + std::regex_replace(bmpname, r, L"") + L".bmp";
				auto show = cv::imread(conv.to_bytes(rootFolder + bmpname), cv::IMREAD_GRAYSCALE);
				auto res = fe.GetFace(show);
				cv::resize(res, res, cv::Size(128, 128));
				std::wcout << fullStorageName << std::endl;
				cv::imwrite(conv.to_bytes(fullStorageName), res);
			}
			catch (std::exception &e)
			{
				std::wcout << bmpname << L":错误:" << e.what() << std::endl;
			}
		}
		std::wcout << L"截取完成" << std::endl;
	}
	catch (std::exception e)
	{
		std::wcout << e.what() << std::endl;
	}
}