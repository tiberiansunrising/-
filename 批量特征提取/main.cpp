#include<iostream>
#include<fstream>
#include<codecvt>
#include <regex>
#include <map>
#include <vector>
#include <string>
#include <windows.h>
#include <locale>
#include<FileManager.h>
#include<CodeConverter.hpp>
#include<ImageFeatureExtractors.h>
int wmain(int argc, wchar_t*argv[])
{
	try
	{
		std::fstream fs;
		Kagamine::cvex::HJPOEMFeatureExtractor fr;
		Kagamine::File::FileManager fm;

		std::wstring RootFolder = argc == 1 ? L"." : argv[1];
		std::wregex r(L"\\.bmp$");
		std::wstring_convert<Kagamine::Encode::CodeConverterForChinese> conv;
		std::wcout.imbue(std::locale("chs"));
		//std::cout.imbue(std::locale("chs"));
		std::wcout << L"扫描中" << std::endl;

		auto bmpNameList = fm.FileFilter(RootFolder, L"bmp");
		std::wcout << L"共" << bmpNameList[L"bmp"].size() << L"个文件" << std::endl;
		std::wstring StorageFolder = RootFolder + L"\\Features\\";
		CreateDirectoryW(StorageFolder.c_str(), NULL);
		std::wcout << L"保存到" << StorageFolder << std::endl;

		for (auto &name : bmpNameList[L"bmp"])
		{
			std::wstring datname = std::regex_replace(name, r, L".dat");
			//std::cout << conv.to_bytes(RootFolder + L"\\" + name) << std::endl;
			auto src = cv::imread(conv.to_bytes(RootFolder + L"\\" + name), cv::IMREAD_GRAYSCALE);
			//std::cout << src.size() << std::endl;
			auto ff = fr.ExtractWithOutDR(src);
			std::wcout << StorageFolder + datname << std::endl;
			fs.open(StorageFolder + datname, std::ios::out | std::ios::binary | std::ios::trunc);
			fs.write((char*)ff.data, ff.size().area()*ff.elemSize());
			fs.close();
		}
		std::wcout << "提取完成" << std::endl;
	}
	catch (std::exception e)
	{
		std::wcout << e.what() << std::endl;
	}
}