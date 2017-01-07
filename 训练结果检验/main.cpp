#include<iostream>
#include<fstream>
#include<codecvt>
#include<algorithm>
#include <regex>
#include <unordered_map>
#include <vector>
#include <string>
#include <windows.h>
#include <locale>
#include<FileManager.h>
#include<CodeConverter.hpp>
#include<ImageFeatureExtractors.h>
#include<opencv2/opencv.hpp>
#include"FaceExtractor.h"

double HJCalcDisC(float *MatX, float *MatY, int arrayLen)
{

	double sum = 0;
	double modX = 0, modY = 0;
	for (int i = 0; i < arrayLen; ++i)

	{
		float ftemp1 = MatX[i];
		float ftemp2 = MatY[i];
		sum += ftemp1 * ftemp2;
		modX += ftemp1 * ftemp1;
		modY += ftemp2 * ftemp2;
	}
	//double CosDis = 0;
	if (modX > 0 && modY > 0)
	{
		return sum / (sqrt(modX) * sqrt(modY));
	}
	else
	{
		return 0;
	}
}

double HJCalcDisK(float *MatX, float* MatY, int arrayLen)
{
	double KDis = 0;
	for (int i = 0; i < arrayLen; i++)
	{
		double dtemp1 = MatX[i];
		double dtemp2 = MatY[i];

		double dtemp3 = dtemp1 + dtemp2;
		if (!dtemp3)continue;
		KDis += pow((dtemp1 - dtemp2), 2) / dtemp3;
	}
	//Normalization
	//KDis = 25000 - KDis;
	//KDis /= 9000;
	return KDis;
	//return 0.5 - 0.5*tanh((KDis - 20000) / 2000);
}

double HJCalcDisE(float *MatX, float* MatY, int arrayLen)
{
	double n = 0, d = 0, dis = 0;
	for (int i = 0; i < arrayLen; i++)
	{
		dis += MatX[i] * MatY[i];
	}
	return sqrt(dis) / arrayLen;
}

int wmain(int argc, wchar_t*argv[])
{
	try
	{
		std::wstring_convert<Kagamine::Encode::CodeConverterForChinese> conv;
		std::wcout.imbue(std::locale("chs"));
		Kagamine::cvex::HJPOEMFeatureExtractor fr;
		Kagamine::File::FileManager fm;

		std::wstring RootFolder = argc == 1 ? L"." : argv[1];
		std::wcout << L"扫描"<< RootFolder <<"中" << std::endl;

		auto bmpNameList = fm.FileFilter(RootFolder, L"bmp;jpg");
		std::wcout << L"共" << bmpNameList[L"bmp"].size()+ bmpNameList[L"jpg"].size() << L"个文件" << std::endl;

		std::wstring storageFolder = RootFolder + L"\\SubFiles\\";
		CreateDirectoryW(storageFolder.c_str(), NULL);
		///解析人员分类
		Kagamine::FaceAnalyse::FaceExtractor fe;
		cv::Mat basicFeature= fr.Extract(fe.GetFace(cv::imread(conv.to_bytes(RootFolder + L"\\" + L"2016-12-30#12.34.33.bmp"), cv::IMREAD_GRAYSCALE)));
		//std::wregex namePattern(L"^(10037|00005)_(.{3})\\.bmp$", std::regex::icase);
		//std::wregex namePattern(L"^(.*)\\.jpg$", std::regex::icase);
		std::wregex namePattern(L"^([[:alnum:]]{4}-[[:alnum:]]{2}-[[:alnum:]]{2}#[[:alnum:]]{2}).*.bmp$", std::regex::icase);
		std::wsmatch matchResult;
		std::unordered_map<std::wstring, std::vector<double>> dismap;
		for (auto &name : bmpNameList[L"bmp"])
		{
			try
			{
				if (!std::regex_match(name, matchResult, namePattern))throw std::exception("不匹配的模式");
				auto  src = cv::imread(conv.to_bytes(RootFolder + L"\\" + name), cv::IMREAD_GRAYSCALE);
				auto face = fe.GetFace(src);
				//auto aface = fr.PreProcessing(face);
				//cv::imwrite(conv.to_bytes(storageFolder + name), aface);
				auto ff = fr.Extract(face);
				auto dis = HJCalcDisC((float*)basicFeature.data, (float*)ff.data, basicFeature.size().area());
				std::wcout << L"\t" << name<<L"相似度："<< dis << std::endl;
				dismap[matchResult[1]].push_back(dis);
			}
			catch (std::regex_error e)
			{
				std::wcout << L"Regex::" << e.what() << std::endl;
			}
			catch (std::exception e)
			{
				auto estr = conv.from_bytes(e.what());
				if (estr != L"不匹配的模式")
				{
					std::wcout << L"STL::" << estr << std::endl;
				}
			}
		}
		std::wcout << L"模拟完成" << std::endl;
		for (auto &person : dismap)
		{
			std::wcout << person.first << L":\t" << *std::max_element(person.second.begin(), person.second.end())<<L",\t"<< *std::min_element(person.second.begin(), person.second.end()) << std::endl;
		}
		std::wcout << L"计算完成" << std::endl;
	}
	catch (std::exception e)
	{
		std::wcout << e.what() << std::endl;
	}
}