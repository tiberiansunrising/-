#include<iostream>
#include <regex>
#include <map>
#include <vector>
#include <string>
#include <windows.h>
#include <locale>
#include<FileManager.h>

int wmain(int argc, wchar_t*argv[])
{
	try
	{
		std::wcout.imbue(std::locale("chs"));
		std::wstring rootFolder = (argc == 1 ? L"." : argv[1]);
		rootFolder += L"\\";

		std::wcout << L"开始归并" << rootFolder << std::endl;

		Kagamine::File::FileManager fm;
		auto folderslist = fm.FileFilter(rootFolder, L"jpg;bmp;jpeg;png");
		std::wcout << L"扫描到" << folderslist[L"文件夹"].size() << L"个子文件夹" << std::endl;

		std::wstring storageFolder = rootFolder + L"\\AllFiles";
		CreateDirectoryW(storageFolder.c_str(), NULL);
		std::wcout << L"归并到" << storageFolder << std::endl;

		for (auto &folder : folderslist[L"文件夹"])
		{
			if (folder == L"AllFiles")continue;
			auto filelist = fm.FileFilter(rootFolder + folder, L"jpg;bmp");
			for (auto &unit : filelist)
			{
				if (unit.first == L"文件夹" || unit.first == L"未知")continue;
				for (auto &name : unit.second)
				{
					fm.CopyFileToFolder(rootFolder + folder + L"\\" + name, storageFolder + L"\\" + folder + L"_" + name, false);
				}
			}
		}
		std::wcout << L"归并完成" << std::endl;
	}
	catch (std::exception e)
	{
		std::wcout << e.what() << std::endl;
	}
}