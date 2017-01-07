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

		std::wcout << L"��ʼ�鲢" << rootFolder << std::endl;

		Kagamine::File::FileManager fm;
		auto folderslist = fm.FileFilter(rootFolder, L"jpg;bmp;jpeg;png");
		std::wcout << L"ɨ�赽" << folderslist[L"�ļ���"].size() << L"�����ļ���" << std::endl;

		std::wstring storageFolder = rootFolder + L"\\AllFiles";
		CreateDirectoryW(storageFolder.c_str(), NULL);
		std::wcout << L"�鲢��" << storageFolder << std::endl;

		for (auto &folder : folderslist[L"�ļ���"])
		{
			if (folder == L"AllFiles")continue;
			auto filelist = fm.FileFilter(rootFolder + folder, L"jpg;bmp");
			for (auto &unit : filelist)
			{
				if (unit.first == L"�ļ���" || unit.first == L"δ֪")continue;
				for (auto &name : unit.second)
				{
					fm.CopyFileToFolder(rootFolder + folder + L"\\" + name, storageFolder + L"\\" + folder + L"_" + name, false);
				}
			}
		}
		std::wcout << L"�鲢���" << std::endl;
	}
	catch (std::exception e)
	{
		std::wcout << e.what() << std::endl;
	}
}