#include"FileManager.h"
#include<iostream>
using namespace Kagamine::File;

FileList FileManager::FileFilter(std::wstring folderPath, std::wstring extensions)
{
	///获取扩展名列表,构造正则表达式字符串
	FileList result;
	std::wstring regexstr = L"([^\\.]*)|(.*\\.(";
	if (extensions != L"*")
	{
		auto extensionList = SplitWString(extensions);
		for (auto &ext : extensionList)
		{
			regexstr += ext + L"|";
		}
		regexstr.pop_back();
		regexstr += L"))$";
	}
	else
	{
		regexstr += L"[[:alpha:]]*))$";
	}
	//std::wcout.imbue(std::locale("chs"));
	//std::wcout << L"正则分割"<< regexstr <<std::endl;
	std::wregex extensionFilter(regexstr);
	std::wcmatch regex_result;

	WIN32_FIND_DATAW fd;
	auto hFindFile = FindFirstFileW((folderPath + L"\\*.*").c_str(), &fd);
	if (hFindFile == INVALID_HANDLE_VALUE)return result;

	do
	{
		
		auto name = fd.cFileName;
		if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
		{
			result[L"文件夹"].push_back(name);
		}
		else
		{
			if (std::regex_match(name, regex_result, extensionFilter))
			{
				if (regex_result[1].matched)
				{
					result[L"未知"].push_back(name);
				}
				else
				{
					auto extensionName = regex_result[3].str();
					std::transform(extensionName.begin(), extensionName.end(), extensionName.begin(), ::tolower);
					result[extensionName].push_back(name);
				}
			}
		}

	} while (FindNextFileW(hFindFile, &fd));
	return result;
}
std::vector<std::wstring> FileManager::SplitWString(std::wstring input)
{
	std::vector<std::wstring> result;
	std::wsregex_token_iterator searcher(input.begin(), input.end(), m_SplitPattern, -1);
	while (searcher!= m_SplitEnd)
	{
		result.push_back((*searcher++).str());
	}
	return result;
}

void FileManager::CopyFileToFolder(std::wstring fileFullPath, std::wstring newFullPath, bool FailIfExists)
{
	CopyFileW(fileFullPath.c_str(), newFullPath.c_str(), FailIfExists);
}