#pragma once
#include <regex>
#include<unordered_map>
#include <map>
#include <vector>
#include <string>
#include <windows.h>
#include <locale>
namespace Kagamine
{
	namespace File
	{
		typedef std::unordered_map<std::wstring, std::vector<std::wstring>> FileList;

		class FileManager
		{
		public:
			FileList FileFilter(std::wstring folderPath = L".\\", std::wstring extensions = L"*");
			void CopyFileToFolder(std::wstring fileFullPath, std::wstring newFullPath, bool FailIfExists);
		private:
			const std::wregex m_SplitPattern{ L";" };
			const std::wsregex_token_iterator m_SplitEnd;

			std::vector<std::wstring> SplitWString(std::wstring input);
		};
	}
}
