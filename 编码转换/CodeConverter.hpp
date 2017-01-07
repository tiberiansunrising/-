#pragma once
#include<codecvt>
namespace Kagamine
{
	namespace Encode
	{
		class CodeConverterForChinese :public std::codecvt_byname<wchar_t, char, std::mbstate_t>
		{
		public:
			CodeConverterForChinese() :codecvt_byname("chs") {}
		};
	}
}
