// This is the main DLL file.

#include "stdafx.h"

#include "LibFm.h"

using namespace System;
using namespace std;

namespace LibFm
{
	// this is managed C++ interface for running libFm through dll function call
	// Calling LibFm via this dll is however not recommended since the memory is partially manged by the caller and the performance is 
	// significatly lower than calling the executable libFm via Process.Start
	// call this function in C# with the following syntax
	// 	[DllImport("LibFm.dll", EntryPoint = "RunLibFm", CharSet=CharSet.Ansi)]
	//  public static extern int RunLibFm(int argc, StringBuilder argv);
	extern "C" { __declspec(dllexport) int RunLibFm(int argc, char* argv); }

	int RunLibFm(int argc, char* argv)
	{
		string argv_str(argv);
		
		char* space = new char[2];
		space[0] = ' ';
		space[1] = '\0';

		string delim(space);
		std::vector<std::string> result = tokenize(argv_str, space);

		return libfm_main(argc, result);
	}

}