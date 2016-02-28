// This is the main DLL file.

#include "stdafx.h"

#include "LibFm.h"

using namespace System;
using namespace System::Collections::Generic;

int main()
{
	std::cout << "Salam";
	
	LibFm::LibFmManager ^lf = gcnew LibFm::LibFmManager();

	List<String^> ^test = gcnew List<String^>();
	lf->Setup(test);
	lf->CreateTrainSet(test, 1, 5, 10, 10);

}