#include <iostream>
#include <map>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <string>
#include "Resolution.h"
#include <ctime>

using namespace std;

int main(int argc, char const *argv[])
{
	// char filepath[] = "../test_files/queens2.txt";
	// char filepath[] = "../test_files/queens3.txt";
	// char filepath[] = "../test_files/3000-CNF.txt";
	char filepath[] = "../test_files/5000-CNF.txt";

	Resolution *resolution = new Resolution();

	resolution->readfile(filepath);

	clock_t start, end;
	start = clock();
	map<int, bool> KB = resolution->full_resolution();
	end = clock();

	float duration = (float) (end - start)/CLOCKS_PER_SEC;
	
	cout << duration << endl;
	// resolution->print_KB(KB);
	resolution->print_result(KB);

	delete resolution;

	return 0;
}