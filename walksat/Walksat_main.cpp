#include <iostream>
#include <map>
#include <vector>
#include "Walksat.h"
#include <ctime>

// #ifndef WALKSAT_H
// #define WALKSAT_H

using namespace std;

int main(int argc, char const *argv[])
{
	// char filepath[] = "../test_files/queens2.txt";
	// char filepath[] = "../test_files/queens3.txt";
	// char filepath[] = "../test_files/3000-CNF.txt";
	char filepath[] = "../test_files/5000-CNF.txt";

	Walksat *walksat = new Walksat();
	walksat->readfile(filepath);


	clock_t start, end;
	start = clock();
	map<int, bool> model = walksat->walksat_alg(0.5, 10000);
	end = clock();

	float duration = (float) (end - start)/CLOCKS_PER_SEC;
	cout << "duration = " << duration << endl;

	// cout << "============================[ Problem Statistics ]=============================" << endl;
	// cout << "|                                                                             |" << endl;
	// cout << "|  Number of variables:             " << get_num_vbles() <<"                                         |" << endl;
	// cout << "|  Number of clauses:              " << get_num_clauses() << "                                         |" << endl;
	// cout << "|  Parse time:                   " << parse_time << " s                                       |" << endl;
	// cout << "|  Eliminated clauses:           " << eliminated_clauses << " Mb                                      |" << endl;
	// cout << "|  Simplification time:          " << 0.00 << " s                                       |" << endl;
	// cout << "|                                                                             |" << endl;
	// cout << "===============================================================================" << endl;
	// cout << "Solved by simplification" << endl;
	// cout << "restarts              : " << restarts << endl;
	// cout << "conflicts             : " <<  << "              (" << conflits_time << " /sec)" << endl;
	// cout << "decisions             : " << decisions << "              (-nan % random) (" << decision_time << " /sec)" << endl;


	if (model.size() == 0) {
		// walksat->print_model(model);
		cout << "UNSAT" <<endl;
	} else {
		// walksat->print_model(model);
		cout << "SATIS" <<endl;
	}

	delete walksat;

	return 0;
}

// #endif