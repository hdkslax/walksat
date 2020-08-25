#include <iostream>
#include <map>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <string>


#ifndef RESOLUTION_H
#define RESOLUTION_H


using namespace std;



class Resolution {
private:
	vector<vector<int>> clauses;
	int nvbles;
	int nclauses;
	

public:
	Resolution();
	~Resolution();

	//getters
	vector<vector<int>> get_clauses();
	int get_nvbles();
	int get_nclauses();


	void readfile(const char *filepath);
	vector<string> split_input(string &clause_str);

	// map<int, bool> build_KB();
	map<int, bool> full_resolution();


	void print_clauses(vector<vector<int>> clauses);
	void print_KB(map<int, bool> KB);

	void print_result(map<int, bool> KB);

};

#endif