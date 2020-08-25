
#include <iostream>
#include <string>
#include <map>
#include <vector>


#ifndef WALKSAT_H
#define WALKSAT_H


using namespace std;

class Walksat {
private:
	vector<vector<int>> clauses;
	int num_vbles;
	int num_clauses;


public:
	Walksat();
	~Walksat();

	// getters
	vector<vector<int>> get_clauses();
	int get_num_vbles();
	int get_num_clauses();

	void readfile(const char *filepath);

	// split the input string by space ' '
	vector<string> split_input(string& clause_str);

	// check if the clause is satisfiable
	bool is_sat(vector<int>& clause, map<int, bool>& model);
	
	// get the number of satisfied clauses when flip a variable vbles
	int get_num_sat_clauses(int vble, map<int, bool> model);


	// get the unsatisfied clause
	vector<vector<int>> get_unsat_clauses(map<int, bool>& model);

	map<int, bool> walksat_alg(float p, int max_flips);

	// void print_info(parse_time, eliminated_clauses, restarts, conflicts, conflits_time, decidions, decision_time, propagations, propagation_time, conflict_literals, memory_use, cpu_time);

	// TODO: test the program, remember to delete
	void print_clauses(vector<vector<int>> clauses);
	void print_model(map<int, bool> model);

};

#endif