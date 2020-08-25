

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <random>
#include <ctime>
#include <vector>
#include <map>
#include "Walksat.h"

// #ifndef WALKSAT_H
// #define WALKSAT_H

using namespace std;

Walksat::Walksat(){}
Walksat::~Walksat(){}


// getters
vector<vector<int>> Walksat::get_clauses() {return this->clauses;}
int Walksat::get_num_vbles() {return this->num_vbles;}
int Walksat::get_num_clauses() {return this->num_clauses;}



void Walksat::readfile(const char *filepath){
	string info;
	string clause_str;

	ifstream sat_file;
	sat_file.open(filepath);



	getline(sat_file, clause_str); // the comment line
	getline(sat_file, info); // the minisat info line

	this->num_vbles = stoi(split_input(info)[2]);
	this->num_clauses = stoi(split_input(info)[3]);

	// cout << num_vbles << " " << num_clauses << endl;

	vector<vector<int>> clauses(this->num_clauses, vector<int>());

	for (int i=0; i<num_clauses; i++){
		getline(sat_file, clause_str);
		vector<string> clause = split_input(clause_str);

		for(int j=0; j<clause.size(); j++){
			// cout << clause[j] << " ";
			// if (stoi(clause[j]) == 0) continue; 
			clauses[i].push_back(stoi(clause[j]));
			// cout << stoi(clause[j]) << endl;
		}
	}

	this->clauses = clauses;
	// print_clauses(this->clauses);

	sat_file.close();
	
}


// split the input string by space ' '
vector<string> Walksat::split_input(string& clause_str){
	vector<string> clause;

	int vble_length = 0;
	int vble_start = 0;
	for(int i=0; i<clause_str.length(); i++){

		if(clause_str[i] == ' '){
			string vble = clause_str.substr(vble_start, vble_length);
			clause.push_back(vble);
			vble_start = i+1;
			vble_length = 0;
		}else{
			vble_length++;
		}

	}
	string vble = clause_str.substr(vble_start, vble_length);
	clause.push_back(vble);
	
	return clause;
}


// check if the clause is satisfiable
bool Walksat::is_sat(vector<int>& clause, map<int, bool>& model){
	for(int i=0; i<clause.size(); i++){
		if (clause[i] > 0 && (model.at(abs(clause[i]))) || clause[i]<0 && (!model.at(abs(clause[i])))){
			return true;
		}
	}
	return false;

	// for(std::vector<int>::iterator iter = clause.begin(); iter != clause.end(); ++iter){
 //    	if (((*iter > 0) && (model.at(abs(*iter)))) || ((*iter < 0) && (!model.at(abs(*iter))))) {
 //      		return true;
 //    	}
 //  	}
 //  	return false;
}



// get the unsatisfied clause
vector<vector<int>> Walksat::get_unsat_clauses(map<int, bool>& model){
	vector<vector<int>> unsat_clauses;
	for (int i=0; i<get_num_clauses(); i++){
		if(!is_sat(this->clauses[i], model)){
			unsat_clauses.push_back(this->clauses[i]);
		}
	}

	return unsat_clauses;
}


// get the number of satisfied clauses when flip a variable vbles
int Walksat::get_num_sat_clauses(int vble, map<int, bool> model){
	model[vble] = !model[vble];
	int sat_clauses_count = get_num_clauses() - get_unsat_clauses(model).size();
	model[vble] = !model[vble];
	return sat_clauses_count;
}




map<int, bool> Walksat::walksat_alg(float p, int max_flips){
	
	map<int, bool> model;

	// initialize model
	for(int i=1; i<get_num_vbles() + 1; i++){
			model[i] = false;
	}

	vector<vector<int>> unsat_clauses = get_unsat_clauses(model);

	// cout << "size = " << unsat_clauses.size() <<endl;

	srand((unsigned)time(NULL));
	int nfilps = 0;
	while(nfilps < max_flips){
		// cout << "nfilps = " << nfilps << endl;
		if (unsat_clauses.size() == 0) {
			return model;
		}

		// randomly choose an unsat clause
		int random_index = rand()%unsat_clauses.size();
		vector<int> clause = unsat_clauses[random_index];

		// randomly choose a probility
		float rand_p = (rand() % 1000) / 1000;
		// cout << "unsat_clauses.size() = " << unsat_clauses.size() << endl;
		random_index = rand()%(clause.size());


		// cout << clause[random_index] << endl;
		if (model.count(abs(clause[random_index]) == 0)) {
			continue;
		}

		// cout << "random_index = " << random_index << endl;
		// cout << "clause[random_index] = " << clause[random_index] << endl;
		int vble_to_flip = abs(clause[random_index]);
		// cout << "vble_to_flip = " << vble_to_flip << endl;

		if (rand_p < p) {
			model[vble_to_flip] = !model[vble_to_flip];
		} else {
			int sat_count = get_num_sat_clauses(vble_to_flip, model); 
			for (map<int, bool>::iterator iter = model.begin(); iter != model.end(); ++iter){
				if (get_num_sat_clauses(iter->first, model) > sat_count){
					model[iter->first] = !model[iter->first];
				}
			}
		}

		unsat_clauses = get_unsat_clauses(model);
		// print_clauses(unsat_clauses);
		// print_model(model);

		nfilps++;
		
	}

	model.clear();

	// model[0] = false;
	return model;

}



//TODO: just for debug, remember to delete this
void Walksat::print_clauses(vector<vector<int>> clauses){
	// for(int i=0; i<clauses.size(); i++){
	// 	cout << "Clause:";
	// 	vector<int> clause = clauses[i];
	// 	for (int j=0; j<clause.size(); j++){
	// 		cout << clause[j] << " ";
	// 	}
	// 	cout << endl;
	// }

	for(vector<vector<int>>::iterator iter1 = clauses.begin(); iter1 != clauses.end(); ++iter1){
    	for (vector<int>::iterator iter2 = iter1->begin(); iter2 != iter1->end(); ++iter2){
      		cout << *iter2 << " ";
    	}
    	cout << endl;
  	}

}

void Walksat::print_model(map<int, bool> model){
	map<int, bool>::iterator iter = model.begin();
	while(iter != model.end()) {
		char prefix;
		if (iter->second) prefix = ' ';
		else prefix = '-';
		cout << prefix << iter->first << ' ';
		iter++;
	}
	cout << endl;
}

// #endif