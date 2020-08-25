#include <iostream>
#include <map>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <string>
#include "Resolution.h"

using namespace std;


Resolution::Resolution(){}
Resolution::~Resolution(){}


// getters
vector<vector<int>> Resolution::get_clauses(){return this->clauses;}
int Resolution::get_nvbles() {return this->nvbles;}
int Resolution::get_nclauses() {return this->nclauses;}



void Resolution::readfile(const char *filepath){
	string info;
	string clause_str;

	ifstream sat_file;
	sat_file.open(filepath);


	getline(sat_file, clause_str); // the comment line

	getline(sat_file, info); // the minisat info line

	
	this->nvbles = stoi(split_input(info)[2]);
	
	this->nclauses = stoi(split_input(info)[3]);

	vector<vector<int>> clauses(this->nclauses, vector<int>());

	for (int i=0; i<nclauses; i++){
		getline(sat_file, clause_str);
		vector<string> clause = split_input(clause_str);

		for(int j=0; j<clause.size()-1; j++){
			// cout << clause[j] << " ";
			clauses[i].push_back(stoi(clause[j]));
			// cout << stoi(clause[j]) << endl;
		}
	}

	this->clauses = clauses;

	sat_file.close();

}

vector<string> Resolution::split_input(string &clause_str){
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


map<int, bool> Resolution::full_resolution(){
	// cout << "clauses = " << endl; 
	// print_clauses(this->clauses);

	map<int, bool> KB;
	vector<int> clause = this->clauses[0];
	for (int i = 0; i<clause.size(); i++){
		if (clause[i] < 0){
			KB[abs(clause[i])] = false;
		}else if (clause[i] > 0) {
			KB[abs(clause[i])] = true;
		}
	}
	
	for(int i=1; i<this->clauses.size(); i++){
		clause = this->clauses[i];
		for (int j=0; j<clause.size(); j++){
			
			if (clause[j] == 0){
				continue;
			} 
			if (KB.count(abs(clause[j]))==0){
				// cout << "clause[j] = " << clause[j] << endl;
				if (clause[j] < 0){
					
					KB[abs(clause[j])] = false;

				}else {

					KB[abs(clause[j])] = true;
					
				}
			}else{
				
				if ((clause[j]>0 && KB[abs(clause[j])] == false) || (clause[j]<0 && KB[abs(clause[j])] == true)){
					KB.erase(abs(clause[j]));
				}
			}
			// cout << "KB = ";
			// print_KB(KB);
			
		}
	}

	// cout << "KB = " << endl;
	// print_KB(KB);

	return KB;
}



void Resolution::print_clauses(vector<vector<int>> clauses){
	for(std::vector<std::vector<int>>::iterator clause = clauses.begin(); clause != clauses.end(); ++clause){
    	for (std::vector<int>::iterator literal = clause->begin(); literal != clause->end(); ++literal){
      		std::cout << *literal << " ";
    	}
    	std::cout << std::endl;
  	}
}

void Resolution::print_KB(map<int, bool> KB){
	map<int, bool>::iterator iter = KB.begin();
	while(iter != KB.end()) {
		char prefix;
		if (iter->second) prefix = ' ';
		else prefix = '-';
		cout << prefix << iter->first << ' ';
		iter++;
	}
	cout << endl;
}

void Resolution::print_result(map<int, bool> KB){
	if (KB.size() == 0){
		cout << "UNSATISIABLE" << endl;
	}else{
		// print_KB(KB);
		cout << "SATISIABLE" << endl;
	}
}