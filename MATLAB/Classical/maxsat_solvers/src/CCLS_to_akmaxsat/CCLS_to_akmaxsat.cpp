#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <chrono> // Leo added for precision timing

using namespace std;

int main(int argc, char **argv)
{
	printf("c This is the CCLS_to_akmaxsat solver, Version: MAXSAT EVALUATION 2014 (2014.03.28)\n");
	printf("c Many thanks to the akmaxsat team!\n");

	if(argc!=3)
	{
		printf("c Usage: %s <instance> <first_lower_bound_threshold>\n", argv[0]);
		return -1;
	}

	int my_pid = getpid();
	int my_time = time(0);
	string my_pid_str;
	string my_time_str;
	string my_instance;
	string my_result_file;
	string my_command;
	stringstream my_sstream;
	
	my_sstream.clear();
	my_sstream.str("");
	my_sstream<<my_pid;
	my_sstream>>my_pid_str;
	my_sstream.clear();
	my_sstream.str("");
	my_sstream<<my_time;
	my_sstream>>my_time_str;
	
	my_instance = argv[1];
	my_instance = "\"" + my_instance + "\"";
	my_result_file = "ccls_res_" + my_pid_str + "_" + my_time_str;
	my_command = "./CCLS2014 " + my_instance + " 1 10 > ./" + my_result_file;
	
    auto time_start = std::chrono::high_resolution_clock::now(); // Leo's addition starts - timing starts
	cout<<"c "<<my_command<<endl;
	printf("c start CCLS\n");
	system(my_command.c_str());
	printf("c stop CCLS\n");
	
	string terminating_gap_str = argv[2];
	my_command = "./akmaxsat " + my_instance + " ./" + my_result_file + " " + terminating_gap_str;
	cout<<"c "<<my_command<<endl;
	printf("c start akmaxsat\n");
	system(my_command.c_str());
	printf("c stop akmaxsat\n");
    // Leo's addition starts
    auto time_fin = std::chrono::high_resolution_clock::now();
    std::cout << "c ** Total CCLS2ak time = "
    << std::chrono::duration_cast<std::chrono::milliseconds>(time_fin-time_start).count()
    << " milliseconds\n";
    my_command = "rm -f " + my_result_file;
    system(my_command.c_str());
    // Leo's addition ends
	
	return 0;
}
