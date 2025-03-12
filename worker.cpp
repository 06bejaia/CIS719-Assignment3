#include <mpi.h>
#include <iostream>
#include <string>
#include <vector>
#include <random>
 
using namespace std;
 
#define MAXLEN 100
 
int main(int argc, char** argv) {
    int rank;
 
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
    // Use C++ vector of strings
    vector<string> greetings = { "Hello", "Hi", "Awaiting your command" };
 
    // Better random number generation
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(0, greetings.size() - 1);
 
    int grID = dist(gen);  // Pick a random greeting
    string message = "Node " + to_string(rank) + " says: " + greetings[grID];
 
    // Send the message to process 0
    MPI_Send(message.c_str(), message.size() + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
 
    MPI_Finalize();
    return 0;
}