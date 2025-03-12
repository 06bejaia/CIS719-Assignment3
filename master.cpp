#include <mpi.h>

#include <iostream>

#include <string>
 
using namespace std;
 
#define MAXLEN 100
 
int main(int argc, char** argv) {

    MPI_Status st;

    int procNum, rank;
 
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &procNum);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
    if (rank == 0) { // Master process

        while (--procNum) { // Receive messages from all workers

            char buff[MAXLEN];

            MPI_Recv(buff, MAXLEN, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &st);

            int aux;

            MPI_Get_count(&st, MPI_CHAR, &aux);

            buff[aux] = '\0';

            cout << "Received: " << buff << endl;

        }

    }
 
    MPI_Finalize();

    return 0;

}
 
 
