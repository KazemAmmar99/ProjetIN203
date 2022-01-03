#include "mpi.h"
#include <stdio.h>
int main( int argc, char * argv[] )
{
    MPI_Init(&nargs,&argv);
    MPI_Comm globComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &globComm);
    int nbp;
    MPI_Comm_size(globComm, &nbp);
    int rank;
    MPI_Comm_rank(globComm, &rank);
    MPI_Status Stat;
    MPI_Datatype stat_point;
    MPI_Type_contiguous(3,MPI_INT,&stat_point);
    
    MPI_Type_commit(&stat_point);
    MPI_Request send_request_quit, rcv_request_quit,send_request1,send_request2;
    

    class {
        Sensibilite sensibilite = Sensibilite::Sensible;
        int temps_incubation = 0;
        int temps_symptomatique = 0;
        int temps_contagieux = 0;
    } m_grippe;
 
    MPI_Finalize();
    return errs;
}