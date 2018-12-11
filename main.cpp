#include <iostream>
#include <string>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <stdio.h>

#include "mesh.hpp"
#include "laplace.hpp"
#include "solver.hpp"
#include "mpitools.hpp"

int main(int argc, char** argv)
{
    if (argc < 5) 
    {
        std::cout << "Program needs M, N, maxiters, tol power, profile flag (optional)" << std::endl;
        exit(0);
    }
    int profile = 0;
    if (argc >= 6) profile = atoi(argv[5]);

    int M = atoi(argv[1]), N = atoi(argv[2]), maxiters = atoi(argv[3]), tol = atoi(argv[4]);

    // MPI/OpenMP
    MPITools mpitools;
    mpitools.init(&argc, &argv, M, N);

    if (mpitools.rank() == 0) 
    {
        std::cout << "Laplace Solver, pure mpi" << std::endl;
        std::cout << "M = " << M << " N = " << N << " maxiters = " << maxiters << " tol = " << pow(10.0, tol) << " profile = " << profile << std::endl;
    }
    //MPI_Abort(mpitools.comm(), 0);
    //exit(0);

    int X1, X2, Y1, Y2;
    X1 = 0; X2 = 2;
    Y1 = 0; Y2 = 1;

    // Initiation of Laplace Operator, RHS for it
    LaplaceOperator L(X1, X2, Y1, Y2, mpitools);
    MeshVec F(mpitools);
    L.rhs(F);

    // Initiation of MRM solver Ax=b, initial guess
    MRM solver(pow(10.0, tol), maxiters);
    MeshVec X(mpitools, 0.0);
    
    // Use solver
    double start, duration;
    int iters;
    start = mpitools.get_time();
    if (profile)
    {
        iters = solver.profile_solve(L, F, X);
    }
    else
    {
        iters = solver.solve(L, F, X);
    }
    duration = mpitools.get_time() - start;
    double maxduration = mpitools.sync_time(duration);
    if (mpitools.rank() == 0) 
    {
        std::cout << "Time of solver: " << duration << std::endl;
        std::cout << "Time of solver (max): " << maxduration << std::endl;
    }

    // Print errors
    double errL2, errC;
    errL2 =  L.errorL2(X);
    errC = L.errorC(X);
    if (mpitools.rank() == 0) 
    {
        std::cout << "Error (L2): " << errL2 << std::endl;
        std::cout << "Error (C): " << errC << std::endl;
        // Latex Output
        printf("Cores & Threads per core &   Mesh  & Time (sec) & Iterations & Error (L2) & Error (C) \n");
        printf("%d & %d & %d x %d & %f & %d & %f & %f \n", 
               mpitools.procs(), mpitools.threads(), M, N, duration, iters, errL2, errC);
    }

/*
    std::stringstream ss;
    ss << mpitools.rank();
    std::string strrank = ss.str();
    std::string fname;
    fname = "results_" + strrank + ".txt";
    std::ofstream results;
    results.open(fname.c_str());
    results << X;
    results.close();
*/

    mpitools.finalize();

    return 0;
}
