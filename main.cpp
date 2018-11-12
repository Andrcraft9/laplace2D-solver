#include <iostream>
#include <string>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>

#include "mesh.hpp"
#include "laplace.hpp"
#include "solver.hpp"
#include "mpitools.hpp"

int main(int argc, char** argv)
{
    if (argc < 4) 
    {
        std::cout << "Program needs M, N, maxiters" << std::endl;
        exit(0);
    }
    int M = atoi(argv[1]), N = atoi(argv[2]), maxiters = atoi(argv[3]);

    // MPI/OpenMP
    MPITools mpitools;
    mpitools.init(&argc, &argv, M, N);

    if (mpitools.rank() == 0) 
    {
        std::cout << "Laplace Solver, parallel mpi" << std::endl;
        std::cout << "M = " << M << " N = " << N << " maxiters = " << maxiters << std::endl;
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
    MRM solver(1.0e-6, maxiters);
    MeshVec X(mpitools, 0.0);
    
    // Use solver
    double res;
    double start, duration;
    start = mpitools.start_timer();
    res = solver.solve(L, F, X);
    duration = mpitools.end_timer(start);
    if (mpitools.rank() == 0) 
    {
        std::cout << "Time: " << duration << std::endl;
    }

    // Print errors
    double errL2, errC;
    errL2 =  L.errorL2(X);
    errC = L.errorC(X);
    if (mpitools.rank() == 0) 
    {
        std::cout << "Resudial: " << res << std::endl;
        std::cout << "Error (L2): " << errL2 << std::endl;
        std::cout << "Error (C): " << errC << std::endl;
    }

    std::stringstream ss;
    ss << mpitools.rank();
    std::string strrank = ss.str();
    std::string fname;
    fname = "results_" + strrank + ".txt";
    std::ofstream results;
    results.open(fname.c_str());
    results << X;
    results.close();

    mpitools.finalize();

    return 0;
}