#include <iostream>
#include <string>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <stdio.h>

#include "mpitools.hpp"
#include "mesh.hpp"
#include "laplace.hpp"
#include "solver.hpp"

int main(int argc, char** argv)
{
    if (argc < 5) 
    {
        std::cout << "Program needs M, N, maxiters, tol power" << std::endl;
        exit(0);
    }
    int M = atoi(argv[1]), N = atoi(argv[2]), maxiters = atoi(argv[3]), tol = atoi(argv[4]);

    // MPI
    MPITools mpitools;
    mpitools.init(&argc, &argv, M, N);

    if (mpitools.rank() == 0) 
    {
        std::cout << "Laplace Solver, MPI/CUDA" << std::endl;
        std::cout << "M = " << M << " N = " << N << " maxiters = " << maxiters << " tol = " << pow(10.0, tol) << std::endl;
        std::cout << "Version with profiling" << std::endl;

        #ifdef _CUDA_AWARE_MPI_SYNC_
            std::cout << "using _CUDA_AWARE_MPI_SYNC_" << std::endl;
        #endif
        #ifdef _CUDA_KERNEL_MATVEC_
            std::cout << "using _CUDA_KERNEL_MATVEC_" << std::endl;
        #endif
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
    F.load_gpu();
    F.sync();

    // Initiation of MRM solver Ax=b, initial guess
    MRM solver(pow(10.0, tol), maxiters);
    MeshVec X(mpitools, 0.0);
    X.load_gpu();
    
    // Use solver
    double start, duration;
    int iters;
    start = mpitools.get_time();
    iters = solver.solve_profile(L, F, X);
    duration = mpitools.get_time() - start;
    if (mpitools.rank() == 0) 
    {
        std::cout << "Time: " << duration << std::endl;
    }

    // Unload solution from GPU
    X.unload_gpu();

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

/* Output
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
