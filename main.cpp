#include <iostream>
#include <string>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include "mesh.hpp"
#include "laplace.hpp"
#include "solver.hpp"
#include <stdio.h>

int main(int argc, char** argv)
{
    std::cout << "Laplace Solver" << std::endl;
    if (argc < 4) 
    {
        std::cout << "Program needs M, N, maxiters" << std::endl;
        exit(0);
    }
    int M = atoi(argv[1]), N = atoi(argv[2]), maxiters = atoi(argv[3]);
    std::cout << "M = " << M << " N = " << N << std::endl;

    // Initiation of Laplace Operator, RHS for it
    int X1, X2, Y1, Y2;
    X1 = 0; X2 = 2; Y1 = 0; Y2 = 1;
    LaplaceOperator L(X1, X2, Y1, Y2, M, N);
    MeshVec F(M, N);
    L.rhs(F);

    // Initiation of MRM solver Ax=b, initial guess
    MRM solver(1.0e-6, maxiters);
    MeshVec X(M, N, 0.0);
    
    // Use solver
    std::clock_t start, end;
    int iters;
    start = std::clock();
    iters = solver.solve(L, F, X);
    end = std::clock();
    std::cout << "Time: " << (end - start) / (double) CLOCKS_PER_SEC << std::endl;

    // Print errors
    double errL2, errC;
    errL2 =  L.errorL2(X);
    errC =  L.errorC(X);
    std::cout << "Error (L2): " << L.errorL2(X) << std::endl;
    std::cout << "Error (C): " << L.errorC(X) << std::endl;
    
    // Latex Output
    printf("Cores & Threads per core &   Mesh  & Time (sec) & Iterations & Error (L2) & Error (C) \n");
    printf("%d & %d & %d x %d & %f & %d & %f & %f \n", 
            1, 1, M, N, (end - start) / (double) CLOCKS_PER_SEC , iters, errL2, errC);

    std::ofstream results;
    results.open("results.txt");
    results << X;
    results.close();

    return 0;
}