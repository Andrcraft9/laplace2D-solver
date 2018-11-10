#include <iostream>
#include <string>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include "mesh.hpp"
#include "laplace.hpp"
#include "solver.hpp"

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

    int X1, X2, Y1, Y2;
    X1 = 0; X2 = 2;
    Y1 = 0; Y2 = 1;

    // Initiation of Laplace Operator, RHS for it
    LaplaceOperator L(X1, X2, Y1, Y2, M, N);
    MeshVec F(M, N);
    L.rhs(F);

    // Initiation of MRM solver Ax=b, initial guess
    MRM solver(1.0e-6, maxiters);
    MeshVec X(M, N, 0.0);
    
    // Use solver
    double res;
    std::clock_t start, end;
    start = std::clock();
    res = solver.solve(L, F, X);
    end = std::clock();
    std::cout << "Time: " << (end - start) / (double) CLOCKS_PER_SEC << std::endl;

    // Print errors
    std::cout << "Resudial: " << res << std::endl;
    std::cout << "Error (L2): " << L.errorL2(X) << std::endl;
    std::cout << "Error (C): " << L.errorC(X) << std::endl;

    return 0;
}