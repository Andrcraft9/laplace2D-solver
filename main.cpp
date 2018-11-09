#include <iostream>
#include <string>
#include <cassert>
#include "mesh.hpp"
#include "laplace.hpp"
#include "solver.hpp"

int main()
{
    std::cout << "Laplace Solver" << std::endl;

    int M, N, X1, X2, Y1, Y2;
    M = 20; N = 20;
    X1 = 0; X2 = 2;
    Y1 = 0; Y2 = 1;

    MeshVec v(M, N);
    for(int i = 0; i < M; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            v(i, j) = 1.0;
        }
    }
    //std::cout << "v: " << std::endl << v;

    LaplaceOperator L(X1, X2, Y1, Y2, M, N);
    MeshVec Av(M, N); 
    MeshVec F(M, N);
    L.matvec(v, Av); 
    L.rhs(F);   
    //std::cout << "Av: " << std::endl << Av;
    //std::cout << "F: " << std::endl << F;

    int iters;
    std::cin >> iters;
    MRM solver(0.001, iters);
    MeshVec X(M, N);
    
    double err;
    err = solver.solve(L, F, X);
    std::cout << "Resudial: " << err << std::endl;
    std::cout << "Error: " << L.error(X) << std::endl;

    
    return 0;
}