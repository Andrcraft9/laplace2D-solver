#include "solver.hpp"

int MRM::solve(const LaplaceOperator& L, const MeshVec& RHS, MeshVec& X) const
{
    int k = 0;
    double err = 1.0;
    int M = X.get_M(), N = X.get_N();

    MeshVec r(M, N);
    MeshVec Ar(M, N);
    L.matvec(X, r); r -= RHS;
    L.matvec(r, Ar);
    err = L.norm_mesh(r);

    while (err > eps && k <= maxIters)
    {    
        double tau = L.dot_mesh(Ar, r) / pow(L.norm_mesh(Ar), 2);
        X.axpy(-tau, r);

        L.matvec(X, r); r -= RHS;
        L.matvec(r, Ar);
        err = L.norm_mesh(r);
        
        ++k;
    }

    std::cout << "MRM: iters: " << k << " resudial: " << err << std::endl;
    if (k >= maxIters) std::cout << "MRM: Warning! Max Iterations in MRM solver!" << std::endl;
    
    return 0;
}