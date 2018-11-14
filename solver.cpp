#include "solver.hpp"

int MRM::solve(const LaplaceOperator& L, const MeshVec& RHS, MeshVec& X) const
{
    int k = 0;
    double err = 1.0;
    MPITools mpitools(X.mpitools());

    MeshVec r(mpitools);
    MeshVec Ar(mpitools);
    
    L.matvec(X, r); r -= RHS; r.sync();
    L.matvec(r, Ar);
    err = L.norm_mesh(r);

    while (err > eps && k <= maxIters)
    {    
        double tau = L.dot_mesh(Ar, r) / pow(L.norm_mesh(Ar), 2);
        X.axpy(-tau, r); X.sync();

        L.matvec(X, r); r -= RHS; r.sync();
        L.matvec(r, Ar);
        err = L.norm_mesh(r);
        
        ++k;
    }

    if (L.mpitools().rank() == 0) std::cout << "MRM: iters: " << k << " resudial: " << err << std::endl;
    if (k >= maxIters && L.mpitools().rank() == 0) 
        std::cout << "MRM: Warning! Max Iterations in MRM solver!" << std::endl;
    
    return k;
}
