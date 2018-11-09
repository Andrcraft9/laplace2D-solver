#include <iostream>
#include <string>
#include <cassert>
#include <cmath> 
#include "mesh.hpp"
#include "laplace.hpp"

#ifndef SOLVER_H
#define SOLVER_H

// Minimal Residual Method
class MRM
{
private:
    double eps;
    int maxIters;
    
    // No copy
    MRM(const MRM& m);
    MRM& operator=(const MRM& m);

public:
    MRM(double eps, int maxiters) : eps(eps), maxIters(maxiters) {}

    double solve(const LaplaceOperator& L, const MeshVec& RHS, MeshVec& X) const
    {
        int k = 0;
        double err = 1.0;
        int M = X.get_M(), N = X.get_N();
        MeshVec r(M, N);
        MeshVec Ar(M, N);

        L.matvec(X, r);
        r -= RHS;
        L.matvec(r, Ar);
        err = r.norm();

        while (err > eps && k <= maxIters)
        {    
            double tau = Ar.dot(r) / pow(Ar.norm(), 2);
            X.axpy(-tau, r);

            L.matvec(X, r);
            r -= RHS;
            L.matvec(r, Ar);
            err = r.norm();
            
            ++k;
        }

        if (k >= maxIters) std::cout << "Warning! Max Iterations in MRM solver!" << std::endl;
        
        return err;
    }
};

#endif