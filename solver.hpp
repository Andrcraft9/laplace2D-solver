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

    int solve(const LaplaceOperator& L, const MeshVec& RHS, MeshVec& X) const;
    int profile_solve(const LaplaceOperator& L, const MeshVec& RHS, MeshVec& X) const;
};

#endif
