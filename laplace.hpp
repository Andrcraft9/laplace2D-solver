#include <iostream>
#include <string>
#include <cassert>
#include <cmath> 
#include "mesh.hpp"

#ifndef LAPLACE_H
#define LAPLACE_H

/*
Laplace Operator: -L u = F

Area: 2d rectangular [X1, X2] x [Y1, Y2]
          (T)
 Y2 xxxxxxxxxxxxxxx
    x             x
    x             x
    x             x
(L) x             x (R)
    x             x
    x             x
 Y1 xxxxxxxxxxxxxxx
   X1     (B)     X2
Mesh: i = 0, M; j = 0, N
Uniform mesh!

Boundary conditions:
L: Neumann
B: Neumann
R: Dirichlet
T: Dirichlet

Discrete Laplace Operator: A u = B,
A  : M*N x M*N (because values on T and R borders are known from bc)
u,B: M*N
*/

class LaplaceOperator
{
private:
    double X1, X2;
    double Y1, Y2;
    double hx, hy;
    int M, N;

    // No copy
    LaplaceOperator(const LaplaceOperator& l);
    LaplaceOperator& operator=(const LaplaceOperator& l);

    double point_x(int i) const
    {
        //assert(i >= 0 && i <= M);
        return X1 + i*hx;
    }

    double point_y(int j) const
    {
        //assert(j >= 0 && j <= N);
        return Y1 + j*hy;
    }
    
    // Right hand side
    double func_F(int i, int j) const
    {
        double x, y;
        x = point_x(i);
        y = point_y(j);
        return cos(M_PI*x*y) * (pow(M_PI*x,2) + pow(M_PI*y, 2));
    }

    // Left Boundary Condition
    double func_LBC(int j) const
    {
        return 0;
    }

    // Right Boundary Condition
    double func_RBC(int j) const
    {
        double y;
        y = point_y(j);
        return 1.0 + cos(X2*M_PI*y);
    }

    // Top Boundary Condition
    double func_TBC(int i) const
    {
        double x;
        x = point_x(i);
        return 1.0 + cos(Y2*M_PI*x);
    }

    // Bottom Boundary Condition
    double func_BBC(int i) const
    {
        return 0;
    }
    double func_solution(int i, int j) const
    {
        double x, y;
        x = point_x(i);
        y = point_y(j);
        return 1.0 + cos(M_PI*x*y);
    }

public:
    LaplaceOperator(double X1, double X2, double Y1, double Y2, int M, int N)
        : X1(X1), X2(X2), Y1(Y1), Y2(Y2), M(M), N(N)
    {
        hx = (X2 - X1) / M;
        hy = (Y2 - Y1) / N;

        assert(func_BBC(0) == func_LBC(0));
    }

    int matvec(const MeshVec &v, MeshVec &Av) const;

    int rhs(MeshVec &f) const;

    double error(const MeshVec& sol) const
    {
        double err = 0.0;
        for(int i = 1; i <= M-1; ++i)
        {
            for(int j = 1; j <= N-1; ++j)
            {
                err = pow((func_solution(i, j) - sol(i, j)), 2);
            }
        }

        return sqrt(err);
    }
};

#endif