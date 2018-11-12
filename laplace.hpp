#include <iostream>
#include <string>
#include <cassert>
#include <cmath> 
#include "mesh.hpp"
#include "mpitools.hpp"

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

MeshVec - mesh functions, always without T and R borders,
size: M*N (because values on T and R borders are known from bc)

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
    int inner_x1;
    int inner_x2;
    int inner_y1;
    int inner_y2;
    MPITools mtls;

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
        return cos(M_PI*x*y) * (pow(M_PI*x,2) + pow(M_PI*y,2));
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
    LaplaceOperator(double X1, double X2, double Y1, double Y2, MPITools mpitools)
        : X1(X1), X2(X2), Y1(Y1), Y2(Y2), mtls(mpitools)
    {
        assert(mtls.initialized());

        M = mtls.M();
        N = mtls.N();
        hx = (X2 - X1) / M;
        hy = (Y2 - Y1) / N;

        inner_x1 = std::max(1, mtls.locx1());
        inner_x2 = std::min(M-2, mtls.locx2());
        inner_y1 = std::max(1, mtls.locy1());
        inner_y2 = std::min(N-2, mtls.locy2());

        assert(func_BBC(0) == func_LBC(0));
        assert(func_TBC(M) == func_RBC(N));
    }

    MPITools mpitools() const { return mtls; }

    int matvec(const MeshVec &v, MeshVec &Av) const;

    int rhs(MeshVec &f) const;

    double dot_mesh(const MeshVec& v1, const MeshVec& v2) const;

    double norm_mesh(const MeshVec& v) const
    {
        return sqrt(dot_mesh(v, v));
    }

    double errorL2(const MeshVec& sol) const
    {
        double err = 0.0, err_out = 0.0;
        for(int i = mtls.locx1(); i <= mtls.locx2(); ++i)
            for(int j = mtls.locy1(); j <= mtls.locy2(); ++j)
                err = err +  pow((func_solution(i, j) - sol(i, j)), 2);

        MPI_Allreduce(&err, &err_out, 1, MPI_DOUBLE, MPI_SUM, mtls.comm());
        return sqrt(err_out);
    }

    double errorC(const MeshVec& sol) const
    {
        double err = 0.0, err_out = 0.0;
        for(int i = mtls.locx1(); i <= mtls.locx2(); ++i)
            for(int j = mtls.locy1(); j <= mtls.locy2(); ++j)
                if (fabs(func_solution(i, j) - sol(i, j)) > err) err = fabs(func_solution(i, j) - sol(i, j));           

        MPI_Allreduce(&err, &err_out, 1, MPI_DOUBLE, MPI_MAX, mtls.comm());

        return err_out;
    }

};

#endif