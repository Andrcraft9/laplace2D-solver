#include "laplace.hpp"

int LaplaceOperator::matvec(const MeshVec &v, MeshVec &Av) const
{
    assert(M == v.mpitools().M()  && N == v.mpitools().N()); 
    assert(M == Av.mpitools().M() && N == Av.mpitools().N()); 

    int i, j;

    // Inner area
    for(i = inner_x1; i <= inner_x2; ++i)
    {
        for(j = inner_y1; j <= inner_y2; ++j)
        {
            Av(i, j) = -1.0/pow(hx,2)*(v(i+1,j) - 2*v(i,j) + v(i-1,j)) 
                       -1.0/pow(hy,2)*(v(i,j+1) - 2*v(i,j) + v(i,j-1));
        }
    }

    // Edge point: (X1, Y1)
    if (mtls.LB() && mtls.BB())
    {
        i = 0; j = 0;
        Av(i, j) = -2.0/pow(hx,2)*(v(1,0) - v(0,0)) 
                   -2.0/pow(hy,2)*(v(0,1) - v(0,0));
    }

    // Bottom boundary
    if (mtls.BB())
    {
        j = 0;
        for(i = inner_x1; i <= inner_x2; ++i)
        {
            Av(i, j) = -2.0/pow(hy,2)*(v(i,1) - v(i,0)) 
                       -1.0/pow(hx,2)*(v(i+1,0) - 2*v(i,0) + v(i-1,0));
        }
        if (mtls.RB())
        {
            i = M-1;
            Av(i, j) = -2.0/pow(hy,2)*(v(M-1,1) - v(M-1,0)) 
                       -1.0/pow(hx,2)*( -2*v(M-1,0) + v(M-2,0));
        }
    }

    // Left boundary
    if (mtls.LB())
    {
        i = 0;
        for(j = inner_y1; j <= inner_y2; ++j)
        {
            Av(i, j) = -2.0/pow(hx,2)*(v(1,j) - v(0,j))
                       -1.0/pow(hy,2)*(v(0,j+1) - 2*v(0,j) + v(0,j-1));
        }
        if (mtls.TB())
        {
            j = N-1;
            Av(i, j) = -2.0/pow(hx,2)*(v(1,N-1) - v(0,N-1))
                       -1.0/pow(hy,2)*( -2*v(0,N-1) + v(0,N-2));
        }
    }

    // Right pre-boundary
    if (mtls.RB())
    {
        i = M-1;
        for(j = inner_y1; j <= inner_y2; ++j)
        {
            Av(i, j) = -1.0/pow(hx,2)*( -2*v(M-1,j) + v(M-2,j)) 
                       -1.0/pow(hy,2)*(v(M-1,j+1)  - 2*v(M-1,j) + v(M-1,j-1));
        }
    }

    // Top pre-boundary
    if (mtls.TB())
    {
        j = N-1;
        for(i = inner_x1; i <= inner_x2; ++i)
        {
            Av(i, j) = -1.0/pow(hx,2)*(v(i+1,N-1)  - 2*v(i,N-1) + v(i-1,N-1)) 
                       -1.0/pow(hy,2)*( -2*v(i,N-1) + v(i,N-2));
        }
    }

    // Edge point: (X2, Y2)
    if (mtls.TB() && mtls.RB())
    {
        i = M-1; 
        j = N-1;
        Av(i, j) = -1.0/pow(hx,2)*( -2*v(M-1,N-1) + v(M-2,N-1)) 
                   -1.0/pow(hy,2)*( -2*v(M-1,N-1) + v(M-1,N-2));
    }

    return 0;
}

int LaplaceOperator::rhs(MeshVec &f) const
{
    assert(M == f.mpitools().M()  && N == f.mpitools().N()); 

    int i, j;

    // Inner area
    for(i = inner_x1; i <= inner_x2; ++i)
    {
        for(j = inner_y1; j <= inner_y2; ++j)
        {
            f(i, j) = func_F(i, j);
        }
    }

    // Edge point: (X1, Y1)
    if (mtls.LB() && mtls.BB())
    {
        i = 0; j = 0;
        f(i, j) = func_F(0, 0) - (2.0/hx + 2.0/hy)*func_BBC(0);
    }

    // Bottom boundary
    if (mtls.BB())
    {
        j = 0;
        for(i = inner_x1; i <= inner_x2; ++i)
        {
            f(i, j) = func_F(i, 0) - 2.0/hy * func_BBC(i);
        }
        if (mtls.RB())
        {
            i = M-1;
            f(i, j) = func_F(M-1, 0) - 2.0/hy * func_BBC(M-1) + 1.0/pow(hx,2)*func_RBC(0);
        }        
    }

    // Left boundary
    if (mtls.LB())
    {
        i = 0;
        for(j = inner_y1; j <= inner_y2; ++j)
        {
            f(i ,j) = func_F(0, j) - 2.0/hx * func_LBC(j);
        }
        if (mtls.TB())
        {
            j = N-1;
            f(i ,j) = func_F(0, N-1) - 2.0/hx * func_LBC(N-1) + 1.0/pow(hy,2)*func_TBC(0);
        }
    }

    // Right pre-boundary
    if (mtls.RB())
    {
        i = M-1;
        for(j = inner_y1; j <= inner_y2; ++j)
        {
            f(i, j) = func_F(M-1, j) + 1.0/pow(hx, 2) * func_RBC(j);
        }
    }

    // Top pre-boundary
    if (mtls.TB())
    {
        j = N-1;
        for(i = inner_x1; i <= inner_x2; ++i)
        {
            f(i, j) = func_F(i, N-1) + 1.0/pow(hy, 2) * func_TBC(i);
        }
    }

    // Edge point: (X2, Y2)
    if (mtls.TB() && mtls.RB())
    {
        i = M-1; 
        j = N-1;
        f(i, j) = func_F(M-1, N-1) + 1.0/pow(hx, 2) * func_RBC(j) + 1.0/pow(hy, 2) * func_TBC(i);
    }

    return 0;
}

double LaplaceOperator::dot_mesh(const MeshVec& v1, const MeshVec& v2) const
{
    assert(M == v1.mpitools().M() && N == v1.mpitools().N()); 
    assert(M == v2.mpitools().M() && N == v2.mpitools().N()); 
    
    double dot = 0.0, dot_out = 0.0;
    int i, j;

    // Edge point
    /*
    i = 0; j = 0;
    dot = dot + 0.25*hx*hy*v1(i, j)*v2(i, j);

    // Bottom boundary
    j = 0;
    for(i = 1; i <= M-1; ++i)
        dot = dot + 0.5*hx*hy*v1(i, j)*v2(i, j);

    // Left boundary
    i = 0;
    for(j = 1; j <= N-1; ++j)
        dot = dot + 0.5*hx*hy*v1(i, j)*v2(i, j);
    */

    // Inner area
    for(i = inner_x1; i <= mtls.locx2(); ++i)
        for(j = inner_y1; j <= mtls.locy2(); ++j)
            dot = dot + hx*hy*v1(i ,j)*v2(i, j);

    MPI_Allreduce(&dot, &dot_out, 1, MPI_DOUBLE, MPI_SUM, mtls.comm());

    return dot_out;
}

double LaplaceOperator::errorL2(const MeshVec& sol) const
{
    double err = 0.0, err_out = 0.0;
    for(int i = mtls.locx1(); i <= mtls.locx2(); ++i)
        for(int j = mtls.locy1(); j <= mtls.locy2(); ++j)
            err = err +  pow((func_solution(i, j) - sol(i, j)), 2);

    MPI_Allreduce(&err, &err_out, 1, MPI_DOUBLE, MPI_SUM, mtls.comm());
    return sqrt(err_out);
}

double LaplaceOperator::errorC(const MeshVec& sol) const
{
    double err = 0.0, err_out = 0.0;
    for(int i = mtls.locx1(); i <= mtls.locx2(); ++i)
        for(int j = mtls.locy1(); j <= mtls.locy2(); ++j)
            if (fabs(func_solution(i, j) - sol(i, j)) > err) err = fabs(func_solution(i, j) - sol(i, j));           

    MPI_Allreduce(&err, &err_out, 1, MPI_DOUBLE, MPI_MAX, mtls.comm());

    return err_out;
}