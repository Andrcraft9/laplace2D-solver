#include "laplace.hpp"

int LaplaceOperator::matvec(const MeshVec &v, MeshVec &Av) const
{
    assert(M == v.get_M()  && N == v.get_N()); 
    assert(M == Av.get_M() && N == Av.get_N()); 

    int i, j;

    // Edge point: (X1, Y1)
    i = 0; j = 0;
    Av(i, j) = -2.0/pow(hx,2)*(v(1,0) - v(0,0)) 
                -2.0/pow(hy,2)*(v(0,1) - v(0,0));

    // Bottom boundary
    j = 0;
    for(i = 1; i <= M-1; ++i)
    {
        Av(i, j) = -2.0/pow(hy,2)*(v(i,1) - v(i,0)) 
                    -1.0/pow(hx,2)*(v(i+1,0) - 2*v(i,0) + v(i-1,0));
    }

    // Left boundary
    i = 0;
    for(j = 1; j <= N-1; ++j)
    {
        Av(i, j) = -2.0/pow(hx,2)*(v(1,j) - v(0,j))
                    -1.0/pow(hy,2)*(v(0,j+1) - 2*v(0,j) + v(0,j-1));
    }

    // Inner area
    for(int i = 1; i <= M-2; ++i)
    {
        for(int j = 1; j <= N-2; ++j)
        {
            Av(i, j) = -1.0/pow(hx,2)*(v(i+1,j) - 2*v(i,j) + v(i-1,j)) 
                        -1.0/pow(hy,2)*(v(i,j+1) - 2*v(i,j) + v(i,j-1));
        }
    }

    // Right boundary
    i = M-1;
    for(int j = 1; j <= N-2; ++j)
    {
        Av(i, j) = -1.0/pow(hx,2)*(func_RBC(j) - 2*v(M-1,j) + v(M-2,j)) 
                    -1.0/pow(hy,2)*(v(M-1,j+1)  - 2*v(M-1,j) + v(M-1,j-1));
    }

    // Top boundary
    j = N-1;
    for(int i = 1; i <= M-2; ++i)
    {
        Av(i, j) = -1.0/pow(hx,2)*(v(i+1,N-1)  - 2*v(i,N-1) + v(i-1,N-1)) 
                    -1.0/pow(hy,2)*(func_TBC(i) - 2*v(i,N-1) + v(i,N-2));
    }

    // Edge point: (X2, Y2)
    i = M-1; 
    j = N-1;
    Av(i, j) = -1.0/pow(hx,2)*(func_RBC(j) - 2*v(M-1,N-1) + v(M-2,N-1)) 
                -1.0/pow(hy,2)*(func_TBC(i) - 2*v(M-1,N-1) + v(M-1,N-2));

    return 0;
}

int LaplaceOperator::rhs(MeshVec &f) const
{
    assert(M == f.get_M()  && N == f.get_N()); 

    int i, j;

    // Edge point: (X1, Y1)
    i = 0; j = 0;
    f(i, j) = func_F(0, 0) - (2.0/hx + 2.0/hy)*func_BBC(0);

    // Bottom boundary
    j = 0;
    for(i = 1; i <= M-1; ++i)
    {
        f(i, j) = func_F(i, 0) - 2.0/hy * func_BBC(i);
    }

    // Left boundary
    i = 0;
    for(j = 1; j <= N-1; ++j)
    {
        f(i ,j) = func_F(0, j) - 2.0/hx * func_LBC(j);
    }

    // Inner area
    for(int i = 1; i <= M-2; ++i)
    {
        for(int j = 1; j <= N-2; ++j)
        {
            f(i, j) = func_F(i, j);
        }
    }

    // Right boundary
    i = M-1;
    for(int j = 1; j <= N-2; ++j)
    {
        f(i, j) = func_F(M-1, j);
    }

    // Top boundary
    j = N-1;
    for(int i = 1; i <= M-2; ++i)
    {
        f(i, j) = func_F(i, N-1);
    }

    // Edge point: (X2, Y2)
    i = M-1; 
    j = N-1;
    f(i, j) = func_F(M-1, N-1);

    return 0;
}