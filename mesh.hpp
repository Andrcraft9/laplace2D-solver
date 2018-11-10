#include <iostream>
#include <string>
#include <cassert>
#include <cmath>

#ifndef MESH_H
#define MESH_H

class MeshVec
{
private:
    int M, N;
    double *vec;

public:
    MeshVec(int m, int n, double val = 0) : M(m), N(n)
    {
        assert(M > 0 && N > 0);
        vec = new double[M*N];

        for(int i = 0; i < M; ++i)
        {
            for(int j = 0; j < N; ++j)
            {
                (*this)(i, j) = val;
            }
        }
    } 

    MeshVec(const MeshVec& v) : M(v.M), N(v.N)
    {
        vec = new double[M*N];
        for(int i = 0; i < M; ++i)
        {
            for(int j = 0; j < N; ++j)
            {
                (*this)(i, j) = v(i, j);
            }
        }
    }
    
    MeshVec& operator=(const MeshVec& v)
    {
        assert(M == v.M && N == v.N);
        for(int i = 0; i < M; ++i)
        {
            for(int j = 0; j < N; ++j)
            {
                (*this)(i, j) = v(i, j);
            }
        }

        return *this;
    }
    
    int get_M() const 
    { 
        return M;
    }
    int get_N() const 
    { 
        return N; 
    }
    
    const double& operator()(int i, int j) const
    {
        //return vec[i + j*M];
        return vec[j + i*N];
    }

    double& operator()(int i, int j)
    {
        //return vec[i + j*M];
        return vec[j + i*N];
    }

    // y = a*x + y
    MeshVec& axpy(double a, const MeshVec& x)
    {
        assert(M == x.M && N == x.N);
        for(int i = 0; i < M; ++i)
        {
            for(int j = 0; j < N; ++j)
            {
                (*this)(i, j) = (*this)(i, j) + a*x(i, j);
            }
        }

        return *this;
    }

    MeshVec& operator+=(const MeshVec& v)
    {
        assert(M == v.M && N == v.N);
        for(int i = 0; i < M; ++i)
        {
            for(int j = 0; j < N; ++j)
            {
                (*this)(i, j) = (*this)(i, j) + v(i, j);
            }
        }
        return *this;
    }

    MeshVec& operator-=(const MeshVec& v)
    {
        assert(M == v.M && N == v.N);
        for(int i = 0; i < M; ++i)
        {
            for(int j = 0; j < N; ++j)
            {
                (*this)(i, j) = (*this)(i, j) - v(i, j);
            }
        }
        return *this;
    }

    friend std::ostream& operator<< (std::ostream& os, const MeshVec& v);

    ~MeshVec()
    {
        delete[] vec;
    }
};

#endif