#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include "mpitools.hpp"

#ifndef MESH_H
#define MESH_H

class MeshVec
{
private:
    MPITools mtls;
    double *vec;
    double *sendbuf, *recvbuf;

    int direct_sync(int src, int dist, double *sbuf, int sn, double *rbuf, int rn);

public:
    MeshVec(MPITools mtls, double val = 0) : mtls(mtls)
    {
        assert(mtls.initialized());

        vec = new double[mtls.bndM() * mtls.bndN()];
        sendbuf = new double[std::max(mtls.bndM(), mtls.bndN())];
        recvbuf = new double[std::max(mtls.bndM(), mtls.bndN())];

        for(int i = mtls.bndx1(); i <= mtls.bndx2(); ++i)
            for(int j = mtls.bndy1(); j <= mtls.bndy2(); ++j)
                (*this)(i, j) = val;
    } 

    MeshVec(const MeshVec& v) : mtls(v.mtls)
    {
        vec = new double[mtls.bndM() * mtls.bndN()];
        sendbuf = new double[std::max(mtls.bndM(), mtls.bndN())];
        recvbuf = new double[std::max(mtls.bndM(), mtls.bndN())];

        for(int i = mtls.bndx1(); i <= mtls.bndx2(); ++i)
            for(int j = mtls.bndy1(); j <= mtls.bndy2(); ++j)
                (*this)(i, j) = v(i, j);
    }
    
    MeshVec& operator=(const MeshVec& v)
    {
        assert(mtls.M() == v.mtls.M() && mtls.N() == v.mtls.N());

        for(int i = mtls.bndx1(); i <= mtls.bndx2(); ++i)
            for(int j = mtls.bndy1(); j <= mtls.bndy2(); ++j)
                (*this)(i, j) = v(i, j);

        return *this;
    }

    int sync();
    
    const MPITools& mpitools() const { return mtls; }
    
    const double& operator()(int i, int j) const
    {
        //return vec[i + j*M];
        // j + iN
        return vec[(j - mtls.bndy1()) + (i - mtls.bndx1())*mtls.bndN()];
    }

    double& operator()(int i, int j)
    {
        //return vec[i + j*M];
        // j + iN
        return vec[(j - mtls.bndy1()) + (i - mtls.bndx1())*mtls.bndN()];
    }

    // y = a*x + y
    MeshVec& axpy(double a, const MeshVec& x)
    {
        assert(mtls.M() == x.mtls.M() && mtls.N() == x.mtls.N());

        for(int i = mtls.locx1(); i <= mtls.locx2(); ++i)
            for(int j = mtls.locy1(); j <= mtls.locy2(); ++j)
                (*this)(i, j) = (*this)(i, j) + a*x(i, j);

        return *this;
    }

    MeshVec& operator+=(const MeshVec& v)
    {
        assert(mtls.M() == v.mtls.M() && mtls.N() == v.mtls.N());

        for(int i = mtls.locx1(); i <= mtls.locx2(); ++i)
            for(int j = mtls.locy1(); j <= mtls.locy2(); ++j)
                (*this)(i, j) = (*this)(i, j) + v(i, j);

        return *this;
    }

    MeshVec& operator-=(const MeshVec& v)
    {
        assert(mtls.M() == v.mtls.M() && mtls.N() == v.mtls.N());

        for(int i = mtls.locx1(); i <= mtls.locx2(); ++i)
            for(int j = mtls.locy1(); j <= mtls.locy2(); ++j)
                (*this)(i, j) = (*this)(i, j) - v(i, j);
    
        return *this;
    }

    friend std::ostream& operator<< (std::ostream& os, const MeshVec& v);

    ~MeshVec()
    {
        delete[] vec;
        delete[] sendbuf;
        delete[] recvbuf;
    }
};

#endif