#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include "mpitools.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#ifndef MESH_H
#define MESH_H

// CUDA/MPI version. 
// For all class members:
// member() - apply on host data
// member_device() - apply on device data
class MeshVec
{
private:
    MPITools mtls;
    thrust::host_vector<double> vec; // host data
    thrust::device_vector<double> vec_device; // device data
    double *sendbuf, *recvbuf;

    int direct_sync(int src, int dist, double *sbuf, int sn, double *rbuf, int rn);

public:
    MeshVec(MPITools mtls, double val = 0) : mtls(mtls), vec(mtls.bndM() * mtls.bndN())
    {
        assert(mtls.initialized());

        sendbuf = new double[std::max(mtls.bndM(), mtls.bndN())];
        recvbuf = new double[std::max(mtls.bndM(), mtls.bndN())];

        for(int i = mtls.bndx1(); i <= mtls.bndx2(); ++i)
            for(int j = mtls.bndy1(); j <= mtls.bndy2(); ++j)
                (*this)(i, j) = val;
    } 

    MeshVec(const MeshVec& v) : mtls(v.mtls), vec(mtls.bndM() * mtls.bndN())
    {
        sendbuf = new double[std::max(mtls.bndM(), mtls.bndN())];
        recvbuf = new double[std::max(mtls.bndM(), mtls.bndN())];

        for(int i = mtls.bndx1(); i <= mtls.bndx2(); ++i)
            for(int j = mtls.bndy1(); j <= mtls.bndy2(); ++j)
                (*this)(i, j) = v(i, j);
    }
    
    MeshVec& operator=(const MeshVec& v)
    {
        assert(mtls == v.mtls);

        for(int i = mtls.bndx1(); i <= mtls.bndx2(); ++i)
            for(int j = mtls.bndy1(); j <= mtls.bndy2(); ++j)
                (*this)(i, j) = v(i, j);

        return *this;
    }

    // From CPU to GPU
    void load_gpu() { vec_device = vec; }
    // From GPU to CPU
    void unload_gpu() { vec = vec_device; }
    // From CPU to GPU, only halo points!
    void load_halo_gpu();
    // From GPU to CPU, only halo points!
    void unload_halo_gpu();
    // Sync halo points between CPUs
    int sync();
    
    // host vec
    thrust::host_vector<double>& get_host_vec() { return vec; }
    const thrust::host_vector<double>& get_host_vec() const { return vec; }
    // device vec
    thrust::device_vector<double>& get_device_vec() { return vec_device; }
    const thrust::device_vector<double>& get_device_vec() const { return vec_device; }

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

    // y = a*x + y, host function
    MeshVec& axpy(double a, const MeshVec& x);
    // y = a*x + y, device function
    MeshVec& axpy_device(double a, const MeshVec& x);

    friend std::ostream& operator<< (std::ostream& os, const MeshVec& v);

    ~MeshVec()
    {
        delete[] sendbuf;
        delete[] recvbuf;
    }
};

#endif
