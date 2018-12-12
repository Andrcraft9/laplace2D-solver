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
    // host data
    thrust::host_vector<double> vec; 
    // device data
    thrust::device_vector<double> vec_device; 

    // Buffers for communications
    // for send
    thrust::host_vector<double> sendbuf;
    thrust::device_vector<double> sendbuf_device;
    // for recv
    thrust::host_vector<double> recvbuf;
    thrust::device_vector<double> recvbuf_device;
    
    // Host Additional buffers, indexes for gather/scatter
    // X+
    thrust::host_vector<double> isendbuf_XP;
    thrust::host_vector<double> irecvbuf_XP;
    // X-
    thrust::host_vector<double> isendbuf_XM;
    thrust::host_vector<double> irecvbuf_XM;
    // Y+
    thrust::host_vector<double> isendbuf_YP;
    thrust::host_vector<double> irecvbuf_YP;
    // Y-
    thrust::host_vector<double> isendbuf_YM;
    thrust::host_vector<double> irecvbuf_YM;
    // Device Additional buffers, indexes for gather/scatter
    // X+
    thrust::device_vector<double> isendbuf_device_XP;
    thrust::device_vector<double> irecvbuf_device_XP;
    // X-
    thrust::device_vector<double> isendbuf_device_XM;
    thrust::device_vector<double> irecvbuf_device_XM;
    // Y+
    thrust::device_vector<double> isendbuf_device_YP;
    thrust::device_vector<double> irecvbuf_device_YP;
    // Y-
    thrust::device_vector<double> isendbuf_device_YM;
    thrust::device_vector<double> irecvbuf_device_YM;

    void init_sync_buffers();

    int direct_sync(int src, int dist, double *sbuf, int sn, double *rbuf, int rn);

    // No copy pls
    MeshVec(const MeshVec& v);
    MeshVec& operator=(const MeshVec& v);

public:
    MeshVec(MPITools mtls, double val = 0);

    // From CPU to GPU
    void load_gpu() { vec_device = vec; }
    // From GPU to CPU
    void unload_gpu() { vec = vec_device; }
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

    int get_index(int i, int j) const
    {
        return (j - mtls.bndy1()) + (i - mtls.bndx1())*mtls.bndN();
    }

    // y = a*x + y, host function
    MeshVec& axpy(double a, const MeshVec& x);
    // y = a*x + y, device function
    MeshVec& axpy_device(double a, const MeshVec& x);

    friend std::ostream& operator<< (std::ostream& os, const MeshVec& v);

    ~MeshVec() {}
};

#endif
