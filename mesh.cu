#include "mesh.hpp"

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>

MeshVec::MeshVec(MPITools mtls, double val) : mtls(mtls), vec(mtls.bndM() * mtls.bndN()), vec_device(mtls.bndM() * mtls.bndN()), 
    sendbuf(std::max(mtls.bndM(), mtls.bndN())), recvbuf(std::max(mtls.bndM(), mtls.bndN())),
    sendbuf_device(std::max(mtls.bndM(), mtls.bndN())), recvbuf_device(std::max(mtls.bndM(), mtls.bndN()))
{
    assert(mtls.initialized());

    for(int i = mtls.bndx1(); i <= mtls.bndx2(); ++i)
        for(int j = mtls.bndy1(); j <= mtls.bndy2(); ++j)
            (*this)(i, j) = val;
} 

MeshVec::MeshVec(const MeshVec& v) : mtls(v.mtls), vec(mtls.bndM() * mtls.bndN()), vec_device(mtls.bndM() * mtls.bndN()), 
    sendbuf(std::max(mtls.bndM(), mtls.bndN())), recvbuf(std::max(mtls.bndM(), mtls.bndN())),
    sendbuf_device(std::max(mtls.bndM(), mtls.bndN())), recvbuf_device(std::max(mtls.bndM(), mtls.bndN()))
{
    for(int i = mtls.bndx1(); i <= mtls.bndx2(); ++i)
        for(int j = mtls.bndy1(); j <= mtls.bndy2(); ++j)
            (*this)(i, j) = v(i, j);
}

struct saxpy_functor : public thrust::binary_function<double,double,double>
{
    const double a;

    saxpy_functor(double _a) : a(_a) {}

    __host__ __device__
        double operator()(const double& x, const double& y) const 
        { 
            return a * x + y;
        }
};

MeshVec& MeshVec::axpy_device(double a, const MeshVec& x)
{
    assert(mtls == x.mtls);

    thrust::transform(x.vec_device.begin(), x.vec_device.end(), 
                      vec_device.begin(), 
                      vec_device.begin(), 
                      saxpy_functor(a));
    
    //cudaDeviceSynchronize();
    return *this;
}

int MeshVec::direct_sync(int src, int dist, double *sbuf, int sn, double *rbuf, int rn)
{
    if (src != MPI_PROC_NULL && dist != MPI_PROC_NULL)
    {
        // Send-recv
        MPI_Sendrecv(sbuf, sn, MPI_DOUBLE, dist, 0, rbuf, rn, MPI_DOUBLE, src, 0, mtls.comm(), MPI_STATUS_IGNORE);
    }
    else 
    {
        if (src != MPI_PROC_NULL)
        {
            // Only recv
            MPI_Recv(rbuf, rn, MPI_DOUBLE, src, 0, mtls.comm(), MPI_STATUS_IGNORE);
        }

        if (dist != MPI_PROC_NULL)
        {
            // Only send
            MPI_Send(sbuf, sn, MPI_DOUBLE, dist, 0, mtls.comm());
        }
    }

    return 0;
}

int MeshVec::sync()
{
    int src[4], dist[4];

    if (mtls.procs() > 1)
    {
        //unload_gpu();

        // X+ direction
        MPI_Cart_shift(mtls.comm(), 0, 1,  src, dist); 
        // Y+ direction
        MPI_Cart_shift(mtls.comm(), 1, 1,  src + 1, dist + 1); 
        // X- direction
        // MPI_Cart_shift(mtls.comm(), 0, -1, src + 2, dist + 2);
        src[2] = dist[0];
        dist[2] = src[0];
        // Y- direction
        // MPI_Cart_shift(mtls.comm(), 1, -1, src + 3, dist + 3); 
        src[3] = dist[1];
        dist[3] = src[1];

        // X+ direction
        for(int j = 0; j < mtls.locN(); ++j)
            sendbuf_device[j] = vec_device[get_index(mtls.locx2(), j + mtls.locy1())];
        #ifdef _CUDA_AWARE_MPI_SYNC_
            direct_sync(src[0], dist[0], thrust::raw_pointer_cast(sendbuf_device.data()),  mtls.locN(), 
                                         thrust::raw_pointer_cast(recvbuf_device.data()),  mtls.locN());
        #else
            sendbuf = sendbuf_device;
            direct_sync(src[0], dist[0], thrust::raw_pointer_cast(sendbuf.data()),  mtls.locN(), 
                                         thrust::raw_pointer_cast(recvbuf.data()),  mtls.locN());
            recvbuf_device = recvbuf;
        #endif
        if (src[0] != MPI_PROC_NULL)
        {
            for(int j = 0; j < mtls.locN(); ++j)
                vec_device[get_index(mtls.bndx1(), j + mtls.locy1())] = recvbuf_device[j];
        }

        // Y+ direction
        for(int i = 0; i < mtls.locM(); ++i)
            sendbuf_device[i] = vec_device[get_index(mtls.locx1() + i, mtls.locy2())];
        #ifdef _CUDA_AWARE_MPI_SYNC_
            direct_sync(src[1], dist[1], thrust::raw_pointer_cast(sendbuf_device.data()),  mtls.locM(), 
                                         thrust::raw_pointer_cast(recvbuf_device.data()),  mtls.locM());
        #else
            sendbuf = sendbuf_device;
            direct_sync(src[1], dist[1], thrust::raw_pointer_cast(sendbuf.data()),  mtls.locM(), 
                                         thrust::raw_pointer_cast(recvbuf.data()),  mtls.locM());
            recvbuf_device = recvbuf;
        #endif
        if (src[1] != MPI_PROC_NULL)
        {
            for(int i = 0; i < mtls.locM(); ++i)
                vec_device[get_index(mtls.locx1() + i, mtls.bndy1())] = recvbuf_device[i];
        }

        // X- direction
        for(int j = 0; j < mtls.locN(); ++j)
            sendbuf_device[j] = vec_device[get_index(mtls.locx1(), j + mtls.locy1())];
        #ifdef _CUDA_AWARE_MPI_SYNC_
            direct_sync(src[2], dist[2], thrust::raw_pointer_cast(sendbuf_device.data()),  mtls.locN(), 
                                         thrust::raw_pointer_cast(recvbuf_device.data()),  mtls.locN());
        #else
            sendbuf = sendbuf_device;
            direct_sync(src[2], dist[2], thrust::raw_pointer_cast(sendbuf.data()),  mtls.locN(), 
                                         thrust::raw_pointer_cast(recvbuf.data()),  mtls.locN());
            recvbuf_device = recvbuf;        
        #endif
        if (src[2] != MPI_PROC_NULL)
        {
            for(int j = 0; j < mtls.locN(); ++j)
                vec_device[get_index(mtls.bndx2(), j + mtls.locy1())] = recvbuf_device[j];
        }

        // Y- direction
        for(int i = 0; i < mtls.locM(); ++i)
            sendbuf_device[i] = vec_device[get_index(mtls.locx1() + i, mtls.locy1())];
        #ifdef _CUDA_AWARE_MPI_SYNC_
            direct_sync(src[3], dist[3], thrust::raw_pointer_cast(sendbuf_device.data()),  mtls.locM(), 
                                         thrust::raw_pointer_cast(recvbuf_device.data()),  mtls.locM());
        #else
            sendbuf = sendbuf_device;
            direct_sync(src[3], dist[3], thrust::raw_pointer_cast(sendbuf.data()),  mtls.locM(), 
                                         thrust::raw_pointer_cast(recvbuf.data()),  mtls.locM());
            recvbuf_device = recvbuf;
        #endif
        if (src[3] != MPI_PROC_NULL)
        {
            for(int i = 0; i < mtls.locM(); ++i)
                vec_device[get_index(mtls.locx1() + i, mtls.bndy2())] = recvbuf_device[i];
        }

    }

    return 0;
}