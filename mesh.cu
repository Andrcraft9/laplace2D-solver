#include "mesh.hpp"

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>

MeshVec::MeshVec(MPITools mtls, double val) : 
    mtls(mtls), 
    vec( mtls.bndM() * mtls.bndN() ), 
    vec_device( mtls.bndM() * mtls.bndN() ), 
    // Host buffers
    sendbuf( std::max(mtls.bndM(), mtls.bndN()) ), 
    recvbuf( std::max(mtls.bndM(), mtls.bndN()) ),
    // Device buffers
    sendbuf_device( std::max(mtls.bndM(), mtls.bndN()) ), 
    recvbuf_device( std::max(mtls.bndM(), mtls.bndN()) ),
    // Host Additional buffers
    // X+
    isendbuf_XP( std::max(mtls.bndM(), mtls.bndN()) ),
    irecvbuf_XP( std::max(mtls.bndM(), mtls.bndN()) ),
    // X-
    isendbuf_XM( std::max(mtls.bndM(), mtls.bndN()) ),
    irecvbuf_XM( std::max(mtls.bndM(), mtls.bndN()) ),
    // Y+
    isendbuf_YP( std::max(mtls.bndM(), mtls.bndN()) ),
    irecvbuf_YP( std::max(mtls.bndM(), mtls.bndN()) ),
    // Y-
    isendbuf_YM( std::max(mtls.bndM(), mtls.bndN()) ),
    irecvbuf_YM( std::max(mtls.bndM(), mtls.bndN()) ),
    // Device Additional buffers
    // X+
    isendbuf_device_XP(std::max(mtls.bndM(), mtls.bndN())),
    irecvbuf_device_XP(std::max(mtls.bndM(), mtls.bndN())),
    // X-
    isendbuf_device_XM(std::max(mtls.bndM(), mtls.bndN())),
    irecvbuf_device_XM(std::max(mtls.bndM(), mtls.bndN())),
    // Y+
    isendbuf_device_YP(std::max(mtls.bndM(), mtls.bndN())),
    irecvbuf_device_YP(std::max(mtls.bndM(), mtls.bndN())),
    // Y-
    isendbuf_device_YM(std::max(mtls.bndM(), mtls.bndN())),
    irecvbuf_device_YM(std::max(mtls.bndM(), mtls.bndN()))
{
    assert(mtls.initialized());
    
    thrust::fill(vec.begin(), vec.end(), val);
    thrust::fill(vec_device.begin(), vec_device.end(), val);

    init_sync_buffers();
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

/*
struct is_halo_y : public thrust::unary_function<int, bool>
{
    const int locN;
    const int istart;
    const int jstart;

    const int bndN;
    const int bnd_x1;
    const int bnd_y1;

    is_halo_y(const int _bndN, const int _bnd_x1, const int _bnd_y1, const int _locN, const int _istart, const int _jstart) 
        :  bndN(_bndN), bnd_x1(_bnd_x1), bnd_y1(_bnd_y1), locN(_locN), istart(_istart), jstart(_jstart) {}

    __host__ __device__
        bool operator()(const int &k) const 
        { 
            const int i = (k / bndN) + bnd_x1;
            const int j = (k % bndN) + bnd_y1;

            if (i == istart && j >= jstart && j < jstart + locN)
                return true;
            else
                return false;
        }
};
typedef struct is_halo_y IsHaloY;

struct is_halo_x : public thrust::unary_function<int, bool>
{
    const int locM;
    const int istart;
    const int jstart;

    const int bndN;
    const int bnd_x1;
    const int bnd_y1;

    is_halo_x(const int _bndN, const int _bnd_x1, const int _bnd_y1, const int _locM, const int _istart, const int _jstart) 
        :  bndN(_bndN), bnd_x1(_bnd_x1), bnd_y1(_bnd_y1), locM(_locM), istart(_istart), jstart(_jstart) {}

    __host__ __device__
        bool operator()(const int &k) const 
        { 
            const int i = (k / bndN) + bnd_x1;
            const int j = (k % bndN) + bnd_y1;

            if (j == jstart && i >= istart && i < istart + locM)
                return true;
            else
                return false;
        }
};
typedef struct is_halo_x IsHaloX;

// X+ direction
//for(int j = 0; j < mtls.locN(); ++j)
//    sendbuf_device[j] = vec_device[get_index(mtls.locx2(), j + mtls.locy1())];

thrust::copy_if(vec_device.begin(), vec_device.end(), thrust::make_counting_iterator(0), sendbuf_device.begin(), 
                IsHaloY(mtls.bndN(), mtls.bndx1(), mtls.bndy1(), 
                        mtls.locN(), mtls.locx2(), mtls.locy1() ));  
*/

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
        //for(int j = 0; j < mtls.locN(); ++j)
        //    sendbuf_device[j] = vec_device[get_index(mtls.locx2(), j + mtls.locy1())];
        thrust::gather(isendbuf_device_XP.begin(), isendbuf_device_XP.begin() + mtls.locN(), 
                       vec_device.begin(), 
                       sendbuf_device.begin());

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
            //for(int j = 0; j < mtls.locN(); ++j)
            //    vec_device[get_index(mtls.bndx1(), j + mtls.locy1())] = recvbuf_device[j];
            thrust::scatter(recvbuf_device.begin(), recvbuf_device.begin() + mtls.locN(), 
                            irecvbuf_device_XP.begin(), 
                            vec_device.begin());
            
        }

        // Y+ direction
        //for(int i = 0; i < mtls.locM(); ++i)
        //    sendbuf_device[i] = vec_device[get_index(mtls.locx1() + i, mtls.locy2())];
        thrust::gather(isendbuf_device_YP.begin(), isendbuf_device_YP.begin() + mtls.locM(), 
                       vec_device.begin(), 
                       sendbuf_device.begin());
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
            //for(int i = 0; i < mtls.locM(); ++i)
            //    vec_device[get_index(mtls.locx1() + i, mtls.bndy1())] = recvbuf_device[i];
            thrust::scatter(recvbuf_device.begin(), recvbuf_device.begin() + mtls.locM(), 
                            irecvbuf_device_YP.begin(), 
                            vec_device.begin());
        }

        // X- direction
        //for(int j = 0; j < mtls.locN(); ++j)
        //    sendbuf_device[j] = vec_device[get_index(mtls.locx1(), j + mtls.locy1())];
        thrust::gather(isendbuf_device_XM.begin(), isendbuf_device_XM.begin() + mtls.locN(), 
                       vec_device.begin(), 
                       sendbuf_device.begin());
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
            //for(int j = 0; j < mtls.locN(); ++j)
            //    vec_device[get_index(mtls.bndx2(), j + mtls.locy1())] = recvbuf_device[j];
            thrust::scatter(recvbuf_device.begin(), recvbuf_device.begin() + mtls.locN(), 
                            irecvbuf_device_XM.begin(), 
                            vec_device.begin());
        }

        // Y- direction
        //for(int i = 0; i < mtls.locM(); ++i)
        //    sendbuf_device[i] = vec_device[get_index(mtls.locx1() + i, mtls.locy1())];
        thrust::gather(isendbuf_device_YM.begin(), isendbuf_device_YM.begin() + mtls.locM(), 
                       vec_device.begin(), 
                       sendbuf_device.begin());
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
            //for(int i = 0; i < mtls.locM(); ++i)
            //    vec_device[get_index(mtls.locx1() + i, mtls.bndy2())] = recvbuf_device[i];
            thrust::scatter(recvbuf_device.begin(), recvbuf_device.begin() + mtls.locM(), 
                            irecvbuf_device_YM.begin(), 
                            vec_device.begin());
        }
    }

    return 0;
}

void MeshVec::init_sync_buffers()
{
    // X+ direction
    for(int j = 0; j < mtls.locN(); ++j)
        isendbuf_XP[j] = get_index(mtls.locx2(), j + mtls.locy1());
        //sendbuf[j] = (*this)(mtls.locx2(), j + mtls.locy1());
    for(int j = 0; j < mtls.locN(); ++j)
        irecvbuf_XP[j] = get_index(mtls.bndx1(), j + mtls.locy1());
        //(*this)(mtls.bndx1(), j + mtls.locy1()) = recvbuf[j];

    // Y+ direction
    for(int i = 0; i < mtls.locM(); ++i)
        isendbuf_YP[i] = get_index(mtls.locx1() + i, mtls.locy2());
        //sendbuf[i] = (*this)(mtls.locx1() + i, mtls.locy2());
    for(int i = 0; i < mtls.locM(); ++i)
        irecvbuf_YP[i] = get_index(mtls.locx1() + i, mtls.bndy1());
        //(*this)(mtls.locx1() + i, mtls.bndy1()) = recvbuf[i];

    // X- direction
    for(int j = 0; j < mtls.locN(); ++j)
        isendbuf_XM[j] = get_index(mtls.locx1(), j + mtls.locy1());
        //sendbuf[j] = (*this)(mtls.locx1(), j + mtls.locy1());
    for(int j = 0; j < mtls.locN(); ++j)
        irecvbuf_XM[j] = get_index(mtls.bndx2(), j + mtls.locy1());

    // Y- direction
    for(int i = 0; i < mtls.locM(); ++i)
        isendbuf_YM[i] = get_index(mtls.locx1() + i, mtls.locy1());
        //sendbuf[i] = (*this)(mtls.locx1() + i, mtls.locy1());
    for(int i = 0; i < mtls.locM(); ++i)
        irecvbuf_YM[i] = get_index(mtls.locx1() + i, mtls.bndy2());
        //(*this)(mtls.locx1() + i, mtls.bndy2()) = recvbuf[i];

    // From host to device
    isendbuf_device_XP = isendbuf_XP;
    irecvbuf_device_XP = irecvbuf_XP;

    isendbuf_device_YP = isendbuf_YP;
    irecvbuf_device_YP = irecvbuf_YP;

    isendbuf_device_XM = isendbuf_XM;
    irecvbuf_device_XM = irecvbuf_XM;

    isendbuf_device_YM = isendbuf_YM;
    irecvbuf_device_YM = irecvbuf_YM;
}