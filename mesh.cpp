#include "mesh.hpp"

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
    //#pragma omp parallel for
    for(int j = 0; j < mtls.locN(); ++j)
        sendbuf[j] = (*this)(mtls.locx2(), j + mtls.locy1());
    direct_sync(src[0], dist[0], sendbuf,  mtls.locN(), recvbuf,  mtls.locN());
    if (src[0] != MPI_PROC_NULL)
    {
        //#pragma omp parallel for
        for(int j = 0; j < mtls.locN(); ++j)
            (*this)(mtls.bndx1(), j + mtls.locy1()) = recvbuf[j];
    }

    // Y+ direction
    //#pragma omp parallel for
    for(int i = 0; i < mtls.locM(); ++i)
        sendbuf[i] = (*this)(mtls.locx1() + i, mtls.locy2());
    direct_sync(src[1], dist[1], sendbuf,  mtls.locM(), recvbuf,  mtls.locM());
    if (src[1] != MPI_PROC_NULL)
    {
        //#pragma omp parallel for
        for(int i = 0; i < mtls.locM(); ++i)
            (*this)(mtls.locx1() + i, mtls.bndy1()) = recvbuf[i];
    }

    // X- direction
    //#pragma omp parallel for
    for(int j = 0; j < mtls.locN(); ++j)
        sendbuf[j] = (*this)(mtls.locx1(), j + mtls.locy1());
    direct_sync(src[2], dist[2], sendbuf,  mtls.locN(), recvbuf,  mtls.locN());
    if (src[2] != MPI_PROC_NULL)
    {
        //#pragma omp parallel for
        for(int j = 0; j < mtls.locN(); ++j)
            (*this)(mtls.bndx2(), j + mtls.locy1()) = recvbuf[j];
    }

    // Y- direction
    //#pragma omp parallel for
    for(int i = 0; i < mtls.locM(); ++i)
        sendbuf[i] = (*this)(mtls.locx1() + i, mtls.locy1());
    direct_sync(src[3], dist[3], sendbuf,  mtls.locM(), recvbuf,  mtls.locM());
    if (src[3] != MPI_PROC_NULL)
    {
        //#pragma omp parallel for
        for(int i = 0; i < mtls.locM(); ++i)
            (*this)(mtls.locx1() + i, mtls.bndy2()) = recvbuf[i];
    }

    return 0;
}

std::ostream& operator<< (std::ostream& os, const MeshVec& v) 
{
    os << v.mtls.locM() << " " << v.mtls.locN() << std::endl;

    //for(int i = v.mtls.locx1(); i <= v.mtls.locx2(); ++i)
    //    for(int j = v.mtls.locy1(); j <= v.mtls.locy2(); ++j)
    for(int i = v.mtls.bndx1(); i <= v.mtls.bndx2(); ++i)
        for(int j = v.mtls.bndy1(); j <= v.mtls.bndy2(); ++j)
            os << "i, j: " << i << " , " << j << " v(i, j) = " << v(i, j) << std::endl;

    return os;
}
