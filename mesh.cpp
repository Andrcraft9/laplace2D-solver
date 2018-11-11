#include "mesh.hpp"

int MeshVec::direct_sync(int src, int dist, double *sbuf, int sn, double *rbuf, int rn)
{
    if (src != MPI_PROC_NULL && dist != MPI_PROC_NULL)
    {
        // Send-recv
        MPI_Sendrecv(sbuf, sn, MPI_DOUBLE, dist, 0, rbuf, rn, MPI_DOUBLE, src, 0, mptls.comm(), MPI_STATUS_IGNORE);
    }
    else 
    {
        if (src != MPI_PROC_NULL)
        {
            // Only recv
            MPI_Recv(rbuf, rn, MPI_DOUBLE, src, 0, mptls.comm(), MPI_STATUS_IGNORE);
        }

        if (dist != MPI_PROC_NULL)
        {
            // Only send
            MPI_Send(sbuf, sn, MPI_DOUBLE, dist, 0, mptls.comm());
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
    for(int j = 0; j <= mptls.locN(); ++j)
        sendbuf[j] = (*this)(mptsl.locx2(), j + mplsl.locy1());
    direct_sync(src[0], dist[0], sendbuf,  mptls.locN(), recvbuf,  mptls.locN());
    if (src[0] != MPI_PROC_NULL)
    {
        for(int j = 0; j <= mptls.locN(); ++j)
            (*this)(mptsl.bndx1(), j + mplsl.locy1()) = recvbuf[j];
    }

    // Y+ direction
    for(int j = 0; j <= mptls.locN(); ++j)
        sendbuf[j] = (*this)(mptsl.locx2(), j + mplsl.locy1());
    direct_sync(src[0], dist[0], sendbuf,  mptls.locN(), recvbuf,  mptls.locN());
    if (src[0] != MPI_PROC_NULL)
    {
        for(int j = 0; j <= mptls.locN(); ++j)
            (*this)(mptsl.bndx1(), j + mplsl.locy1()) = recvbuf[j];
    }

}

std::ostream& operator<< (std::ostream& os, const MeshVec& v) 
{
    os << v.mtls.locM() << " " << v.mtls.locN() << std::endl;

    for(int i = v.mtls.locx1(); i <= v.mtls.locx2(); ++i)
        for(int j = v.mtls.locy1(); j <= v.mtls.locy2(); ++j)
            os << "i, j: " << i << " , " << j << " v(i, j) = " << v(i, j) << std::endl;

    return os;
}