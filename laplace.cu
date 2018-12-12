#include "laplace.hpp"

#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#define BLOCK_SIZE 16

typedef thrust::tuple<double, double, double> Double3;
typedef thrust::tuple<double, int> DoubleInt;

// This functors takes: r, Ar 
// and returns: (r, r), (Ar, r), (Ar, Ar)
struct dot_general_mult: public thrust::binary_function<double, double, Double3>
{
    __host__ __device__
        Double3 operator()(const double& r, const double& Ar) const
        {
            return Double3( r * r, 
                            Ar * r,
                            Ar * Ar );
        }
};
struct dot_general_plus: public thrust::binary_function<Double3, Double3, Double3>
{
    __host__ __device__
        Double3 operator()(const Double3& v1, const Double3& v2) const
        {
            return Double3( thrust::get<0>(v1) + thrust::get<0>(v2), 
                            thrust::get<1>(v1) + thrust::get<1>(v2),
                            thrust::get<2>(v1) + thrust::get<2>(v2) );
        }
};

// This returns value only in inner_area
struct inner_area: public thrust::unary_function<DoubleInt, double>
{
    const int inner_x1; 
    const int inner_x2; 
    const int inner_y1; 
    const int inner_y2;
    const int bndN;
    const int bnd_x1;
    const int bnd_y1;

    inner_area(const int _inner_x1, const int _inner_x2, const int _inner_y1, const int _inner_y2,
               const int _bndN, const int _bnd_x1, const int _bnd_y1) 
    : inner_x1(_inner_x1), inner_x2(_inner_x2), inner_y1(_inner_y1), inner_y2(_inner_y2),
      bndN(_bndN), bnd_x1(_bnd_x1), bnd_y1(_bnd_y1) {}
    __host__ __device__
        double operator()(const DoubleInt& vk) const 
        { 
            const double v = thrust::get<0>(vk);
            const int k = thrust::get<1>(vk);

            const int i = (k / bndN) + bnd_x1;
            const int j = (k % bndN) + bnd_y1;

            double inner_v = 0.0;

            // Inner area
            if (i >= inner_x1 && i <= inner_x2 && j >= inner_y1 && j <= inner_y2) 
                inner_v = v;
            
            return inner_v;
        }
};


std::vector<double> LaplaceOperator::dot_general_mesh_device(const MeshVec& r, const MeshVec& Ar) const
{
    assert(mtls == r.mpitools()); 
    assert(mtls == Ar.mpitools()); 

    // Make iterator only in inner area, without halo points
    struct inner_area inner_op(mtls.locx1(),  mtls.locx2(),  mtls.locy1(),  mtls.locy2(),
                               mtls.bndN(), mtls.bndx1(), mtls.bndy1()); 
    //auto = thrust::transform_iterator<clamp, thrust::device_vector<int>::iterator> 
    auto r_begin = thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(r.get_device_vec().begin(), thrust::make_counting_iterator(0))),
                                                   inner_op);
    auto r_end = r_begin + r.get_device_vec().size();
    auto Ar_begin = thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(Ar.get_device_vec().begin(), thrust::make_counting_iterator(0))),
                                                    inner_op);

    // Compute inner products: (r, r), (Ar, r), (Ar, Ar)
    Double3 init(0.0, 0.0, 0.0);
    struct dot_general_plus binary_plus;
    struct dot_general_mult binary_mult;                  
    Double3 dot = thrust::inner_product(r_begin, r_end, 
                                        Ar_begin, 
                                        init, 
                                        binary_plus, 
                                        binary_mult);

/*
    Double3 dot = thrust::inner_product(//thrust::device,
                                        r.get_device_vec().begin(), r.get_device_vec().end(), 
                                        Ar.get_device_vec().begin(), 
                                        init, 
                                        binary_plus, 
                                        binary_mult);
*/                                      
/*
    thrust::plus<double>       binary_plus;
    thrust::multiplies<double> binary_mult;
    double init;

    std::vector<double> dot_out(3);
    init = 0.0;
    dot_out[0] = hx * hy * thrust::inner_product(r.get_device_vec().begin(), r.get_device_vec().end(), 
                                                 r.get_device_vec().begin(), 
                                                 init, 
                                                 binary_plus, 
                                                 binary_mult);
    
    init = 0.0;
    dot_out[1] = hx * hy * thrust::inner_product(Ar.get_device_vec().begin(), Ar.get_device_vec().end(), 
                                                 r.get_device_vec().begin(), 
                                                 init, 
                                                 binary_plus, 
                                                 binary_mult);
    
    init = 0.0;
    dot_out[2] = hx * hy * thrust::inner_product(Ar.get_device_vec().begin(), Ar.get_device_vec().end(), 
                                                 Ar.get_device_vec().begin(), 
                                                 init, 
                                                 binary_plus, 
                                                 binary_mult);
*/    
    //cudaDeviceSynchronize();
    // Sync and return inner products
    std::vector<double> dot_out(3);
    dot_out[0] = hx * hy * thrust::get<0>(dot); // (r, r)
    dot_out[1] = hx * hy * thrust::get<1>(dot); // (Ar, r)
    dot_out[2] = hx * hy * thrust::get<2>(dot); // (Ar, Ar)

    double idot, idot_out;
    
    idot = dot_out[0];
    MPI_Allreduce(&idot, &idot_out, 1, MPI_DOUBLE, MPI_SUM, mtls.comm());
    dot_out[0] = idot_out;
    
    idot = dot_out[1];
    MPI_Allreduce(&idot, &idot_out, 1, MPI_DOUBLE, MPI_SUM, mtls.comm());
    dot_out[1] = idot_out;
    
    idot = dot_out[2];
    MPI_Allreduce(&idot, &idot_out, 1, MPI_DOUBLE, MPI_SUM, mtls.comm());
    dot_out[2] = idot_out;

    return dot_out;
}

/****************************************************************************************************************/
// CUDA Kernel matvec
__host__ __device__ int iDivUp(int a, int b)
{ 
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__global__ void matvec_kernel(const double *v, const double *f, double *Av, 
                              const double hx, const double hy,
                              const int M, const int N,
                              const int inner_x1, const int inner_x2, const int inner_y1, const int inner_y2,
                              const int bndN, const int bnd_x1, const int bnd_y1)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    // MeshVec index, see MeshVec class
    //int k = (j - bnd_y1) + (i - bnd_x1)*bndN;
    const int k = j + i*bndN;
    i = (k / bndN) + bnd_x1;
    j = (k % bndN) + bnd_y1;

    double cAv; // Av value

    const double hx2 = hx*hx;
    const double hy2 = hy*hy;
    const double vij   = v[k];     // v(i, j)
    //double vipj  = v[k + bndN]; // v(i+1, j)
    //double vijp  = v[k + 1]; // v(i, j+1)
    //double vimj  = v[k - bndN]; // v(i-1, j)
    //double vijm  = v[k - 1]; // v(i, j-1)

    // Compute Av value
    if (i >= inner_x1 && i <= inner_x2 && j >= inner_y1 && j <= inner_y2)
    {
        // Inner area
        
        const double vipj  = v[k + bndN]; // v(i+1, j)
        const double vijp  = v[k + 1]; // v(i, j+1)
        const double vimj  = v[k - bndN]; // v(i-1, j)
        const double vijm  = v[k - 1]; // v(i, j-1)

        cAv = -1.0/hx2*(vipj - 2*vij + vimj) 
              -1.0/hy2*(vijp - 2*vij + vijm);
    }
    else if (i == 0 && j == 0)
    {
        // Edge point: (X1, Y1)
        
        const double vipj  = v[k + bndN]; // v(i+1, j)
        const double vijp  = v[k + 1]; // v(i, j+1)

        cAv = -2.0/hx2*(vipj - vij) 
              -2.0/hy2*(vijp - vij);
    }
    else if (j == 0)
    {
        // Bottom boundary
        
        const double vipj  = v[k + bndN]; // v(i+1, j)
        const double vijp  = v[k + 1]; // v(i, j+1)
        const double vimj  = v[k - bndN]; // v(i-1, j)

        if (i >= inner_x1 && i <= inner_x2)
        {
            cAv = -2.0/hy2*(vijp - vij) 
                  -1.0/hx2*(vipj - 2*vij + vimj);
        }
        else if (i == M-1)
        {
            cAv = -2.0/hy2*(vijp - vij) 
                  -1.0/hx2*( -2*vij + vimj);
        }
    }
    else if (i == 0)
    {
        // Left boundary
        
        const double vipj  = v[k + bndN]; // v(i+1, j)
        const double vijp  = v[k + 1]; // v(i, j+1)
        const double vijm  = v[k - 1]; // v(i, j-1)

        if (j >= inner_y1 && j <= inner_y2)
        {
            cAv = -2.0/hx2*(vipj - vij)
                  -1.0/hy2*(vijp - 2*vij + vijm);
        }
        else if (j == N-1)
        {
            cAv = -2.0/hx2*(vipj - vij)
                  -1.0/hy2*( -2*vij + vijm);
        }
    }
    else if (i == M-1 && j >= inner_y1 && j <= inner_y2)
    {
        // Right pre-boundary

        const double vijp  = v[k + 1]; // v(i, j+1)
        const double vimj  = v[k - bndN]; // v(i-1, j)
        const double vijm  = v[k - 1]; // v(i, j-1)

        cAv = -1.0/hx2*( -2*vij + vimj) 
              -1.0/hy2*(vijp  - 2*vij + vijm);
    }
    else if (j == N-1 && i >= inner_x1 && i <= inner_x2)
    {
        // Top pre-boundary

        const double vipj  = v[k + bndN]; // v(i+1, j)
        const double vimj  = v[k - bndN]; // v(i-1, j)
        const double vijm  = v[k - 1]; // v(i, j-1)

        cAv = -1.0/hx2*(vipj  - 2*vij + vimj) 
              -1.0/hy2*( -2*vij + vijm);
    
    }
    else if (i == M-1 && j == N-1)
    {
        // Edge point: (X2, Y2)
        
        const double vimj  = v[k - bndN]; // v(i-1, j)
        const double vijm  = v[k - 1]; // v(i, j-1)

        cAv = -1.0/hx2*( -2*vij + vimj) 
              -1.0/hy2*( -2*vij + vijm);
    }
    else
    {
        cAv = f[k];
    }

    // Update device vector
    Av[k] = cAv - f[k];
}
/****************************************************************************************************************/

/****************************************************************************************************************/
 // Thrust transform matvec
struct matvec_functor: public thrust::binary_function<double, int, double>
{
    const double *v;
    const double *f;

    const double hx; 
    const double hy; 
    const int M; 
    const int N;
    const int inner_x1; 
    const int inner_x2; 
    const int inner_y1; 
    const int inner_y2;
    const int bndN;
    const int bnd_x1;
    const int bnd_y1;

    matvec_functor(const thrust::device_vector<double> &_v, const thrust::device_vector<double> &_f,
                   const double _hx, const double _hy, const int _M, const int _N,
                   const int _inner_x1, const int _inner_x2, const int _inner_y1, const int _inner_y2,
                   const int _bndN, const int _bnd_x1, const int _bnd_y1) 
    : v(thrust::raw_pointer_cast(_v.data())), f(thrust::raw_pointer_cast(_f.data())),
      hx(_hx), hy(_hy), M(_M), N(_N), 
      inner_x1(_inner_x1), inner_x2(_inner_x2), inner_y1(_inner_y1), inner_y2(_inner_y2),
      bndN(_bndN), bnd_x1(_bnd_x1), bnd_y1(_bnd_y1) {}

    __host__ __device__
        double operator()(const double& vij, const int& k) const
        {
            // MeshVec index, see MeshVec class
            //int k = (j - bnd_y1) + (i - bnd_x1)*N;
            const int i = (k / bndN) + bnd_x1;
            const int j = (k % bndN) + bnd_y1;

            double cAv; // Av value

            const double hx2 = hx*hx;
            const double hy2 = hy*hy;

            // Compute Av value
            if (i >= inner_x1 && i <= inner_x2 && j >= inner_y1 && j <= inner_y2)
            {
                // Inner area
                
                const double vipj  = v[k + bndN]; // v(i+1, j)
                const double vijp  = v[k + 1]; // v(i, j+1)
                const double vimj  = v[k - bndN]; // v(i-1, j)
                const double vijm  = v[k - 1]; // v(i, j-1)

                cAv = -1.0/hx2*(vipj - 2*vij + vimj) 
                      -1.0/hy2*(vijp - 2*vij + vijm);
            }
            else if (i == 0 && j == 0)
            {
                // Edge point: (X1, Y1)
                
                const double vipj  = v[k + bndN]; // v(i+1, j)
                const double vijp  = v[k + 1]; // v(i, j+1)

                cAv = -2.0/hx2*(vipj - vij) 
                      -2.0/hy2*(vijp - vij);
            }
            else if (j == 0)
            {
                // Bottom boundary
                
                if (i >= inner_x1 && i <= inner_x2)
                {
                    const double vipj  = v[k + bndN]; // v(i+1, j)
                    const double vijp  = v[k + 1]; // v(i, j+1)
                    const double vimj  = v[k - bndN]; // v(i-1, j)

                    cAv = -2.0/hy2*(vijp - vij) 
                          -1.0/hx2*(vipj - 2*vij + vimj);
                }
                else if (i == M-1)
                {
                    const double vijp  = v[k + 1]; // v(i, j+1)
                    const double vimj  = v[k - bndN]; // v(i-1, j)

                    cAv = -2.0/hy2*(vijp - vij) 
                          -1.0/hx2*( -2*vij + vimj);
                }
            }
            else if (i == 0)
            {
                // Left boundary

                if (j >= inner_y1 && j <= inner_y2)
                {
                    const double vipj  = v[k + bndN]; // v(i+1, j)
                    const double vijp  = v[k + 1]; // v(i, j+1)
                    const double vijm  = v[k - 1]; // v(i, j-1)

                    cAv = -2.0/hx2*(vipj - vij)
                          -1.0/hy2*(vijp - 2*vij + vijm);
                }
                else if (j == N-1)
                {
                    const double vipj  = v[k + bndN]; // v(i+1, j)
                    const double vijm  = v[k - 1]; // v(i, j-1)

                    cAv = -2.0/hx2*(vipj - vij)
                          -1.0/hy2*( -2*vij + vijm);
                }
            }
            else if (i == M-1 && j >= inner_y1 && j <= inner_y2)
            {
                // Right pre-boundary

                const double vijp  = v[k + 1]; // v(i, j+1)
                const double vimj  = v[k - bndN]; // v(i-1, j)
                const double vijm  = v[k - 1]; // v(i, j-1)

                cAv = -1.0/hx2*( -2*vij + vimj) 
                      -1.0/hy2*(vijp  - 2*vij + vijm);
            }
            else if (j == N-1 && i >= inner_x1 && i <= inner_x2)
            {
                // Top pre-boundary

                const double vipj  = v[k + bndN]; // v(i+1, j)
                const double vimj  = v[k - bndN]; // v(i-1, j)
                const double vijm  = v[k - 1]; // v(i, j-1)

                cAv = -1.0/hx2*(vipj  - 2*vij + vimj) 
                      -1.0/hy2*( -2*vij + vijm);
            
            }
            else if (i == M-1 && j == N-1)
            {
                // Edge point: (X2, Y2)
                
                const double vimj  = v[k - bndN]; // v(i-1, j)
                const double vijm  = v[k - 1]; // v(i, j-1)

                cAv = -1.0/hx2*( -2*vij + vimj) 
                      -1.0/hy2*( -2*vij + vijm);
            }
            else
            {
                cAv = f[k];
            }

            // Update device vector
            return cAv - f[k];
        }
};
/****************************************************************************************************************/

void LaplaceOperator::matvec_device(const MeshVec &v, const MeshVec &f, MeshVec &Av) const
{
    assert(mtls == v.mpitools()); 
    assert(mtls == f.mpitools()); 
    assert(mtls == Av.mpitools()); 

#ifdef _CUDA_KERNEL_MATVEC_
    // CUDA Kernel matvec
    const double *raw_ptr_v = thrust::raw_pointer_cast(v.get_device_vec().data());
    const double *raw_ptr_f = thrust::raw_pointer_cast(f.get_device_vec().data());
    double *raw_ptr_Av = thrust::raw_pointer_cast(Av.get_device_vec().data());

    dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 DimGrid(iDivUp(mtls.bndM(), BLOCK_SIZE), iDivUp(mtls.bndN(), BLOCK_SIZE));

    matvec_kernel<<<DimGrid, DimBlock>>>(raw_ptr_v, raw_ptr_f, raw_ptr_Av, 
                                         hx, hy, M, N,
                                         inner_x1, inner_x2, inner_y1, inner_y2,
                                         mtls.bndN(), mtls.bndx1(), mtls.bndy1());
#else
    // Thrust transform matvec
    struct matvec_functor mvfunc(v.get_device_vec(), f.get_device_vec(), 
                                 hx, hy, M, N,
                                 inner_x1, inner_x2, inner_y1, inner_y2,
                                 mtls.bndN(), mtls.bndx1(), mtls.bndy1());

    thrust::transform(v.get_device_vec().begin(), v.get_device_vec().end(), 
                      thrust::make_counting_iterator(0), 
                      Av.get_device_vec().begin(),
                      mvfunc);
#endif
    //cudaDeviceSynchronize();
}

