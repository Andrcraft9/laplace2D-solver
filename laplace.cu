#include "laplace.hpp"

#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#define BLOCK_SIZE 16

typedef thrust::tuple<double, double, double> Double3;

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

struct square
{
    __host__ __device__
        double operator()(const double& x) const 
        { 
            return x * x;
        }
};


std::vector<double> LaplaceOperator::dot_general_mesh_device(const MeshVec& r, const MeshVec& Ar) const
{
    assert(mtls == r.mpitools()); 
    assert(mtls == Ar.mpitools()); 

    Double3 init(0.0, 0.0, 0.0);

    struct dot_general_plus binary_plus;
    struct dot_general_mult binary_mult;

    Double3 dot = thrust::inner_product(thrust::device,
                                        r.get_device_vec().begin(), r.get_device_vec().end(), 
                                        Ar.get_device_vec().begin(), 
                                        init, 
                                        binary_plus, 
                                        binary_mult);
                                        

    //cudaDeviceSynchronize();
    std::vector<double> dot_out(3);
    dot_out[0] = hx * hy * thrust::get<0>(dot); // (r, r)
    dot_out[1] = hx * hy * thrust::get<1>(dot); // (Ar, r)
    dot_out[2] = hx * hy * thrust::get<2>(dot); // (Ar, Ar)


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

/*
    std::vector<double> dot_out(3);

    square               unary_op;
    thrust::plus<double> binary_op;
    double init = 0;

    // compute norm
    dot_out[0] = hx * hy * std::sqrt( thrust::transform_reduce(r.get_device_vec().begin(), r.get_device_vec().end(), unary_op, init, binary_op) );
    dot_out[1] = hx * hy * std::sqrt( thrust::transform_reduce(Ar.get_device_vec().begin(), Ar.get_device_vec().end(), unary_op, init, binary_op) );
    dot_out[2] = hx * hy * std::sqrt( thrust::transform_reduce(Ar.get_device_vec().begin(), Ar.get_device_vec().end(), unary_op, init, binary_op) );

    //double dot = 0.0, dot_out = 0.0;
    //MPI_Allreduce(&dot, &dot_out, 1, MPI_DOUBLE, MPI_SUM, mtls.comm());
*/

    return dot_out;
}

__host__ __device__ int iDivUp(int a, int b)
{ 
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__global__ void matvec_kernel(const double *v, const double *f, double *Av, 
                              const double hx, const double hy,
                              const int M, const int N,
                              const int inner_x1, const int inner_x2, const int inner_y1, const int inner_y2,
                              const int bnd_x1, const int bnd_y1)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    // MeshVec index, see MeshVec class
    int k = (j - bnd_y1) + (i - bnd_x1)*N;

    double cAv; // Av value

    double hx2 = hx*hx;
    double hy2 = hy*hy;
    double vij   = v[k];     // v(i, j)
    //double vipj  = v[k + N]; // v(i+1, j)
    //double vijp  = v[k + 1]; // v(i, j+1)
    //double vimj  = v[k - N]; // v(i-1, j)
    //double vijm  = v[k - 1]; // v(i, j-1)

    // Compute Av value
    if (i >= inner_x1 && i <= inner_x2 && j >= inner_y1 && j <= inner_y2)
    {
        // Inner area
        
        double vipj  = v[k + N]; // v(i+1, j)
        double vijp  = v[k + 1]; // v(i, j+1)
        double vimj  = v[k - N]; // v(i-1, j)
        double vijm  = v[k - 1]; // v(i, j-1)

        cAv = -1.0/hx2*(vipj - 2*vij + vimj) 
              -1.0/hy2*(vijp - 2*vij + vijm);
    }
    else if (i == 0 && j == 0)
    {
        // Edge point: (X1, Y1)
        
        double vipj  = v[k + N]; // v(i+1, j)
        double vijp  = v[k + 1]; // v(i, j+1)

        cAv = -2.0/hx2*(vipj - vij) 
              -2.0/hy2*(vijp - vij);
    }
    else if (j == 0)
    {
        // Bottom boundary
        
        double vipj  = v[k + N]; // v(i+1, j)
        double vijp  = v[k + 1]; // v(i, j+1)
        double vimj  = v[k - N]; // v(i-1, j)

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
        
        double vipj  = v[k + N]; // v(i+1, j)
        double vijp  = v[k + 1]; // v(i, j+1)
        double vijm  = v[k - 1]; // v(i, j-1)

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

        double vijp  = v[k + 1]; // v(i, j+1)
        double vimj  = v[k - N]; // v(i-1, j)
        double vijm  = v[k - 1]; // v(i, j-1)

        cAv = -1.0/hx2*( -2*vij + vimj) 
              -1.0/hy2*(vijp  - 2*vij + vijm);
    }
    else if (j == N-1 && i >= inner_x1 && i <= inner_x2)
    {
        // Top pre-boundary

        double vipj  = v[k + N]; // v(i+1, j)
        double vimj  = v[k - N]; // v(i-1, j)
        double vijm  = v[k - 1]; // v(i, j-1)

        cAv = -1.0/hx2*(vipj  - 2*vij + vimj) 
              -1.0/hy2*( -2*vij + vijm);
    
    }
    else if (i == M-1 && j == N-1)
    {
        // Edge point: (X2, Y2)
        
        double vimj  = v[k - N]; // v(i-1, j)
        double vijm  = v[k - 1]; // v(i, j-1)

        cAv = -1.0/hx2*( -2*vij + vimj) 
              -1.0/hy2*( -2*vij + vijm);
    }

    // Update device vector
    Av[k] = cAv - f[k];
}

void LaplaceOperator::matvec_device(const MeshVec &v, const MeshVec &f, MeshVec &Av) const
{
    assert(mtls == v.mpitools()); 
    assert(mtls == f.mpitools()); 
    assert(mtls == Av.mpitools()); 

    //return;

    const double *raw_ptr_v = thrust::raw_pointer_cast(v.get_device_vec().data());
    const double *raw_ptr_f = thrust::raw_pointer_cast(f.get_device_vec().data());
    double *raw_ptr_Av = thrust::raw_pointer_cast(Av.get_device_vec().data());

    //return;

    dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 DimGrid(iDivUp(M, BLOCK_SIZE), iDivUp(N, BLOCK_SIZE));

    matvec_kernel<<<DimGrid, DimBlock>>>(raw_ptr_v, raw_ptr_f, raw_ptr_Av, 
                                         hx, hy,
                                         M, N,
                                         inner_x1, inner_x2, inner_y1, inner_y2,
                                         mtls.bndx1(), mtls.bndy1());

    //cudaDeviceSynchronize();
}