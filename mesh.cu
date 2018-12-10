#include "mesh.hpp"

#include <thrust/transform.h>
#include <thrust/functional.h>

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

void MeshVec::load_halo_gpu()
{

}

void MeshVec::unload_halo_gpu()
{
    
}
