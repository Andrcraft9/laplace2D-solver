#include "mesh.hpp"

MeshVec& MeshVec::axpy(double a, const MeshVec& x)
{
    assert(mtls == x.mtls);

    for(int i = mtls.locx1(); i <= mtls.locx2(); ++i)
        for(int j = mtls.locy1(); j <= mtls.locy2(); ++j)
            (*this)(i, j) = (*this)(i, j) + a*x(i, j);
            
    return *this;
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
