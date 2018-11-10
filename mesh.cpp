#include "mesh.hpp"

std::ostream& operator<< (std::ostream& os, const MeshVec& v) 
{
    os << v.M << " " << v.N << std::endl;
    for(int i = 0; i < v.get_M(); ++i)
    {
        for(int j = 0; j < v.get_N(); ++j)
        {
            os << "i, j: " << i << " , " << j << " v(i, j) = " << v(i, j) << std::endl;
        }
    }

    return os;
}