#include "solver.hpp"

int MRM::solve(const LaplaceOperator& L, const MeshVec& RHS, MeshVec& X) const
{
    int k = 0;
    double err = 1.0;
    MPITools mpitools(X.mpitools());

    MeshVec r(mpitools);
    MeshVec Ar(mpitools);
    
    // Residual
    L.matvec(X, r); r -= RHS; r.sync();
    // Ar
    L.matvec(r, Ar);
    err = L.norm_mesh(r);

    while (err > eps && k <= maxIters)
    {    
        // X at k+1 step 
        double tau = L.dot_mesh(Ar, r) / pow(L.norm_mesh(Ar), 2);
        X.axpy(-tau, r); X.sync();
        // Residual at k+1 step
        L.matvec(X, r); r -= RHS; r.sync();
        // Ar at k+1 step
        L.matvec(r, Ar);
        err = L.norm_mesh(r);
        
        ++k;
    }

    if (L.mpitools().rank() == 0) std::cout << "MRM: iters: " << k << " residual: " << err << std::endl;
    if (k >= maxIters && L.mpitools().rank() == 0) 
        std::cout << "MRM: Warning! Max Iterations in MRM solver!" << std::endl;
    
    return k;
}

int MRM::profile_solve(const LaplaceOperator& L, const MeshVec& RHS, MeshVec& X) const
{
    int k = 0;
    double err = 1.0;
    // Times
    double time_loc = 0.0;
    double time_sync = 0.0, time_sync_max = 0.0;
    double time_matvec = 0.0, time_matvec_max = 0.0;
    double time_norm = 0.0, time_norm_max = 0.0;
    double time_axpy = 0.0, time_axpy_max = 0.0;

    MPITools mpitools(X.mpitools());

    MeshVec r(mpitools);
    MeshVec Ar(mpitools);
    
    // Residual
    time_loc = mpitools.get_time();
    L.matvec(X, r); 
    time_matvec = time_matvec + (mpitools.get_time() - time_loc);

    time_loc = mpitools.get_time();
    r -= RHS; 
    time_axpy = time_axpy + (mpitools.get_time() - time_loc);
    
    // Sync
    time_loc = mpitools.get_time();
    r.sync();
    time_sync = time_sync + (mpitools.get_time() - time_loc);
    
    // Ar
    time_loc = mpitools.get_time();
    L.matvec(r, Ar);
    time_matvec = time_matvec + (mpitools.get_time() - time_loc);

    time_loc = mpitools.get_time();
    err = L.norm_mesh(r);
    time_norm = time_norm + (mpitools.get_time() - time_loc);

    while (err > eps && k <= maxIters)
    {   
        // X at k+1 step 
        time_loc = mpitools.get_time();
        double tau = L.dot_mesh(Ar, r) / pow(L.norm_mesh(Ar), 2);
        time_norm = time_norm + (mpitools.get_time() - time_loc);

        time_loc = mpitools.get_time();
        X.axpy(-tau, r); 
        time_axpy = time_axpy + (mpitools.get_time() - time_loc);
        
        // Sync
        time_loc = mpitools.get_time();
        X.sync();
        time_sync = time_sync + (mpitools.get_time() - time_loc);
        
        // Residual at k+1 step
        time_loc = mpitools.get_time();
        L.matvec(X, r); 
        time_matvec = time_matvec + (mpitools.get_time() - time_loc);
        
        time_loc = mpitools.get_time();
        r -= RHS; 
        time_axpy = time_axpy + (mpitools.get_time() - time_loc);
        
        // Sync
        time_loc = mpitools.get_time();
        r.sync();
        time_sync = time_sync + (mpitools.get_time() - time_loc);
        
        // Ar at k+1 step
        time_loc = mpitools.get_time();
        L.matvec(r, Ar);
        time_matvec = time_matvec + (mpitools.get_time() - time_loc);

        time_loc = mpitools.get_time();
        err = L.norm_mesh(r);
        time_norm = time_norm + (mpitools.get_time() - time_loc);
        
        ++k;
    }

    time_sync_max = mpitools.sync_time(time_sync);
    time_matvec_max = mpitools.sync_time(time_matvec);
    time_norm_max = mpitools.sync_time(time_norm);
    time_axpy_max = mpitools.sync_time(time_axpy);

    if (L.mpitools().rank() == 0) 
    {
        std::cout << "MRM: iters: " << k << " residual: " << err << std::endl;
        std::cout << "MRM: time sync: " << time_sync << " time sync (max): " << time_sync_max << std::endl;
        std::cout << "MRM: time matvec: " << time_matvec << " time matvec (max): " << time_matvec_max << std::endl;
        std::cout << "MRM: time norm: " << time_norm << " time norm (max): " << time_norm_max << std::endl;
        std::cout << "MRM: time axpy: " << time_axpy << " time axpy (max): " << time_axpy_max << std::endl;
    }
    if (k >= maxIters && L.mpitools().rank() == 0) 
        std::cout << "MRM: Warning! Max Iterations in MRM solver!" << std::endl;
    
    return k;
}