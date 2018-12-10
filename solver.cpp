#include "solver.hpp"

int MRM::solve_profile(const LaplaceOperator& L, const MeshVec& RHS, MeshVec& X) const
{
    int k = 0;
    double err = 1.0;
    double r_r, Ar_r, Ar_Ar;
    // Times
    double time_loc = 0.0;
    double time_sync = 0.0, time_sync_max = 0.0;
    double time_matvec = 0.0, time_matvec_max = 0.0;
    double time_dot = 0.0, time_dot_max = 0.0;
    double time_axpy = 0.0, time_axpy_max = 0.0;

    MPITools mpitools(X.mpitools());

    MeshVec r(mpitools, 0.0);
    r.load_gpu();
    MeshVec Ar(mpitools, 0.0);
    Ar.load_gpu();
    MeshVec zero(mpitools, 0.0);
    zero.load_gpu();
    
    // r = Ax - RHS
    time_loc = mpitools.get_time();
    L.matvec_device(X, RHS, r); 
    time_matvec = time_matvec + (mpitools.get_time() - time_loc);
    //r.sync();
    // Ar
    time_loc = mpitools.get_time();
    L.matvec_device(r, zero, Ar);
    time_matvec = time_matvec + (mpitools.get_time() - time_loc);
    
    // Compute: (r, r), (Ar, r), (Ar, Ar)
    time_loc = mpitools.get_time();
    std::vector<double> dots = L.dot_general_mesh_device(r, Ar);
    time_dot = time_dot + (mpitools.get_time() - time_loc);
    r_r = dots[0];
    Ar_r = dots[1];
    Ar_Ar = dots[2];
    // Compute error and tau
    err = sqrt(r_r);
    double tau = Ar_r / Ar_Ar;

    while ( err > eps && k <= maxIters )
    {   
        // update X = X - tau*r 
        time_loc = mpitools.get_time();
        X.axpy_device(-tau, r); 
        time_axpy = time_axpy + (mpitools.get_time() - time_loc);
        //X.sync();

        // r = Ax - RHS
        time_loc = mpitools.get_time();
        L.matvec_device(X, RHS, r); 
        time_matvec = time_matvec + (mpitools.get_time() - time_loc);
        //r.sync();
        // Ar
        time_loc = mpitools.get_time();
        L.matvec_device(r, zero, Ar);
        time_matvec = time_matvec + (mpitools.get_time() - time_loc);

        // Compute: (r, r), (Ar, r), (Ar, Ar)
        time_loc = mpitools.get_time();
        dots = L.dot_general_mesh_device(r, Ar);
        time_dot = time_dot + (mpitools.get_time() - time_loc);
        r_r = dots[0];
        Ar_r = dots[1];
        Ar_Ar = dots[2];
        // Compute error and tau
        err = sqrt(r_r);
        tau = Ar_r / Ar_Ar;

        ++k;
    }

    time_sync_max = mpitools.sync_time(time_sync);
    time_matvec_max = mpitools.sync_time(time_matvec);
    time_dot_max = mpitools.sync_time(time_dot);
    time_axpy_max = mpitools.sync_time(time_axpy);

    if (L.mpitools().rank() == 0) 
    {
        std::cout << "MRM: iters: " << k << " residual: " << err << std::endl;
        std::cout << "MRM: time sync: " << time_sync << " time sync (max): " << time_sync_max << std::endl;
        std::cout << "MRM: time matvec: " << time_matvec << " time matvec (max): " << time_matvec_max << std::endl;
        std::cout << "MRM: time dot: " << time_dot << " time dot (max): " << time_dot_max << std::endl;
        std::cout << "MRM: time axpy: " << time_axpy << " time axpy (max): " << time_axpy_max << std::endl;
    }

    if (L.mpitools().rank() == 0) std::cout << "MRM: iters: " << k << " resudial: " << err << std::endl;
    if (k >= maxIters && L.mpitools().rank() == 0) 
        std::cout << "MRM: Warning! Max Iterations in MRM solver!" << std::endl;
    
    return k;
}
