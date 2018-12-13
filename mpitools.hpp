#include <iostream>
#include <string>
#include <cassert>
#include <ctime>
#include <mpi.h>

#ifndef MPITOOLS_H
#define MPITOOLS_H

class MPITools
{
private:
    bool initialized_;

    MPI_Comm comm_; // Main Cartesian Communicator
    int procs_; // Total procs
    int threads_; // Total threads per current proc
    int rank_; // Current proc
    int npx_, npy_; // Procs grid
    int px_, py_; // Procs coords
    int M_, N_; // Total size

    int gpus_; 
    int rank_gpu_;
    
    // Local area
    int locM_, locN_;
    int locx1_, locx2_;
    int locy1_, locy2_;

    // Local area with halo poitns
    int bndM_, bndN_;
    int bndx1_, bndx2_;
    int bndy1_, bndy2_;

    // Proc has boundaries? right, left, top, bottom
    bool RB_, LB_, TB_, BB_; 

public:
    MPITools() : initialized_(false) {}
    
    int init(int *argc, char ***argv, int m, int n);

    bool operator==(const MPITools &m) const;

    // Getters
    bool initialized() const { return initialized_; }

    int rank() const { return rank_; }
    MPI_Comm comm() const { return comm_; }
    int procs() const { return procs_; }
    int threads() const { return threads_; }
    
    int npx() const { return npx_; } 
    int npy() const { return npy_; }
    int px() const { return px_; } 
    int py() const { return py_; }
    
    int M() const { return M_; }
    int N() const { return N_; }

    int locM() const { return locM_; }
    int locN() const { return locN_; }
    int locx1() const { return locx1_; }
    int locx2() const { return locx2_; }
    int locy1() const { return locy1_; }
    int locy2() const { return locy2_; }

    int bndM() const { return bndM_; }
    int bndN() const { return bndN_; }
    int bndx1() const { return bndx1_; }
    int bndx2() const { return bndx2_; }
    int bndy1() const { return bndy1_; }
    int bndy2() const { return bndy2_; }

    bool RB() const { return RB_; }
    bool LB() const { return LB_; }
    bool TB() const { return TB_; }
    bool BB() const { return BB_; }

    double get_time();

    double sync_time(double t)
    {
        double tout;
        MPI_Allreduce(&t, &tout, 1, MPI_DOUBLE, MPI_MAX, comm_);
        return tout;
    }

    int finalize() 
    { 
        assert(initialized_); 
        initialized_ = false;
        return MPI_Finalize(); 
    }

};

#endif
