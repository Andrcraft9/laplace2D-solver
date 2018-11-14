#include "mpitools.hpp"

int MPITools::init(int *argc, char ***argv, int m, int n)
{
    assert(!initialized_); 

    M_ = m;
    N_ = n;
    assert(M_ > 0 && N_ > 0);

    int p_size[2];
    int p_coord[2];
    p_size[0] = 0; p_size[1] = 0;
    p_coord[0] = 0; p_coord[1] = 0;

    // MPI
    MPI_Init(argc, argv);
    MPI_Comm_size(MPI_COMM_WORLD, &procs_);
    MPI_Dims_create(procs_, 2, p_size);
    int p_period[2] = {0 ,0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, p_size, p_period, 0, &comm_);
    MPI_Comm_size(comm_, &procs_);
    MPI_Comm_rank(comm_, &rank_);
    MPI_Cart_coords(comm_, rank_, 2, p_coord);

    // OpenMP
    #pragma omp parallel
    {
        threads_ = omp_get_num_threads();
        //num_thread = omp_get_thread_num()
        //if (num_thread .eq. 0) print *, "OMP Threads: ", count_threads
    }

    npx_ = p_size[0];
    npy_ = p_size[1];
    px_ = p_coord[0];
    py_ = p_coord[1];

    // Uniform Domain Decomposition
    assert(M_ % npx_ == 0 && N_ % npy_ == 0);
    locM_ = M_ / npx_;
    locN_ = N_ / npy_;
    locx1_ = px_ * locM_;
    locx2_ = locx1_ + locM_ - 1;
    locy1_ = py_ * locN_;
    locy2_ = locy1_ + locN_ - 1;

    bndx1_ = std::max(0, locx1_ - 1);
    bndx2_ = std::min(M_ - 1, locx2_ + 1);
    bndM_ = bndx2_ - bndx1_ + 1;
    bndy1_ = std::max(0, locy1_ - 1);
    bndy2_ = std::min(N_ - 1, locy2_ + 1);
    bndN_ = bndy2_ - bndy1_ + 1;

    // Check: has proc boundaries?
    RB_ = false; LB_ = false; 
    TB_ = false; BB_ = false;
    if (px_ == 0) LB_ = true;
    if (px_ == npx_ - 1) RB_ = true;
    if (py_ == 0) BB_ = true;
    if (py_ == npy_ - 1) TB_ = true;
    
    // Info 
    if (rank_ == 0)
    {
        std::cout << "MPI/OpenMP init is ok. procs = " << procs_ << " threads = " << threads_ << std::endl;
        std::cout << "npx = " << npx_ << " npy = " << npy_ << " locM = " << locM_ << " locN = " << locN_ << std::endl;
    }
    /*
    std::cout << "rank = " << rank_ << 
        " locx1 = " << locx1_ << " locx2 = " << locx2_ << " locy1 = " << locy1_ << " locy2 = " << locy2_ << 
        " bndM = " << bndM_ << " bndN = " << bndN_ << 
        " LB, RB, TB, BB: " << LB_ << RB_ << TB_ << BB_ << std::endl;
    */
    
    initialized_ = true;

    return  0;
}

bool MPITools::operator==(const MPITools &m) const
{
    return comm_ == m.comm_ && M_ == m.M_ && N_ == m.N_ && locM_ == m.locM_ && locN_ == m.locN_;
}
