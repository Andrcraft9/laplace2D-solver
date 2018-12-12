HOST_COMP = mpic++
GPU_COMP = nvcc
ARCH = sm_60

#
# DEFINES:
# _CUDA_KERNEL_MATVEC_   : use cuda kernel for matvec operation, otherwise use thrust transform for matvec
# _CUDA_AWARE_MPI_SYNC_  : use  CUDA-Aware MPI for communications (sync call)
#
# You can use any of this features by adding option: -D <define>
FLAGS = -O3

ultra:
	$(GPU_COMP) $(FLAGS) -arch=$(ARCH) -o lapsol --compiler-bindir $(HOST_COMP) main.cpp mpitools.cpp mesh.cpp mesh.cu laplace.cpp laplace.cu solver.cpp

ultra_CAMS:
	$(GPU_COMP) $(FLAGS) -D _CUDA_AWARE_MPI_SYNC_ -arch=$(ARCH) -o lapsol --compiler-bindir $(HOST_COMP) main.cpp mpitools.cpp mesh.cpp mesh.cu laplace.cpp laplace.cu solver.cpp

ultra_CKM:
	$(GPU_COMP) $(FLAGS) -D _CUDA_KERNEL_MATVEC_ -arch=$(ARCH) -o lapsol --compiler-bindir $(HOST_COMP) main.cpp mpitools.cpp mesh.cpp mesh.cu laplace.cpp laplace.cu solver.cpp
clean:
	rm *.o
	rm lapsol