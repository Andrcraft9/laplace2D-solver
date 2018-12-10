HOST_COMP = mpic++
GPU_COMP = nvcc
ARCH = sm_30
FLAGS = -O3

#all:
#	$(CC) $(CFLAGS) -o lapsol main.cpp mpitools.cpp mesh.cpp laplace.cpp solver.cpp

#separate:
#	nvcc -c --compiler-bindir mpic++ mesh.cu
#	$(CC) -c main.cpp mesh.cpp mpitools.cpp
#	nvcc -o lapsol main.o mesh.o mpitools.o

ultra:
	$(GPU_COMP) $(FLAGS) -arch=$(ARCH) -o lapsol --compiler-bindir $(HOST_COMP) main.cpp mpitools.cpp mesh.cpp mesh.cu laplace.cpp laplace.cu solver.cpp

clean:
	rm *.o
	rm lapsol