CC = mpic++
CFLAGS = -O3

all:
	$(CC) $(CFLAGS) -o lapsol main.cpp mpitools.cpp mesh.cpp laplace.cpp solver.cpp

#separate:
#	nvcc -c --compiler-bindir mpic++ mesh.cu
#	$(CC) -c main.cpp mesh.cpp mpitools.cpp
#	nvcc -o lapsol main.o mesh.o mpitools.o

ultra:
	nvcc -o lapsol --compiler-bindir mpic++ main.cpp mpitools.cpp mesh.cpp mesh.cu laplace.cpp laplace.cu solver.cpp

clean:
	rm *.o
	rm lapsol