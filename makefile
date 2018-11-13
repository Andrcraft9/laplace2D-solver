CC = mpic++
CFLAGS = -O3 -fopenmp

all:
	$(CC) $(CFLAGS) -o lapsol main.cpp mpitools.cpp mesh.cpp laplace.cpp solver.cpp


