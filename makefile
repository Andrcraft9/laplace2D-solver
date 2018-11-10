CC = g++
CFLAGS = -O3

all:
	$(CC) $(CFLAGS) -o lapsol main.cpp mesh.cpp laplace.cpp

