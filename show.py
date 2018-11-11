import numpy as np
import matplotlib.pyplot as plt
import sys
import os

M = int(sys.argv[1])
N = int(sys.argv[2])
X = np.zeros((M, N))

for file in os.listdir():
    if file.startswith("results_"):
        f = open(file, "r")
        
        line = f.readline()
        #M = int(line.split()[0])
        #N = int(line.split()[1])

        for line in f:
            i = int(line.split()[2])
            j = int(line.split()[4])
            X[i, j] = float(line.split()[-1])


plt.matshow(X)
plt.colorbar()
plt.show()