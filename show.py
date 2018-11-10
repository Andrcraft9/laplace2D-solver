import numpy as np
import matplotlib.pyplot as plt
import sys

f = open(sys.argv[1], "r")
line = f.readline()
M = int(line.split()[0])
N = int(line.split()[1])
X = np.zeros((M, N))
for line in f:
    i = int(line.split()[2])
    j = int(line.split()[4])
    X[i, j] = float(line.split()[-1])
    
plt.matshow(X)
plt.colorbar()
plt.show()

