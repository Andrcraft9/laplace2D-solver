import numpy as np
import matplotlib.pyplot as plt
 
# data to plot
n_groups = 5
means_seq = (85, 62, 54, 20)
means_mpi = (444.983, 116.511, 167.804, 118.371, 81.7204)
 
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, means_seq, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Sequential')
 
rects2 = plt.bar(index + bar_width, means_mpi, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Parallel, MPI')
 
#plt.xlabel('')
plt.ylabel('Time, sec')
plt.title('Profiling')
plt.xticks(index + bar_width, ('Total time', 'Sync time', 'MatVec time', 'Norm time', 'AXPY time'))
plt.legend()
 
plt.tight_layout()
plt.show()