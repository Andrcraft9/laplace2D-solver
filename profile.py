import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
objects = ('Total time', 'Sync time', 'MatVec time', 'Norm time', 'AXPY time')
y_pos = np.arange(len(objects))
performance_mpi = [444.983, 116.511, 167.804, 118.371, 81.7204]
performance = [152.44, 0.0, 88.2069, 26.6843, 37.0875]
performance = [1000.0*performance[i]/performance_mpi[i] for i in range(len(performance))]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
#plt.ylabel('Time, sec')
plt.ylabel('SpeedUp')
plt.title('Profiling')
 
plt.show()