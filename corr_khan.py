import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 30.0

arr = np.array([[0.123, 0.145, 0.112, 0.113],
                [0.145, 0.197, 0.163, 0.193],
                [0.112, 0.163, 0.211, 0.058],
                [0.113, 0.193, 0.058, 0.186]])

plt.plot(arr.T)
plt.ylim((-0.1, 0.25))
plt.savefig('corr_khan.png')
plt.show()
