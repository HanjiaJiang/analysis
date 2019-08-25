import numpy as np
import os
import matplotlib.pyplot as plt
import microcircuit_tools as tools
from datetime import datetime
from scipy import stats
from time import time
import matplotlib

matplotlib.rcParams['font.size'] = 30.0

populations = \
    ['L2/3 E', 'L2/3 PV', 'L2/3 SOM', 'L2/3 VIP',
     'L4 E', 'L4 PV', 'L4 SOM',
     'L5 E', 'L5 PV', 'L5 SOM',
     'L6 E', 'L6 PV', 'L6 SOM']

colors = ['b', 'r', 'orange', 'g',
          'b', 'r', 'orange',
          'b', 'r', 'orange',
          'b', 'r', 'orange']

layer_label = ['L2/3', 'L4', 'L5', 'L6']

subtype_label = ['E', 'PV+', 'SOM+', 'VIP+']

cwd = os.getcwd()


fig = plt.figure(figsize=(7.5, 16))
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

for i in range(4):
    plt.plot([0, 1, 2], [-1, -1, -1], color=colors[i],
                 label='{} \'horizontal\' clusters'.format(subtype_label[int(i)]), linewidth=4)
    plt.plot([0, 1, 2], [-1, -1, -1], color=colors[i],
                 label='{} \'vertical\' clusters'.format(subtype_label[int(i)]), linewidth=4, ls='--')

legend = plt.legend(bbox_to_anchor=(-0.1, -0.24, 0.1, 1.0),
           ncol=1, mode="expand", fontsize=30)
legend.get_frame().set_edgecolor('w')
plt.xlim((0.0, 10.0))
plt.ylim((0.0, 10.0))
plt.savefig(os.path.join(cwd, 'test.png'), dpi=300)
plt.show()