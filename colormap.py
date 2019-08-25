import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROIs = ['1st01ms', '1st10ms', '2nd10ms', '3rd10ms', '4th10ms', '5th10ms', '6th10ms', '7th10ms', '8th10ms', '9th10ms', '10th10ms'] 

pop_cell_nums = np.array([20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948])

mean_shift = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
std_shift = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
X, Y = np.meshgrid(mean_shift, std_shift)
Z = [np.zeros(np.shape(X)) for i in range(8)]

mpl.rcParams['font.size'] = 12.0
labels = ['L23 E', 'L23 I', 'L4 E', 'L4 I', 'L5 E', 'L5 I', 'L6 E', 'L6 I']

for ROI in ROIs:
    fig, axs = plt.subplots(4, 2, figsize=(6, 12), sharex=True, sharey=True, constrained_layout=True)
    axs = axs.ravel()
    fig.suptitle(ROI + ' avg neuron firing rate\n', fontsize=16)
    for i in range(len(std_shift)):
        for j in range(len(mean_shift)):
            target = 'mean_shift_' + str(mean_shift[j]) + '_std_shift_' + str(std_shift[i])
            data = np.loadtxt(target + '/data/' + target + '_fr_' + ROI + '.dat')   
            data = np.divide(data, pop_cell_nums)
            if ROI != '1st01ms':
                data /= 10
            for k in range(len(data)):
                if k < len(Z):
                    Z[k][i][j] = data[k]

    levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(len(Z)):
        cs = axs[i].contourf(X, Y, Z[i], levels)
        axs[i].set_title(labels[i])
        if i == 6:        
            axs[i].set_xlabel('mean shift (mV)', fontsize=14)
            axs[i].set_ylabel('std shift (mV)', fontsize=14)
        elif i == len(Z) -1:
            clb = plt.colorbar(cs)
            clb.set_label('spikes/ms', rotation=270)
            #clb.ax.set_title('spikes/ms')

    plt.savefig(ROI + '.png')
    plt.close()
