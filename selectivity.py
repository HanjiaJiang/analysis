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
plot_length = 80.0
y_boundary = [-1.25, 1.25]
stimulus = 10

def count_spikes(path, name, begin, end, t_stim_list, len_window):
    # t_stim_list: list of stimulation time points
    # len_window: length of each window to count spikes
    t_f0 = time()
    data_all, gids = tools.load_spike_times(path, name, begin, end)
    counts_all = [] # pop x window x trial x id
    t_f1 = time()
    for i in list(range(len(data_all))):
        # data = data_all[i]
        data_ids = data_all[i][:, 0]
        data_times = data_all[i][:, 1]
        if len(data_ids) > 0:
            counts_pop = [] # window x trial x id
            for t_shift in np.arange(0.0, t_stim_list[1] - t_stim_list[0],
                                     len_window):
                t_f00 = time()
                # ids_list: trial x ids (each spikes)
                ids_list = [
                    data_ids[(data_times[:] > t + t_shift) & (
                            data_times[:] < t + t_shift + len_window)]
                    for t in t_stim_list
                ]
                # ids_list = [
                #     data[(data[:, 1] > t + t_shift) & (data[:, 1] < t + t_shift + len_window), 0] for t in t_stim_list
                # ]
                # print(ids_list)
                t_f01 = time()
                counts = [] # trial x ids (firing counts)
                for k, ids in enumerate(ids_list):
                    ids_subtracted = np.subtract(ids, gids[i][0])
                    try:
                        counts.append(np.bincount(ids_subtracted.astype(int), minlength=gids[i][1]-gids[i][0]+1))
                    except ValueError:
                        print(gids[i][0])
                    else:
                        pass
                counts_pop.append(counts)
                t_f02 = time()
                # if i == 0:
                #     print('dt_f01 = {:.4f}, dt_f12 = {:.4f}'.
                #           format(t_f01 - t_f00, t_f02 - t_f01))
            counts_all.append(counts_pop)
        else:
            counts_all.append([])
    t_f2 = time()
    print('dt_f01 = {:.3f}, dt_f12 = {:.3f}'.format(t_f1 - t_f0, t_f2 - t_f1))
    return counts_all

# hs = horizontal stimulus; vs = vertical stimulus
path_hs = tools.folders_contain_str('o=0')[0] + '/data/'
path_vs = tools.folders_contain_str('o=1')[0] + '/data/'

t_begin = 2000.0            # begin of stimulation
t_end = t_begin + 20000.0   # end of stimulation
interval = 1000.0         # interval of stimulation
window = 10.0             # window of analysis

t_list = np.arange(t_begin, t_end, interval)
n_window = int(interval/window)
t_plot = np.arange(0.0, interval, window) + window/2.0

# selectivity index
# si_abs_arr = np.zeros((n_window, 13))
mean_si_hg = np.zeros((13, n_window))
mean_si_vg = np.zeros((13, n_window))
std_si_hg = np.zeros((13, n_window)) # standard deviation
std_si_vg = np.zeros((13, n_window))
se_si_hg = np.zeros((13, n_window)) # standard error
se_si_vg = np.zeros((13, n_window))
p_value_si = np.zeros((13, n_window))
significance_si = np.zeros((13, n_window))

# counts: [population][window][trial][neuron id]
t0 = time()
# if os.path.isfile(os.path.join(cwd, 'counts_hs.npy')) is False \
#         or os.path.isfile(os.path.join(cwd, 'counts_vs.npy')) is False:
# if True:
#     data_hs = count_spikes(path_hs, 'spike_detector',
#                              t_begin, t_end, t_list, window)
#     data_vs = count_spikes(path_vs, 'spike_detector',
#                              t_begin, t_end, t_list, window)
#
#     np.save(os.path.join(cwd, 'counts_hs.npy'), data_hs)
#     np.save(os.path.join(cwd, 'counts_vs.npy'), data_vs)
#
# counts_hs = np.load('counts_hs.npy')
# counts_vs = np.load('counts_vs.npy')

counts_hs = count_spikes(path_hs, 'spike_detector',
                             t_begin, t_end, t_list, window)
counts_vs = count_spikes(path_vs, 'spike_detector',
                             t_begin, t_end, t_list, window)

t1 = time()
# print('time of count_spikes = {:.3f}'.format(t1-t0))

check_arr = []
for i in range(13):
    if len(counts_hs[i]) > 0 and len(counts_vs[i]) > 0:
        for j in range(n_window):
            t00 = time()
            cnt_hs = counts_hs[i][j]
            cnt_vs = counts_vs[i][j]
            idx1 = int(len(cnt_hs[0][:])/4)
            idx2 = int(len(cnt_hs[0][:])/2)
            idx3 = int(len(cnt_hs[0][:])*3/4)
            idx4 = len(cnt_hs[0][:])
            idx_hc = np.arange(idx1, idx3)
            idx_vc = np.concatenate((np.arange(0, idx1), np.arange(idx3, idx4)))
            mean_hs = np.mean(cnt_hs, axis=0)   # mean across trials
            mean_vs = np.mean(cnt_vs, axis=0)   # mean across trials
            std_cnt = np.std(np.concatenate((cnt_hs, cnt_vs), axis=0), axis=0) # pooled std
            if i == 0 and j == 0:
                print('population {} cell number = {}'.format(i, len(mean_hs)))

            # check if there are cases of: mean not zero but std is zero
            check_arr.append(np.count_nonzero([(mean_hs[:] != 0.0) & (std_cnt[:] == 0.0)]))

            t01 = time()

            si_hc = []  # si of horizontal cells
            si_vc = []  # si of vertical cells
            si_abs = []
            mean_hs_to_hc = []
            mean_vs_to_hc = []
            # horizontal cells
            for k in idx_hc:
                if std_cnt[k] == 0.0: # just checking
                    if mean_hs[k] == 0.0 and mean_vs[k] == 0.0:
                        si_hc.append(0.0)
                        si_abs.append(0.0)
                        mean_hs_to_hc.append(0.0)
                        mean_vs_to_hc.append(0.0)
                    else:
                        print('cell # {0} std = 0 '
                              'but means != 0'.format(k))
                else:
                    si_hc.append((mean_hs[k]-mean_vs[k])/std_cnt[k])
                    si_abs.append(np.abs(mean_hs[k]-mean_vs[k])/std_cnt[k])
                    mean_hs_to_hc.append(mean_hs[k])
                    mean_vs_to_hc.append(mean_vs[k])

            mean_hs_to_vc = []
            mean_vs_to_vc = []
            # vertical cells
            for l in idx_vc:
                if std_cnt[l] == 0.0:
                    if mean_hs[l] == 0.0 and mean_vs[l] == 0.0:
                        si_vc.append(0.0)
                        si_abs.append(0.0)
                        mean_hs_to_vc.append(0.0)
                        mean_vs_to_vc.append(0.0)
                    else:
                        print('cell # {0} std = 0 '
                              'but means != 0'.format(l))
                else:
                    si_vc.append((mean_hs[l]-mean_vs[l])/std_cnt[l])
                    si_abs.append(np.abs(mean_hs[l]-mean_vs[l])/std_cnt[l])
                    mean_hs_to_vc.append(mean_hs[l])
                    mean_vs_to_vc.append(mean_vs[l])

            t02 = time()

            # si results of respective populations and windows
            # hg = horizontal group, vg = vertical group
            mean_si_hg[i, j] = np.mean(si_hc)
            mean_si_vg[i, j] = np.mean(si_vc)
            # std_si_hg[i, j] = np.std(si_hc)
            # std_si_vg[i, j] = np.std(si_vc)
            se_si_hg[i, j] = np.std(si_hc)/np.sqrt(len(si_hc))
            se_si_vg[i, j] = np.std(si_vc)/np.sqrt(len(si_vc))
            t_value, p_value = stats.ttest_ind(si_hc, si_vc)
            p_value_si[i, j] = p_value
            sig = -100.0
            if t_value > 0 and p_value <= 0.05:
                sig = 1.0
            significance_si[i, j] = sig
            # si_abs_arr[j, i] = np.mean(si_abs)

            t03 = time()
            # if i == 0:
                # print('dt00-01 = {:.3f}, dt01-02 = {:.3f}, dt02-03 = {:.3f}'.format(t01-t00, t02-t01, t03-t02))


# to check: any data excluded?
print('any data excluded? {}'.format(np.count_nonzero(check_arr)))

fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True, sharey=True, constrained_layout=True)
#fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True, sharey=True, constrained_layout=True)
plt.suptitle('            (A) {}-ms stimulus\n'.format(stimulus), fontsize=40.0)

for i, pop in enumerate(populations):
    # indexing for plots
    if i < 4:
        a = 0
        b = i % 3.0
        if i == 3:
            b = 3.0
    else:
        a = int((i-1) / 3)
        b = (i-1) % 3.0

    # horizontal and vertical cells plot
    axs[a].plot(t_plot, mean_si_hg[i, :],
                    color=colors[i],
                    label='{} \'horizontal\' cells'.format(subtype_label[int(b)]), linewidth=4)
    axs[a].plot(t_plot, mean_si_vg[i, :],
                    color=colors[i],
                    label='{} \'vertical\' cells'.format(subtype_label[int(b)]), linewidth=4, ls='--')
    # axs[a].errorbar(t_plot, mean_si_hg[i, :],
    #                 yerr=se_si_hg[i, :], color=colors[i],
    #                 label='{} \'horizontal\' cells'.format(
    #                     subtype_label[int(b)]))
    # axs[a].errorbar(t_plot, mean_si_vg[i, :],
    #                 yerr=se_si_vg[i, :], color=colors[i],
    #                 label='{} \'vertical\' cells'.format(
    #                     subtype_label[int(b)]), ls='--')
    sig_list = significance_si[i, :]

    # marker of significance
    axs[a].plot(t_plot, mean_si_hg[i, :] + sig_list*0.15,
                color=colors[i], marker='*', ls='None', markersize=10)
    # p-value
    # for x, t in enumerate(t_plot):
    #     axs[a].text(t, -0.80 - 0.1*b, '{:.3f}'.format(p_value_si[i, x]), fontsize=10)

for i, layer in enumerate(layer_label):
    # if i == 0:
    #     legend = axs[i].legend(bbox_to_anchor=(0.0, 1.0, 1.0, 0.1),
    #        ncol=2, mode="expand",borderaxespad=0., fontsize=20)
    #     legend.get_frame().set_edgecolor('w')
    axs[i].set_xticks(np.arange(t_plot[0], t_plot[-1], window) - window/2.0)
    # axs[i].tick_params(labelsize=20.0)
    axs[i].set_ylim(y_boundary[0], y_boundary[1])
    axs[i].set_xlim(0.0 - plot_length / 40.0, plot_length + plot_length / 40.0)
    axs[i].set_ylabel(layer + '                    ', rotation='horizontal')
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['top'].set_visible(False)
    if i == 0:
        axs[i].plot([0.0, stimulus], [y_boundary[1], y_boundary[1]], color='k', linewidth=20)
        # axs[i].text(x=stimulus + 1.0, y=y_boundary[1] - 0.25, s='stimulus')
    if i == 3:
        axs[i].text(-16.0, 0.5, 'selectivity\nindex', rotation='vertical')
        # ax = axs[i].twinx()
        # ax.set_ylabel('normalized\nselectivity')

# plt.hlines(y=7.0, xmin=0.0, xmax=stimulus, color='k', linewidth=4)
# plt.text(x=31.0, y=7.0, s='stimulus')
plt.xlabel('t (ms)')
plt.savefig(os.path.join(cwd, 'selectivity_{}.png'.format(str(datetime.now()))), dpi=300)
plt.show()
