import sys
import numpy as np
import matplotlib.pyplot as plt
from random import sample
on_server = False
if not on_server:
    sys.path.insert(1, '/home/hanjia/Documents/Potjans_2014_selectivity/')
import network
from network_params import net_dict
from sim_params import sim_dict
from stimulus_params import stim_dict
import microcircuit_tools as tools

run_sim = False
corr_roi = [1200, 2200]

# correlation
# def get_mean_corr(list_1, list_2, same_pop=False):
#     coef_list = []
#     for i, hist1 in enumerate(list_1):
#         if not same_pop:
#             list_2_tmp = list_2
#         else:   # same population, avoid repetition
#             list_2_tmp = list_2[i + 1:]
#         for j, hist2 in enumerate(list_2_tmp):
#             if np.sum(hist1) != 0 and np.sum(hist2) != 0:
#                 coef = np.corrcoef(hist1, hist2)[0, 1]
#                 coef_list.append(coef)
#     return np.mean(coef_list)


def network_corr(path, name, begin, end):
    data_all, gids = tools.load_spike_times(path, name, begin, end)
    l23_hist_arr = []   # 4 pops x neuron counts in each pop x histogram
    corr_bin_width = 125.0
    net_coef_arr = np.full((4, 4), np.nan)
    if len(data_all) >= 4:
        # collect histograms
        for h in range(4):
            pop_nids = (data_all[h][:, 0])
            pop_times = (data_all[h][:, 1])
            pop_hist_list = []
            n_cnt = gids[h][1] - gids[h][0] + 1
            shuffled_n_list = sample(list(range(gids[h][0], gids[h][1] + 1)),
                                     n_cnt)

            # get historgram of each neuron
            for n in shuffled_n_list:
                # spike times of each neuron
                times = pop_times[pop_nids == n]
                if len(times) > 0:
                    # make histogram
                    hist, bin_edges = np.histogram(
                        times,
                        int((end - begin) / corr_bin_width), # nr. of bins
                        (begin, end))   # window of analysis
                    pop_hist_list.append(hist)
                    # if len(pop_hist_list) >= 50:
                    #     break

            l23_hist_arr.append(pop_hist_list)

        for i, pop_hist_list_1 in enumerate(l23_hist_arr):
            for j, pop_hist_list_2 in enumerate(l23_hist_arr):
                if j == i:
                    net_coef_arr[i, j] = \
                        tools.get_mean_corr(pop_hist_list_1, pop_hist_list_2, True)
                elif j > i:
                    net_coef_arr[i, j] = \
                        tools.get_mean_corr(pop_hist_list_1, pop_hist_list_2, False)
                    net_coef_arr[j, i] = net_coef_arr[i, j]
                else:
                    pass

    return net_coef_arr

if run_sim:
    net = network.Network(sim_dict, net_dict, stim_dict)
    net.setup()
    net.simulate()
    tools.plot_raster(
        sim_dict['data_path'], 'spike_detector', 2000.0, 2200.0
    )
    plt.close()
coef_arr = network_corr(
    sim_dict['data_path'], 'spike_detector', corr_roi[0], corr_roi[1])
print(coef_arr)
plt.plot(coef_arr.T)
plt.ylim((-0.1, 0.25))
plt.savefig('corr_model.png')
plt.show()