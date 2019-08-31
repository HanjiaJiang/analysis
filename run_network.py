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
import nest

run_sim = True
run_calc = True
test_pairwise = True
corr_roi = [1000.0, 2000.0]
tau_max = 1000.0

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

def pairwise_corr(path, name, begin, end):
    data_all, gids = tools.load_spike_times(path, name, begin, end)
    l23_hist_arr = []
    corr_bin_width = 125.0
    net_coef_arr = np.full((4, 4), np.nan)
    if len(data_all) >= 1:
        # collect histograms
        # for h in range(4):
        pop_nids = (data_all[1][:, 0])
        pop_times = (data_all[1][:, 1])
        pop_hist_list = []
        n_cnt = gids[1][1] - gids[1][0] + 1
        shuffled_n_list = sample(list(range(gids[1][0], gids[1][1] + 1)),
                                 n_cnt)

        # get historgram of each neuron
        for n in shuffled_n_list:
            # spike times of each neuron
            times = pop_times[pop_nids == n]
            # if len(times) > 0:
            # make histogram
            hist, bin_edges = np.histogram(
                times,
                int((end - begin) / corr_bin_width),  # nr. of bins
                (begin, end))  # window of analysis
            pop_hist_list.append(hist)
            # if len(pop_hist_list) >= 50:
            #     break

        l23_e_hist_list = pop_hist_list
        corr_arr = []
        for hist1 in l23_e_hist_list:
            tmp_list = []
            for hist2 in l23_e_hist_list:
                tmp_list.append(np.corrcoef(hist1, hist2)[0, 1])
            corr_arr.append(tmp_list)
        print('real L2/3 E corr matrix = {}'.formage(corr_arr))


# nest.SetDefaults('correlomatrix_detector', {'delta_tau': 100.0})
correlo = nest.Create('correlomatrix_detector')
if run_sim:
    net = network.Network(sim_dict, net_dict, stim_dict)
    net.setup()
    if test_pairwise:
        corr_n = 0
        for i in range(4):
            corr_n += net_dict['N_full'][i] # summing number of neurons in L2/3
        print(corr_n)
        nest.SetStatus(correlo, {'N_channels': corr_n, 'delta_tau': 125.0, 'tau_max': 1000.0, 'Tstart': 1000.0})
        # list of all n in L2/3
        l23_pops = []
        for i in range(4):
            tmp_pop = net.pops[i]
            for j in tmp_pop:
                l23_pops.append(j)
        if len(l23_pops) == corr_n:
            nest.Connect(l23_pops, correlo)
    net.simulate()
    # tools.plot_raster(
    #     sim_dict['data_path'], 'spike_detector', 2000.0, 2200.0
    # )
    plt.close()

if test_pairwise:
    correlo_arr = nest.GetStatus(n.correlo, 'count_covariance')
    print('test L2/3 E corr matrix = {}'.formage(correlo_arr))
    pass
else:
    if run_calc:
        tmp_arr = network_corr(
            sim_dict['data_path'], 'spike_detector', corr_roi[0], corr_roi[1])
        np.save('coef_arr.npy', tmp_arr)

    coef_arr = np.load('coef_arr.npy')
    print(coef_arr)

    labels = ['E', 'PV', 'SOM', 'VIP']
    tools.interaction_barplot(coef_arr, -0.1, 0.25, labels, 'mean corr coef')
# x = np.arange(len(labels))
# barwidth = 0.1
# fig, ax = plt.subplots(figsize=(12, 12))
# for i in range(4):
#     ax.bar(x + barwidth*(i - 1.5), coef_arr[i, :], barwidth, label=labels[i])
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()
# plt.ylim((-0.1, 0.25))
# fig.tight_layout()
# plt.savefig('corr_model.png')
# plt.show()
