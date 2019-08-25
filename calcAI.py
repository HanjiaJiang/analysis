import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from random import sample
import microcircuit_tools as tools
from time import time

# exec(tools.print2txt_init())
rootdir = os.getcwd()
matplotlib.rcParams['font.size'] = 30.0

# calculation
do_calc = True

# plotting
subtype = 'SOM'
all_population = True
constant_pv_level = 4
constant_som_level = 3

# set levels for color map
rate_levels = np.arange(0.0, 10.1, 1.0)
corr_levels = np.arange(-0.008, 0.0081, 0.002)
cv_levels = np.arange(0.95, 1.21, 0.05)
peak_freq_levels = np.arange(0.0, 50.1, 5.0)
spread_levels = np.arange(0.0, 10.1, 1.0)
amp_levels = np.arange(0.0, 50.1, 5.0)

# begin and end time of analysis
begin_end = [2000.0, 22000.0]

# parameter levels to be analyzed
para_dict = {
    'bg': [4.0],
    'g': [-4],
    'PVext': [1200, 1400, 1600, 1800, 2000],
    'SOMext': [1200, 1400, 1600, 1800, 2000],
    'VIPext': [400, 450, 500, 550, 600],
}

# for output
populations = ['L23_PC', 'L23_PV', 'L23_SOM', 'L23_VIP', 'L4_PC', 'L4_PV',
                'L4_SOM', 'L5_PC', 'L5_PV', 'L5_SOM', 'L6_PC', 'L6_PV',
                'L6_SOM']

# response quantification
th_start = np.arange(begin_end[0], begin_end[1], 1000.0)
window = 20.0 # ms

# options
calc_psd = False
calc_response = False

# power spectrum function
def hist_spectral(hist, bin_width, given_freq_resol):
    # bin_width: in second
    # given_freq_resol: in Hz

    # limits of frequency resolutions for output
    freq_resol_lower_limit = 1/(len(hist) * bin_width)  # lower limit = 1/time length of samples
    freq_resol_upper_limit = 1/(bin_width*2)            # upper limit = sampling rate/2
    if given_freq_resol < freq_resol_lower_limit or given_freq_resol > freq_resol_upper_limit:
        print('frequency resolution out of range; force using the limit')
        given_freq_resol = freq_resol_lower_limit

    # calculation of psd
    # assign the length of output according to given_freq_resol
    psd = np.fft.fft(hist, n=int(len(hist) * freq_resol_lower_limit / given_freq_resol))
    psd_positive = psd[0:int(len(psd)/2)]
    psd_negative = psd[int(len(psd)/2):-1]
    freq_list = np.linspace(0.0, (len(psd_positive) - 1) * given_freq_resol, len(psd_positive))
    return psd_positive, psd_negative, freq_list


def calcAI(path, name, begin, end):
    # files, gids = tools.read_name(path, name)
    tf0 = time()
    data_all, gids = tools.load_spike_times(path, name, begin, end)
    tf1 = time()
    print('file reading time = {:.3f}'.format(tf1 - tf0))
    data_exist = False
    corr_bin_width = 10.0

    # caches
    rate_return = []
    coef_return = []
    cv_return = []
    peak_freq_return = []
    spread_return = []
    amp_return = []

    if len(data_all) == 13:
    # if len(files) > 0 and len(data_all) == 13:
        data_exist = True
        for h in list(range(len(data_all))):
            spk_data = data_all[h]    # all spikes of this group
            cell_n = gids[h][1] - gids[h][0] + 1

            if len(spk_data) == 0:
                rate_return.append(0)
                coef_return.append(0)
                cv_return.append(0)
                peak_freq_return.append(0)
                spread_return.append(0)
                amp_return.append(0)
            else:
                # calculate firing rate
                t0 = time()
                n_fil = spk_data[:, 0]
                t_all = spk_data[:, 1]
                if len(n_fil) == 0:
                    rate_return.append(0.0)
                else:
                    n_fil = n_fil.astype(int)
                    count_of_n = np.bincount(n_fil)
                    count_of_n_fil = count_of_n[gids[h][0] - 1:gids[h][1]]
                    rate_each_n = count_of_n_fil * 1000. / (end - begin)
                    rate_averaged = np.mean(rate_each_n)
                    rate_return.append(rate_averaged)

                # power spectrum
                if calc_psd:
                    psd_bin_width = 0.01            # in second
                    psd_freq_resol = 5.0            # in Hz
                    bin_number = int((end-begin)/(1000.0*psd_bin_width))
                    all_t_hist, all_t_bin_edges = np.histogram(t_all, bin_number, (begin, end))
                    psd_p, psd_n, freqs = hist_spectral(all_t_hist, psd_bin_width, psd_freq_resol)
                    avg_psd_p = np.abs(psd_p)/cell_n
                    # find max in ac range
                    avg_psd_p_ac = avg_psd_p[1:]
                    freq_ac = freqs[1:]
                    max_val = 0
                    max_idx = 0
                    for i, item in enumerate(avg_psd_p_ac):
                        if item > max_val:
                            max_val = item
                            max_idx = i
                    peak_freq_return.append(freq_ac[max_idx])

                # pick up random data for cc and cv
                t1 = time()
                hist_list = []
                t_list_isi = []
                shuffled_n_list = sample(list(range(gids[h][0], gids[h][1] + 1)), cell_n)
                # times_list = np.array([])  # times of spikes
                hist_pop = np.zeros(int((end - begin) / corr_bin_width))
                for n in shuffled_n_list:
                    # spike times of each neuron
                    times = t_all[n_fil == n]
                    if len(times) > 0:
                        if len(hist_list) < 50: # stop collecting if enough
                            # make histogram
                            hist, bin_edges = np.histogram(
                                times,
                                int((end - begin) / corr_bin_width),
                                (begin, end))
                            hist_list.append(hist)
                            hist_pop = np.add(hist_pop, hist)
                        if len(t_list_isi) < 50:
                            if len(times) > 3:  # only if more than 3 ISIs
                                t_list_isi.append(times)
                        else:
                            break

                # calculate cc
                # the time is proportional to n_coef x n_coef !!
                # --> must limit n_coef to 50, otherwise it takes forever
                t2 = time()
                coef_list = []
                n_coef = len(hist_list)
                if n_coef < 50 and h in [0, 4, 7, 10]:
                    print('{0} n for corr={1}'.format(populations[h], n_coef))
                    if n_coef == 0:
                        print('--> n = 0 !!')
                for i, hist1 in enumerate(hist_list):
                    for j, hist2 in enumerate(hist_list[i+1:]):
                        coef = np.corrcoef(hist1, hist2)[0, 1]
                        coef_list.append(coef)
                coef_mean = np.mean(coef_list)
                coef_return.append(coef_mean)

                # calculate cv
                # cv_list = []
                # n_cv = len(t_list_isi)
                # if n_cv < 50 and h in [0, 4, 7, 10]:
                #     print('{0} n for CV={1}'.format(populations[h], n_cv))
                #     if n_cv == 0:
                #         print('--> n = 0 !!')
                # for i, times in enumerate(t_list_isi):
                #     isi = np.diff(times)
                #     cv_list.append(np.std(isi)/np.mean(isi))
                # cv_return.append(np.mean(cv_list))
                cv_return.append(0)

                # response spread and amp
                if calc_response:
                    spread_list = []
                    amp_list = []
                    for i, stim_t in enumerate(th_start):
                        tmp_t_list = t_all[    # t list of each response
                            (t_all > stim_t) & (t_all <= stim_t + window)
                        ]
                        t_std = np.std(tmp_t_list)
                        if len(tmp_t_list) > 0:
                            spread_list.append(t_std)
                            amp_list.append(len(tmp_t_list))
                        else:
                            spread_list.append(0.0)
                            amp_list.append(0)
                    spread_return.append(np.mean(spread_list))
                    amp_return.append(np.mean(amp_list))

                t3 = time()
                print('dt = {:.3f}, {:.3f}, {:.3f},'.format(t1-t0, t2-t1, t3-t2))

    return data_exist, rate_return, coef_return, cv_return, peak_freq_return, spread_return, amp_return


def plotAI(analysis, para_dict, data, levels, subtype):
    labels = ['PC', 'PV', 'SOM', 'VIP']
    for bg_idx, bg in enumerate(para_dict['bg']):
        for g_idx, g in enumerate(para_dict['g']):
            title_str = 'bg={0}Hz,g={1}'.format(bg, g)

            # create map grid data
            if subtype == 'SOM':
                X, Y = np.meshgrid(para_dict['SOMext'], para_dict['VIPext'])
                Z = [np.zeros(np.shape(X)) for i in range(13)]
                for i in range(len(para_dict['SOMext'])):
                    for j in range(len(para_dict['VIPext'])):
                        for n in range(13):
                            Z[n][j][i] = data[bg_idx][g_idx][constant_pv_level][i][j][n]    # j is y-axis
            else:
                X, Y = np.meshgrid(para_dict['PVext'], para_dict['VIPext'])
                Z = [np.zeros(np.shape(X)) for i in range(13)]
                for i in range(len(para_dict['PVext'])):
                    for j in range(len(para_dict['VIPext'])):
                        for n in range(13):
                            Z[n][j][i] = data[bg_idx][g_idx][i][constant_som_level][j][n]  # j is y-axis


            # plotting
            if all_population:
                fig1, axs1 = plt.subplots(4, 4, figsize=(12, 12), sharex=True, sharey=True, constrained_layout=True)
                axs1 = axs1.ravel()
                fig1.suptitle(analysis + ': ' + title_str)
                for i in range(len(Z)):
                    # shift the subplot position
                    if i <= 6:
                        j = i
                    elif i <= 9:
                        j = i + 1
                    else:
                        j = i + 2

                    # data color; NaN is white
                    cs = axs1[j].contourf(X, Y, Z[i], levels, extend='both')
                    cs.cmap.set_over("orange")
                    cs.cmap.set_under("purple")

                    if i < 4:
                        axs1[i].title.set_text(labels[i])
                    if i == len(Z) - 1:
                        clb = fig1.colorbar(cs)
                    if j == 12:
                        axs1[j].set_xlabel('indegree to ' + subtype + '+')
                        axs1[j].set_ylabel('indegree to\nVIP+')

                fig1.savefig(analysis + '_' + title_str + '.png')
                plt.close()

            else:
                fig2, axs2 = plt.subplots(4, 1, figsize=(6, 14), sharex=True, sharey=True, constrained_layout=True)
                axs2 = axs2.ravel()
                fig2.suptitle('    ', fontsize=40)
                Z_excitatory = [Z[0], Z[4], Z[7], Z[10]]
                axs2[3].set_xlabel('indegree to ' + subtype + '+', fontsize=30)
                plt.xticks(np.arange(para_dict[subtype + 'ext'][0], para_dict[subtype + 'ext'][-1] +1, 200), tuple([str(x) for x in para_dict[subtype + 'ext']]), rotation=35)
                # plt.yticks(np.arange(1200, 2001, 200), ('1200', '1400', '1600', '1800', '2000'))
                axs2[3].set_ylabel('\n\nindegree to\nVIP+', fontsize=30)
                for i in range(4):
                    cs = axs2[i].contourf(X, Y, Z_excitatory[i], levels, extend='both')
                    cs.cmap.set_over("orange")
                    cs.cmap.set_under("purple")
                    if i == 3:
                        clb = fig2.colorbar(cs, orientation='horizontal')
                        xticklabels = clb.ax.get_xticklabels()
                        if analysis == 'corr':
                            clb.ax.set_xticklabels(xticklabels, rotation=45, fontsize=18)
                        #else:
                            #clb.ax.set_xticklabels(xticklabels, rotation=35)
                        if analysis == 'fr':    # firing rate
                            clb.set_label('spikes per s', fontsize=30)
                        # else:
                        #     clb.set_label('   ')
                fig2.savefig(analysis + '_E_' + title_str + '.png')
                plt.close()

# calculation
if do_calc:
    rate_arr = np.full((len(para_dict['bg']), len(para_dict['g']), len(para_dict['PVext']), len(para_dict['SOMext']), len(para_dict['VIPext']), 13), np.nan)
    corr_arr = np.full((len(para_dict['bg']), len(para_dict['g']), len(para_dict['PVext']), len(para_dict['SOMext']), len(para_dict['VIPext']), 13), np.nan)
    cv_arr = np.full((len(para_dict['bg']), len(para_dict['g']), len(para_dict['PVext']), len(para_dict['SOMext']), len(para_dict['VIPext']), 13), np.nan)
    peak_freq_arr = np.full((len(para_dict['bg']), len(para_dict['g']), len(para_dict['PVext']), len(para_dict['SOMext']), len(para_dict['VIPext']), 13), np.nan)
    spectrum_arr = np.full((len(para_dict['bg']), len(para_dict['g']), len(para_dict['PVext']), len(para_dict['SOMext']), len(para_dict['VIPext']), 13), np.nan)
    spread_arr = np.full((len(para_dict['bg']), len(para_dict['g']), len(para_dict['PVext']), len(para_dict['SOMext']), len(para_dict['VIPext']), 13), np.nan)
    amp_arr = np.full((len(para_dict['bg']), len(para_dict['g']), len(para_dict['PVext']), len(para_dict['SOMext']), len(para_dict['VIPext']), 13), np.nan)
    for i, bg in enumerate(para_dict['bg']):
        for j, g in enumerate(para_dict['g']):
            for k, PV_ext in enumerate(para_dict['PVext']):
                for l, SOM_ext in enumerate(para_dict['SOMext']):
                    for m, VIP_ext in enumerate(para_dict['VIPext']):
                        para_str = 'bg={0}_g={1}_PV_ext={2}_SOM_ext={3}_VIP_ext={4}'.format(bg, g, PV_ext, SOM_ext, VIP_ext)
                        data_path = rootdir + '/' + para_str + '/data/'
                        if os.path.isdir(data_path):
                            print('\nstart analyzing:' + para_str)
                            have_data, rate_list, corr_list, cv_list, peak_freq_return, spread_return, amp_return = \
                                calcAI(data_path, 'spike_detector', begin_end[0], begin_end[1])
                            if have_data:
                                for n in range(13):
                                    rate_arr[i][j][k][l][m][n] = rate_list[n]
                                    corr_arr[i][j][k][l][m][n] = corr_list[n]
                                    cv_arr[i][j][k][l][m][n] = cv_list[n]
                                    if calc_psd:
                                        peak_freq_arr[i][j][k][l][m][n] = peak_freq_return[n]
                                    if calc_response:
                                        spread_arr[i][j][k][l][m][n] = spread_return[n]
                                        amp_arr[i][j][k][l][m][n] = amp_return[n]
                            else:
                                print('folder no valid data !!')
    np.save(os.path.join(rootdir, 'rate.npy'), rate_arr)
    np.save(os.path.join(rootdir, 'corr.npy'), corr_arr)
    np.save(os.path.join(rootdir, 'cv.npy'), cv_arr)
    if calc_psd:
        np.save(os.path.join(rootdir, 'psd.npy'), peak_freq_arr)
    if calc_response:
        np.save(os.path.join(rootdir, 'spread.npy'), spread_arr)
        np.save(os.path.join(rootdir, 'amp.npy'), amp_arr)

# plot
plotAI('fr', para_dict, np.load('rate.npy'), rate_levels, subtype)
plotAI('corr', para_dict, np.load('corr.npy'), corr_levels, subtype)
plotAI('cv', para_dict, np.load('cv.npy'), cv_levels, subtype)
if calc_psd:
    plotAI('psd', para_dict, np.load('psd.npy'), peak_freq_levels, subtype)
if calc_response:
    plotAI('response dispersion', para_dict, np.load('spread.npy'), spread_levels, subtype)
    plotAI('response amp', para_dict, np.load('amp.npy'), amp_levels, subtype)

# exec(tools.print2txt_end())
