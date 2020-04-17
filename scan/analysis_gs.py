import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.tri as tri
from matplotlib.patches import Polygon
# my own modules
try:
    import tools
    import interpol
except ImportError:
    pass

matplotlib.rcParams['font.size'] = 20.0

do_interpol = True

layers = ['L2/3', 'L4', 'L5', 'L6']

# criteria from Maksimov et al., 2018
criteria_fr = [0.0, 10.0]
criteria_corr = [0.0001, 0.008]
criteria_cv = [0.76, 1.2]
criteria_sf_spread = [-100.0, 100.0]
criteria_sf_amp = [-10000.0, 10000.0]

# extra criteria: Four-layer excitatory firing rate data from Yu, Svoboda, 2019
exc_fr_high = [2.7 + 3.7/np.sqrt(5), 0.5 + 0.8/np.sqrt(95), 6.8 + 5.2/np.sqrt(23), 6.1 + 6.9/np.sqrt(30)]
exc_fr_low = [2.7 - 3.7/np.sqrt(5), 0.5 - 0.8/np.sqrt(95), 6.8 - 5.2/np.sqrt(23), 6.1 - 6.9/np.sqrt(30)]


# read comma separated fr/corr/cv data, return as an array
def read_data(name):
    r_arr = []
    if os.path.isfile(name):
        f = open(name, 'r')
        s = f.read()
        f.close()
        for line in s.splitlines():
            r_arr.append(line.split(','))
    return r_arr


# function to draw the colormap
# low, high: criteria boundaries
def colormap(prefix, name, data, xs, ys, low, high,
             low_extra=None, high_extra=None, fit_mtx=None, v_range=None, cmap='RdBu'):
    xs = np.array(xs)
    ys = np.array(ys)

    # the threshold for outer circle of delaunay triangles
    delaunay_alpha = np.sqrt((xs[1]-xs[0])**2 + (ys[1]-ys[0])**2)

    # rotate_format = '%.1e'
    contour_color = 'black'

    # for extra criteria
    flg_extra_line = False
    if isinstance(low_extra, list) and isinstance(high_extra, list):
        if len(low_extra) == 4 and len(high_extra) == 4:
            flg_extra_line = True

    # set plotting variables
    fig, axs = plt.subplots(4, 1, figsize=(6, 14), sharex=True, sharey=True, constrained_layout=True)
    axs = axs.ravel()
    plot_name = '{}_{}.png'.format(prefix, name)

    # color range
    if type(v_range) is tuple and len(v_range) == 2:
        vmin = v_range[0]
        vmax = v_range[1]
    else:
        vmin = low - (high - low) * 2
        vmax = high + (high - low) * 2

    # x and y labels to be improved
    plt.xlabel('g\n')
    plt.ylabel('bg_rate (spikes/s)', va='center', rotation='vertical')
    # fig.text(0.16, 0.58, 'bg_rate (spikes/s)', va='center', rotation='vertical')
    plt.xticks(xs, rotation=30)
    plt.yticks(ys)
    plt.xlim((xs[0], xs[-1]))
    plt.ylim((ys[0], ys[-1]))

    for k, data_layer in enumerate(data):
        # data transposed so that row lies in x and column lies in y
        Z = data_layer.T

        # define plot borders
        extent = [np.min(xs), np.max(xs), np.min(ys), np.max(ys)]

        # colormap plots
        cs = axs[k].imshow(Z, interpolation='none', cmap=cmap, origin='lower',
                           extent=extent, vmax=vmax, vmin=vmin)

        # contour main criteria
        cs_line = axs[k].contour(Z, levels=[low, high], colors=contour_color,
                                 extent=extent, linewidths=2)

        # contour extra criteria
        if flg_extra_line:
            axs[k].contour(Z, levels=[low_extra[k], high_extra[k]], colors=contour_color,
                           extent=extent, linewidths=4, linestyles='dashed')

        # shaded patch for 'all fit' data
        if type(fit_mtx) == np.ndarray:
            idx1, idx2 = np.where(fit_mtx[k] == 1)
            xlist = xs[idx1]
            ylist = ys[idx2]
            axs[k].scatter(xlist, ylist, s=50, color='r', zorder=10)
            # if do_interpol:
                # if len(xlist) > 3:
                    # xi, yi = interpol.get_outline(xlist.tolist(), ylist.tolist(), delaunay_alpha)
                    # xi, yi = interpol.sort_by_angle(xlist.tolist(), ylist.tolist())
                    # xi, yi = interpol.interpol_spline(xi, yi)
                    # axs[k].add_patch(Polygon(np.array([xi, yi]).T, closed=True, fill=False, hatch='x', color='r', zorder=10))
            #         triang = tri.Triangulation(xlist, ylist)
            #         axs[k].triplot(triang, 'r-', zorder=10)

        # set off-limit colors
        cs.cmap.set_over("midnightblue")
        cs.cmap.set_under("firebrick")

        # mark layer labels (L2/3 ~ L6)
        axs[k].text(xs[0] - 0.5*(xs[-1] - xs[0]), ys[0] + 0.5*(ys[-1] - ys[0]), layers[k])

        # set plot aspect ratio
        axs[k].set_aspect(float((xs[-1] - xs[0])/(ys[-1] - ys[0])))


    # colorbar
    cbar = fig.colorbar(cs, ax=axs.tolist(), orientation='horizontal', shrink=0.6)
    # cbar = fig.colorbar(cs, cax=axins1, orientation='horizontal')
    cbar.ax.plot([low, low], [vmin, vmax], color='k', linewidth=2)
    cbar.ax.plot([high, high], [vmin, vmax], color='k', linewidth=2)

    name += '\nscore = {}'.format(np.count_nonzero(fit_mtx))
    fig.suptitle(name)
    fig.savefig(plot_name)
    plt.close()
    return plot_name


# check the data fitness
def check_fitness(data_fr, data_corr, data_cv, cri_fr, cri_corr, cri_cv):
    if type(data_fr) == np.ndarray and \
            type(data_corr) == np.ndarray and \
            type(data_cv) == np.ndarray:
        chk_mtx = np.full(data_fr.shape, 0)
        for i, lyr in enumerate(chk_mtx):
            cnt_lyr = 0
            for j, row in enumerate(lyr):
                for k, itm in enumerate(row):
                    flg = True
                    tmp1 = data_fr[i][j][k]
                    tmp2 = data_corr[i][j][k]
                    tmp3 = data_cv[i][j][k]
                    if tmp1 < cri_fr[0] or tmp1 > cri_fr[1] \
                            or tmp2 < cri_corr[0] or tmp2 > cri_corr[1] \
                            or tmp3 < cri_cv[0] or tmp3 > cri_cv[1] \
                            or np.isnan(tmp1) or np.isnan(tmp2) or np.isnan(tmp3):
                        flg = False
                    if flg is True:
                        chk_mtx[i, j, k] = 1
                        cnt_lyr += 1
    else:
        chk_mtx = None
    return chk_mtx


def str2list(list_str):
    cache = []
    for item in list_str:
        cache.append(np.array(item.split('/')[-2].split('_')).astype(int))
    cache = np.array(cache)
    lvls_1 = list(set(cache[:, 0]))
    lvls_2 = list(set(cache[:, 1]))
    lvls_3 = list(set(cache[:, 2]))
    lvls_4 = list(set(cache[:, 3]))
    return lvls_1, lvls_2, lvls_3, lvls_4

# to-do: put into several functions
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # get input and output names
    inputs = sys.argv[1:]
    lvls_1, lvls_2, lvls_g, lvls_bg = str2list(inputs)
    print('levels =\n{}\n{}\n{}\n{}\n'.format(lvls_1, lvls_2, lvls_g, lvls_bg))

    # get dimension shape
    input_shape = (len(layers), len(lvls_g), len(lvls_bg))

    print('data shape = {}\n'.format(input_shape))

    # constant parameters, e.g. ctsp and stp
    for a, lvl1 in enumerate(lvls_1):
        for b, lvl2 in enumerate(lvls_2):
            data_a = np.full(input_shape, np.nan)
            data_i = np.full(input_shape, np.nan)
            data_fr_exc = np.full(input_shape, np.nan)
            data_fr_pv = np.full(input_shape, np.nan)
            data_fr_som = np.full(input_shape, np.nan)
            data_fr_vip = np.full(input_shape, np.nan)
            data_sf_spread = np.full(input_shape, np.nan)
            data_sf_amp = np.full(input_shape, np.nan)
            for c, lvlg in enumerate(lvls_g):
                for d, lvlbg in enumerate(lvls_bg):
                    ai = read_data(os.path.join('scans', '{}_{}_{}_{}'.format(lvl1, lvl2, lvlg, lvlbg), 'ai.dat'))
                    fr = read_data(os.path.join('scans', '{}_{}_{}_{}'.format(lvl1, lvl2, lvlg, lvlbg), 'fr.dat'))
                    sf = read_data(os.path.join('scans', '{}_{}_{}_{}'.format(lvl1, lvl2, lvlg, lvlbg), 'sf.dat'))
                    # read data from each layer and group
                    if len(ai) == 4:
                        for i, ai_lyr in enumerate(ai):
                            data_a[i, c, d] = float(ai_lyr[0])
                            data_i[i, c, d] = float(ai_lyr[1])
                    if len(fr) == 13:
                        for i, fr_lyr in enumerate(np.array(fr)[[0, 4, 7, 10]]):
                            data_fr_exc[i, c, d] = float(fr_lyr[0])
                        for i, fr_lyr in enumerate(np.array(fr)[[1, 5, 8, 11]]):
                            data_fr_pv[i, c, d] = float(fr_lyr[0])
                        for i, fr_lyr in enumerate(np.array(fr)[[2, 6, 9, 12]]):
                            data_fr_som[i, c, d] = float(fr_lyr[0])
                        data_fr_vip[0, c, d] = float(fr[3][0])
                    if len(sf) == 4:
                        for i, sf_lyr in enumerate(sf):
                            data_sf_spread[i, c, d] = float(sf_lyr[0])
                            data_sf_amp[i, c, d] = float(sf_lyr[1])

            # check fitness
            fitness_mtx = check_fitness(data_fr_exc, data_a, data_i, criteria_fr, criteria_corr, criteria_cv)

            # plotting:
            # ground state
            names_gs = []
            tag_gs = '({},{})_'.format(lvl1, lvl2)
            names_gs.append(colormap(tag_gs + 'A', 'fr-exc', data_fr_exc, lvls_g, lvls_bg, criteria_fr[0], criteria_fr[1],
                     low_extra=exc_fr_low, high_extra=exc_fr_high, fit_mtx=fitness_mtx, v_range=(0.0, 30.0), cmap='Blues'))
            names_gs.append(colormap(tag_gs + 'B', 'pair-corr', data_a, lvls_g, lvls_bg, criteria_corr[0], criteria_corr[1],
                     v_range=(-0.02, 0.02), fit_mtx=fitness_mtx))
            names_gs.append(colormap(tag_gs + 'C', 'cv-isi', data_i, lvls_g, lvls_bg, criteria_cv[0], criteria_cv[1],
                     fit_mtx=fitness_mtx, v_range=(0.0, 1.5), cmap='Blues'))
            names_gs.append(colormap(tag_gs + 'D', 'fr-pv', data_fr_pv, lvls_g, lvls_bg, -np.inf, np.inf,
                     v_range=(0.0, 50.0), cmap='Blues'))
            names_gs.append(colormap(tag_gs + 'E', 'fr-som', data_fr_som, lvls_g, lvls_bg, -np.inf, np.inf,
                     v_range=(0.0, 50.0), cmap='Blues'))
            names_gs.append(colormap(tag_gs + 'F', 'fr-vip', data_fr_vip, lvls_g, lvls_bg, -np.inf, np.inf,
                     v_range=(0.0, 50.0), cmap='Blues'))

            # join plots
            try:
                tools.hori_join(names_gs, tag_gs)

                # remove original plots; to be improved ...
                for name in names_gs:
                    if os.path.exists(name):
                          os.remove(name)

                for name in names_sf:
                    if os.path.exists(name):
                          os.remove(name)
            except NameError:
                pass
