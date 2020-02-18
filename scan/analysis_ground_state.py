import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.tri as tri
matplotlib.rcParams['font.size'] = 20.0

layers = ['L2/3', 'L4', 'L5', 'L6']

# criteria from Maksimov et al., 2018
criteria_fr = [0.0, 10.0]
criteria_corr = [0.0001, 0.008]
criteria_cv = [0.76, 1.2]

# levels of g and bg_rate; must match the 'scans' data
lvls_g = np.linspace(4.0, 8.0, 5)
lvls_bg = np.linspace(2.0, 6.0, 5)

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
def colormap(name, data, xs, ys, low, high,
             low_extra=None, high_extra=None, fit_mtx=None, v_range=None, cmap='RdBu'):
    # rotate_format = '%.1e'
    criteria_color = 'black'

    # for extra criteria
    flg_extra_line = False
    if isinstance(low_extra, list) and isinstance(high_extra, list):
        if len(low_extra) == 4 and len(high_extra) == 4:
            flg_extra_line = True

    # set plotting variables
    fig, axs = plt.subplots(4, 1, figsize=(6, 14), sharex=True, sharey=True, constrained_layout=True)
    axs = axs.ravel()
    plot_name = '{}.png'.format(name)
    vmax = high + (high - low) * 2
    vmin = low - (high - low) * 2
    if type(v_range) is tuple and len(v_range) == 2:
        vmin = v_range[0]
        vmax = v_range[1]

    # x and y labels (to be improved)
    axs[-1].set_xlabel('g\n')
    # axs[-1].set_ylabel('bg_rate (spikes/s)', rotation=90, color='w')
    fig.text(0.16, 0.58, 'bg_rate (spikes/s)', va='center', rotation='vertical')

    for k, data_layer in enumerate(data):
        # data transposed so that g is x axis and bg_rage is y axis
        Z = data_layer.T

        # define x and y border of the plot
        extent = [np.min(xs), np.max(xs), np.min(ys), np.max(ys)]

        # origin='lower': origin of the plot at lower-left corner
        cs = axs[k].imshow(Z, interpolation='gaussian', cmap=cmap, origin='lower',
                           extent=extent, vmax=vmax, vmin=vmin)

        # contour of the main criteria
        cs_line = axs[k].contour(Z, levels=[low, high], colors=[criteria_color, criteria_color],
                                 extent=extent, linewidths=2)

        # contour of the extra criteria
        if flg_extra_line:
            axs[k].contour(Z, levels=[low_extra[k], high_extra[k]], colors=[criteria_color, criteria_color],
                           extent=extent, linewidths=4, linestyles='dashed')

        # scatter plot for 'all fit' data points
        if type(fit_mtx) == np.ndarray:
            idx1, idx2 = np.where(fit_mtx[k] == 1)
            # axs[k].scatter(xs[idx1], ys[idx2], s=150, marker='*', color='yellow', zorder=10, edgecolor='black')
            xlist = xs[idx1]
            ylist = ys[idx2]
            if all(x==xlist[0] for x in xlist) or all(y==ylist[0] for y in ylist):
                axs[k].plot(xlist, ylist, 'r', zorder=10)
            else:
                triang = tri.Triangulation(xlist, ylist)
                axs[k].triplot(triang, 'r-', zorder=10)
                # axs[k].tripcolor(xlist, ylist, np.zeros(len(xlist)))

        # set off-limit colors
        cs.cmap.set_over("midnightblue")
        cs.cmap.set_under("firebrick")

        # mark layer labels (L2/3 ~ L6)
        axs[k].text(xs[0] - 0.5*(xs[-1] - xs[0]), ys[0] + 0.5*(ys[-1] - ys[0]), layers[k])

        # set color bar
        if k == 3:
            cbar = fig.colorbar(cs, orientation='horizontal')
            cbar.add_lines(cs_line)
            # if flg_extra_line:
            #     cbar.add_lines(cs_line_extra) # will override cs_line; to be solved
    fig.savefig(plot_name)
    plt.close()


# check the data fitness
def check_fitness(data_fr, data_corr, data_cv, cri_fr, cri_corr, cri_cv):
    if type(data_fr) == np.ndarray and \
            type(data_corr) == np.ndarray and \
            type(data_cv) == np.ndarray:
        chk_mtx = np.full(data_fr.shape, 0)
        for i, lyr in enumerate(chk_mtx):
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
                        # print(tmp1, tmp2, tmp3)
                        chk_mtx[i, j, k] = 1
    else:
        chk_mtx = None
    return chk_mtx


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # get input and output names
    inputs = sys.argv[1:-1]
    output = sys.argv[-1]
    print('inputs =\n{}'.format(inputs))
    print('output = {}'.format(output))

    # get dimension shape from the last
    input_shape = (len(layers), len(lvls_g), len(lvls_bg))
    data_a = np.full(input_shape, np.nan)
    data_i = np.full(input_shape, np.nan)
    data_fr_exc = np.full(input_shape, np.nan)
    data_fr_pv = np.full(input_shape, np.nan)
    data_fr_som = np.full(input_shape, np.nan)

    print('data shape = {}'.format(data_a.shape))

    for item in inputs:
        # get parameters
        params = tuple(map(int, item.split('/')[1].split('_')))
        ai = read_data(os.path.join(item, 'ai.dat'))
        fr = read_data(os.path.join(item, 'fr.dat'))

        # read data from each layer and group
        if len(ai) == 4:
            for i, ai_lyr in enumerate(ai):
                print('ai', params, ai_lyr[0], ai_lyr[1])
                data_a[i, lvls_g.tolist().index(params[1]), lvls_bg.tolist().index(params[2])] = float(ai_lyr[0])
                data_i[i, lvls_g.tolist().index(params[1]), lvls_bg.tolist().index(params[2])] = float(ai_lyr[1])
        if len(fr) == 13:
            for i, fr_lyr in enumerate(np.array(fr)[[0, 4, 7, 10]]):
                print('fr', params, fr_lyr[0], fr_lyr[1])
                data_fr_exc[i, lvls_g.tolist().index(params[1]), lvls_bg.tolist().index(params[2])] = float(fr_lyr[0])
            for i, fr_lyr in enumerate(np.array(fr)[[1, 5, 8, 11]]):
                print('fr', params, fr_lyr[0], fr_lyr[1])
                data_fr_pv[i, lvls_g.tolist().index(params[1]), lvls_bg.tolist().index(params[2])] = float(fr_lyr[0])
            for i, fr_lyr in enumerate(np.array(fr)[[2, 6, 9, 12]]):
                print('fr', params, fr_lyr[0], fr_lyr[1])
                data_fr_som[i, lvls_g.tolist().index(params[1]), lvls_bg.tolist().index(params[2])] = float(fr_lyr[0])

    # check fitness
    fitness_mtx = check_fitness(data_fr_exc, data_a, data_i, criteria_fr, criteria_corr, criteria_cv)

    # plotting
    colormap('pair-corr', data_a, lvls_g, lvls_bg, criteria_corr[0], criteria_corr[1], v_range=(-0.02, 0.02), fit_mtx=fitness_mtx)
    colormap('cv-isi', data_i, lvls_g, lvls_bg, criteria_cv[0], criteria_cv[1], fit_mtx=fitness_mtx, v_range=(0.0, 1.5), cmap='Blues')
    colormap('fr-exc', data_fr_exc, lvls_g, lvls_bg, criteria_fr[0], criteria_fr[1],
             low_extra=exc_fr_low, high_extra=exc_fr_high, fit_mtx=fitness_mtx, v_range=(0.0, 30.0), cmap='Blues')
    colormap('fr-pv', data_fr_pv, lvls_g, lvls_bg, criteria_fr[0], criteria_fr[1], v_range=(0.0, 50.0), cmap='Blues')
    colormap('fr-som', data_fr_som, lvls_g, lvls_bg, criteria_fr[0], criteria_fr[1], v_range=(0.0, 50.0), cmap='Blues')
