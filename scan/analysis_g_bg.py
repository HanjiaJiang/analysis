import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pdb
import matplotlib
matplotlib.rcParams['font.size'] = 20.0

layers = ['L2/3', 'L4', 'L5', 'L6']

lvl_g = np.linspace(4.0, 8.0, 5)
lvl_bg = np.linspace(2.0, 6.0, 5)

# read comma separated data, return as an array
def read_data(name):
    r_arr = []
    if os.path.isfile(name):
        f = open(name, 'r')
        s = f.read()
        f.close()
        for line in s.splitlines():
            r_arr.append(line.split(','))
    return r_arr


def colormap(name, data, lvls1, lvls2, low, high):
    X, Y = np.meshgrid(lvls1, lvls2)
    fig, axs = plt.subplots(4, 1, figsize=(6, 14), sharex=True, sharey=True, constrained_layout=True)
    axs = axs.ravel()
    plot_name = '{}.png'.format(name)
    for k, data_layer in enumerate(data):
        cs = axs[k].contourf(X, Y, data_layer.T)
        #cs.cmap.set_over("orange")
        #cs.cmap.set_under("black")
        if k == 3:
            cbar = fig.colorbar(cs, orientation='horizontal')
            xticklabels = cbar.ax.get_xticklabels()
            cbar.ax.set_xticklabels(xticklabels, rotation=30, fontsize=18)

    fig.savefig(plot_name)
    plt.close()

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # get input and output names
    inputs = sys.argv[1:-1]
    output = sys.argv[-1]
    print('inputs =\n{}'.format(inputs))
    print('output = {}'.format(output))

    # get dimension shape from the last
    input_shape = (len(lvl_g), len(lvl_bg))
    input_shape = tuple(np.insert(input_shape, 0, 4))
    data_a = np.full(input_shape, np.nan)
    data_i = np.full(input_shape, np.nan)
    data_fr = np.full(input_shape, np.nan)

    print(data_a.shape)

    for item in inputs:
        params = tuple(map(int, item.split('/')[1].split('_')))   # get [int,int]
        ai = read_data(os.path.join(item, 'ai.dat'))
        fr = read_data(os.path.join(item, 'fr.dat'))
        # each layer
        if len(ai) == 4:
            for i, layer in enumerate(ai):
                print('ai ', params, layer[0], layer[1])
                data_a[i, lvl_g.tolist().index(params[1]), lvl_bg.tolist().index(params[2])] = float(layer[0])
                data_i[i, lvl_g.tolist().index(params[1]), lvl_bg.tolist().index(params[2])] = float(layer[1])
       # if len(fr) == 13:
        #    for i, layer in enumerate(np.array(fr)[[0, 4, 7, 10]]):
         #       print('fr ', params, layer[0], layer[1])
          #      data_fr[i, params[0], params[1]] = float(layer[0])

    colormap('pair-corr', data_a, lvl_g, lvl_bg, -0.008, 0.008)
    colormap('irreg', data_i, lvl_g, lvl_bg, 0.5, 1.0)
    #colormap('fr', data_fr, lvl_g, lvl_bg, 0.0, 10.0)

    # np.save(output, [])
