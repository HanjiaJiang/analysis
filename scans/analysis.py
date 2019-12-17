import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pdb

layers = ['L2/3', 'L4', 'L5', 'L6']

levels = np.linspace(200, 2000, 2)


# read comma separated data, return as an array
def read_data(name):
    f = open(name, 'r')
    s = f.read()
    f.close()
    r_arr = []
    for line in s.splitlines():
        r_arr.append(line.split(','))
    return r_arr


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # get input and output names
    inputs = sys.argv[1:-1]
    output = sys.argv[-1]
    print(inputs)
    print(output)

    # get dimension shape from the last
    input_shape = np.array(inputs[-1].split('/')[1].split('_')).astype(int) + 1
    input_shape = tuple(np.insert(input_shape, 2, 4))
    data_all = np.full(input_shape, np.nan)
    print(data_all.shape)

    for item in inputs:
        params = tuple(map(int, item.split('/')[1].split('_')))   # get [int,int,int,int]
        data_arr = read_data(item)
        # each layer
        for i, layer in enumerate(data_arr):
            print(params, layer[0])
            data_all[params[0], params[1], i, params[2], params[3]] = float(layer[0])
            print(data_all[params[0], params[1], i, params[2], params[3]])
        # print(levels_list)

    X, Y = np.meshgrid(levels, levels)
    for i, data_conn in enumerate(data_all):
        for j, data_stp in enumerate(data_conn):
            fig, axs = plt.subplots(4, 1, figsize=(6, 14), sharex=True, sharey=True,
                                    tight_layout=True)
            axs = axs.ravel()
            print('conn{}_stp{}.png'.format(i, j))
            for k, data_layer in enumerate(data_stp):
                cs = axs[k].contourf(X, Y, data_layer, np.linspace(-0.008, 0.008, 11), extend='both')
                cs.cmap.set_over("orange")
                cs.cmap.set_under("orange")
            fig.savefig('conn{}_stp{}.png'.format(i, j))
            plt.close()

    # np.save(output, [])
