import os
import numpy as np
import microcircuit_tools as tools
fire_rate_roi = np.array([2000.0, 202000.0])
raster_plot_roi = np.array([2100.0, 2300.0])

network_dict = {
    'populations':
        ['L2/3 E', 'L2/3 PV+', 'L2/3 SOM+', 'L23 VIP+', 'L4 E', 'L4 PV+', 'L4 SOM+', 'L5 E', 'L5 PV+', 'L5 SOM+', 'L6 E', 'L6 PV+', 'L6 SOM+'],
    'N_full': np.array(
        [5096, 520, 64, 88, 4088, 288, 64, 3264, 544, 144, 4424, 288, 104])
}

def evaluate(data_path, raster_plot_time_idx, fire_rate_time_idx):
    """ Displays output of the simulation.

    Calculates the firing rate of each population,
    creates a spike raster plot and a box plot of the
    firing rates.

    """

    #tools.fire_rate(
    #    data_path, 'spike_detector',
    #    fire_rate_time_idx[0], fire_rate_time_idx[1]
    #    )

    tools.plot_raster(
        data_path, 'spike_detector',
        raster_plot_time_idx[0], raster_plot_time_idx[1]
        )
    #tools.plot_psth(
    #    data_path, 'spike_detector',
    #    raster_plot_time_idx[0], raster_plot_time_idx[1]
    #)
    tools.boxplot(
        network_dict, data_path
    )


rootdir = os.getcwd()
for dir in filter(os.path.isdir, os.listdir(os.getcwd())):
    data_path = rootdir + '/' + dir +'/data/'
    if os.path.isdir(data_path):
        print(dir)
        para_str = dir
        evaluate(data_path, raster_plot_roi, fire_rate_roi)
