import os
import numpy as np

on_server = False
if not on_server:
    import sys
    sys.path.insert(1, '/home/hanjia/Documents/Potjans_2014_selectivity/')
    # import example

simname = 'genetic_hj'
jdf_name = simname + '_batch.jdf'
workingdir = os.getcwd()
output_dir = workingdir + '/output/'
pyf_name = 'run_network.py'

def evaluate_network_corr(conn_map_list):
    l23_corr_list = []

    for i, map in enumerate(conn_map_list):
        # output directory for this parameter combination
        this_output_dir = 'network_{}'.format(i)
        full_output_dir = output_dir + this_output_dir

        # create directory if it doesn't exist yet
        if this_output_dir not in os.listdir(output_dir):
            os.system('mkdir ' + full_output_dir)
            os.system('mkdir ' + full_output_dir + '/data')

        os.chdir(workingdir)

        # copy all the relevant files to the output directory
        os.system('cp run_network.py ' + full_output_dir)
        os.system('cp network.py ' + full_output_dir)
        os.system('cp network_params.py ' + full_output_dir)
        os.system('cp sim_params.py ' + full_output_dir)
        os.system('cp helpers.py ' + full_output_dir)
        os.system('cp stimulus_params.py ' + full_output_dir)
        os.system('cp conn.py ' + full_output_dir)
        os.system('cp functions.py ' + full_output_dir)
        os.system('cp scan_params.py ' + full_output_dir)

        os.chdir(full_output_dir)

        this_pyf_name = full_output_dir + '/' + pyf_name

        # write job description file
        f = open(full_output_dir + '/' + jdf_name, 'w')
        f.write('#!/bin/bash \n')
        # set job name
        f.write('#SBATCH --job-name ' + simname + '_{}'.
                format(i) + '\n')
        # output file to send standard output stream to
        f.write('#SBATCH -o ./data/outfile.txt' + '\n')
        # send error stream to same file
        f.write('#SBATCH -e ./data/errorfile.txt' + '\n')
        # request a total of nodes*processes_per_node for this job
        f.write('#SBATCH --ntasks=1\n')
        f.write('#SBATCH --cpus-per-task=48\n')
        f.write('#SBATCH --ntasks-per-node=1\n')
        # request processor time
        f.write('#SBATCH --time=02:00:00\n')
        f.write('source $HOME/.bashrc\n')
        f.write('conda activate nest\n')
        f.write(
            'source $HOME/hanjia/opt/nest-lognormal-nest/bin/nest_vars.sh\n')
        f.write('python %s\n' % this_pyf_name)
        f.close()

        # submit job
        os.system('sbatch ' + jdf_name)

    return l23_corr_list

