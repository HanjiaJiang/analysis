#!/bin/bash
#SBATCH -o ./out-gs-plot.txt
#SBATCH -e ./err-gs-plot.txt
#SBATCH --job-name hj-gs-plot
#SBATCH --mem=4G
#SBATCH --time=0
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --partition=hambach,blaustein,hamstein
#SBATCH --mail-type=FAIL # notifications for job done & fail
#SBATCH --mail-user=h.jiang@fz-juelich.de

source activate nest-log
mkdir -p out

snakemake --unlock\

snakemake --jobs 1\
          --cluster-config cluster.json\
          --cluster "sbatch -n {cluster.n} \
                            -o out/gs-plot.{jobid}.out \
                            --cpus-per-task {cluster.nCPUs} \
                            --mem {cluster.mem} \
                            --time {cluster.time} \
                            --partition=hambach,blaustein,hamstein \
                            --mail-type=FAIL \
                            --mail-user=h.jiang@fz-juelich.de"\
          --configfile config.yml\
          --use-conda\
          --rerun-incomplete\
          --jobname "gs-plot.{jobid}"
