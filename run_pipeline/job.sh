#!/bin/bash -l
#PBS -N ocn_mxl
#PBS -A UWAS0118
#PBS -q casper
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=1:mpiprocs=1
#PBS -o mxl_ens_run_transformer.o
#PBS -e mxl_ens_run_transformer.e


module load conda
conda activate cenv

#python /glade/work/zespinosa/Projects/SI-Antarctic/run_pipeline/run_dataloader.py
python /glade/work/zespinosa/Projects/SI-Antarctic/run_pipeline/run_datatransformer.py
# python /glade/work/zespinosa/Projects/SI-Antarctic/pipeline/seaice_transformer.py
