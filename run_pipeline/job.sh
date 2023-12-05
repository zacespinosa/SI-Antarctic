#!/bin/bash -l
#PBS -N ensemble
#PBS -A UWAS0118
#PBS -q casper
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=1:mpiprocs=1
#PBS -o run_transformer.o
#PBS -e run_transformer.e

module load conda
conda activate cenv

#python /glade/work/zespinosa/Projects/SI-Antarctic/run_pipeline/run_dataloader.py
python /glade/work/zespinosa/Projects/SI-Antarctic/run_pipeline/run_datatransformer.py
