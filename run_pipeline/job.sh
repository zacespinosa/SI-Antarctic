#!/bin/bash -l
#PBS -N ensemble
#PBS -A UWAS0118
#PBS -q economy@chadmin1.ib0.cheyenne.ucar.edu
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=1:mpiprocs=1

module load conda
conda activate cenv

python /glade/work/zespinosa/Projects/SI-Antarctic/run_pipeline/run_dataloader.py
