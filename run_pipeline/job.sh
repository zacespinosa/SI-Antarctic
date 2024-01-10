#!/bin/bash -l
#PBS -N atm_ensemble_transformer
#PBS -A UWAS0118
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=1:mpiprocs=1
#PBS -o atm_ens_run_transformer.o
#PBS -e atm_ens_run_transformer.e


module load conda
conda activate cenv

#python /glade/work/zespinosa/Projects/SI-Antarctic/run_pipeline/run_dataloader.py
python /glade/work/zespinosa/Projects/SI-Antarctic/run_pipeline/run_datatransformer.py
# python /glade/work/zespinosa/Projects/SI-Antarctic/pipeline/seaice_transformer.py
