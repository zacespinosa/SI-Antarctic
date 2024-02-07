#!/bin/bash -l
#PBS -N ocn_sub
#PBS -A UWAS0118
#PBS -q casper
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=36:mem=109GB
#PBS -o sub_ens_run_transformer.o
#PBS -e sub_ens_run_transformer.e


module load conda
conda activate cenv

#python /glade/work/zespinosa/Projects/SI-Antarctic/run_pipeline/run_dataloader.py
python /glade/work/zespinosa/Projects/SI-Antarctic/run_pipeline/subsurface_run_datatransformer.py
# python /glade/work/zespinosa/Projects/SI-Antarctic/pipeline/seaice_transformer.py
