#!/bin/bash -l
#PBS -N subsurface
#PBS -A UWAS0118
#PBS -q casper
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=36:mem=109GB
#PBS -o subsurface.o
#PBS -e subsurface.e


module load conda
conda activate cenv

#python /glade/work/zespinosa/Projects/SI-Antarctic/run_pipeline/run_dataloader.py
python /glade/work/zespinosa/Projects/SI-Antarctic/run_pipeline/subsurface_run_datatransformer.py
# python /glade/work/zespinosa/Projects/SI-Antarctic/run_pipeline/run_datatransformer_forecast.py
# python /glade/work/zespinosa/Projects/SI-Antarctic/pipeline/seaice_transformer.py
