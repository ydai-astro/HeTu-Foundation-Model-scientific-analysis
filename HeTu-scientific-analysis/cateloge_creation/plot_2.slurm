#!/bin/bash

#SBATCH --job-name=HeTu
#SBATCH --partition=insp-128C4T
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --comment=hetu_ai
#SBATCH --output=log/bbox_overlap_removal_all.out
#SBATCH --error=log/bbox_overlap_removal_all.err


python bbox_overlap_removal_all.py \
  --input_dir /home/ydai240628/analysis_hetu/file/wcs_output_new/wcs_resnet101\
  --output_dir /home/ydai240628/analysis_hetu/file/bbox_overlap_removal/output_resnet\

