#!/bin/bash

#SBATCH --job-name=HeTu
#SBATCH --partition=insp-128C4T
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --comment=hetu_ai
#SBATCH --output=log/add_wcs_all.out
#SBATCH --error=log/add_wcs_all.err


python add_wcs_all.py \
    -c /home/ydai240628/analysis_hetu/file/only_label/output_internimage_0722 \
    -f /groups/hetu_ai/home/share/HeTu/xzj_code/fits \
    -o /home/ydai240628/analysis_hetu/file/wcs_output/output_internimage_0722

