#!/bin/bash

#SBATCH --job-name=HeTu
#SBATCH --partition=insp-128C4T
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --comment=hetu_ai
#SBATCH --output=log/crossmatch_bdsf_hetu_1.out
#SBATCH --error=log/crossmatch_bdsf_hetu_1.err

python crossmatch_bdsf_hetu_1.py

