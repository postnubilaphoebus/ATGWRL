#!/bin/bash
#SBATCH --job-name=train_gan_2mio_gdf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=60GB
#SBATCH --nodes=1
#SBATCH --time=14:00:00
python main.py
