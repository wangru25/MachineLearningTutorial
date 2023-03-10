import numpy as np
import sys
import os

pathname = '/mnt/home/wangru25/Rui_Wang/12_GAN'
if os.path.exists(pathname):
    name = "gan_1"
    filename = name + '.sb'
    qsubfile = open(filename, "w")
    # Define the system parameter for sbatch file
    qsubfile.write("#!/bin/bash \n")
    qsubfile.write("#SBATCH --time=2:00:00 \n")
    qsubfile.write("#SBATCH --nodes=1-3 \n")
    qsubfile.write("#SBATCH --ntasks=5 \n")
    qsubfile.write("#SBATCH --cpus-per-task=2 \n")
    qsubfile.write("#SBATCH --mem-per-cpu=20G \n")
    qsubfile.write("module purge \n")
    qsubfile.write("module load GCC/5.4.0-2.26 OpenMPI/1.10.3-CUDA \n")
    qsubfile.write("module unload Python \n")
    qsubfile.write("source activate ENV \n")
    qsubfile.write("cd /mnt/home/wangru25/ \n")
    qsubfile.write("cd /mnt/home/wangru25/Rui_Wang/12_GAN \n")
    qsubfile.write("python gan.py \n")
    qsubfile.close()
    os.system("sbatch "+ filename)
