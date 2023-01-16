#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J vgg13gan
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: 20:00 --  maximum 24 hours for GPU-queues right now
#BSUB -W 20:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=16GB]"
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o logs/gpu_%J.out
#BSUB -e logs/gpu_%J.err

#mkdir -p logs
#pip3 install pandas --user
#pip3 install numpy --user
#pip3 install torch --user
#pip3 install matplotlib --user
#pip3 install pillow --user
#pip3 install tdqm --user
#pip3 install jupyter --user

DATAROOT="/work3/s202464/master-thesis/data/"
SYNTH="None"
for DATASET in "Din1" "Din2"
do
    python3 ../src/train_vgg13gan.py --dataset $DATASET --dataroot $DATAROOT --synth_data $SYNTH
done
