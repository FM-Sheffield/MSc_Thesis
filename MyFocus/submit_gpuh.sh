#! /bin/bash
#
#$ -cwd
#$ -j y
#$ -V
##$ -S /bin/bash

#$ -q all.q@compute-0-9
#$ -q gpu.q


##$ -a 03202000
## pido una placa
##$ -l cpu=1


## ejecuto el binario
hostname
./a.out  
