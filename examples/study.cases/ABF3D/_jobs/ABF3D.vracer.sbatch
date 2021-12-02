#!/bin/bash -l
#SBATCH --job-name="ABF3D_VRACER"
#SBATCH --output=ABF3D_VRACER_%j.out
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --account=s929
#SBATCH --partition=normal
#SBATCH --constraint=gpu

date

pushd ..

cat run-vracer.cpp

expDir=$SCRATCH/ABF3D_VRACER/$SLURM_JOB_ID
mkdir -p $expDir
cp run-vracer $expDir
cp -r _config $expDir

popd

pushd $expDir

OMP_NUM_THREADS=12 ./run-vracer
python -m korali.rlview --dir _result_vracer --out vracer.png


popd

date
