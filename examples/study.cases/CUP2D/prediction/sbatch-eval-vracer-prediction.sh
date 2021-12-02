#! /usr/bin/env bash

if [ $# -lt 1 ] ; then
	echo "Usage: ./sbatch-eval-vracer-prediction.sh RUNNAME"
	exit 1
fi
if [ $# -gt 0 ] ; then
	RUNNAME=$1
fi

# Number of parallel environments
NNODES=1

# Setup run directory and copy necessary files
RUNPATH="${SCRATCH}/korali/${RUNNAME}"
# mkdir -p ${RUNPATH}
cp eval-vracer-prediction ${RUNPATH}
# cp settings.sh ${RUNPATH}
cd ${RUNPATH}

source settings.sh

cat <<EOF >daint_sbatch
#!/bin/bash -l
#SBATCH --job-name="${RUNNAME}"
#SBATCH --output=${RUNNAME}_out_%j.txt
#SBATCH --error=${RUNNAME}_err_%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=$((NNODES+1))
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s929

# export OMP_NUM_THREADS=12

srun ./eval-vracer-prediction ${OPTIONS} -shapes "${OBJECTS}" -nAgents $NAGENTS
EOF

chmod 755 daint_sbatch
sbatch daint_sbatch