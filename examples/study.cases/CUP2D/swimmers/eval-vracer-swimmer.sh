#!/bin/bash

if [ $# -lt 1 ] ; then
	echo "Usage: ./eval-vracer-swimmer.sh RUNNAME"
	exit 1
fi
if [ $# -gt 0 ] ; then
	RUNNAME=$1
fi

# Set number of nodes here
mpiflags="mpirun -n 2"

if [ ! -z $SLURM_NNODES ]; then
 N=$SLURM_NNODES
 mpiflags="srun -N $N -n $((N+1)) -c 12"
fi

RUNPATH="${SCRATCH}/korali/${RUNNAME}"
cp eval-vracer-swimmer ${RUNPATH}
cd ${RUNPATH}

source settings.sh

set -ux

$mpiflags ./eval-vracer-swimmer ${OPTIONS} -shapes "${OBJECTS}" -nAgents $NAGENTS