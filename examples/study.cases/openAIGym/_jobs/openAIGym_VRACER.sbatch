#!/bin/bash -l

#SBATCH --job-name="OpenAI_VRACER"
#SBATCH --output=OpenAI_%j.out
#SBATCH --error=OpenAI_err_%j.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s929

# Read environment, policy distribution, l2 penalization, off-policy target and learning rate
source ../settings.sh

RUNPATH=$SCRATCH/OpenAI/$ENV/$SLURM_JOB_ID
mkdir -p $RUNPATH

pushd ..

cat run-vracer.py

cp run-vracer.py $RUNPATH
cp settings.sh $RUNPATH
cp -r _model/ $RUNPATH

popd

pushd $RUNPATH

OMP_NUM_THREADS=12 python3 run-vracer.py --env "$ENV" --dis "$DIS" --l2 $L2 --opt $OPT --lr $LR

resdir=$(ls -d _result_vracer_*)
DISNW="$(echo -e "${DIS}" | tr -d '[:space:]')"
figureName=vracer_${ENV}_${DISNW}_${L2}_${OPT}_${LR}.png
python3 -m korali.rlview --dir $resdir --output $figureName

popd

date
