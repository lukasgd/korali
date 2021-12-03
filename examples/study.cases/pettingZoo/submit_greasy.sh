#!/bin/bash -l

if [ $# -lt 1 ] ; then
echo "Usage: ./submit_greasy.sh RUNNAME"
exit 1
fi

# create base folder for running and move necessary files
RUNNAME=$2
BASEPATH="${SCRATCH}/korali"
FOLDERNAME=${BASEPATH}/${RUNNAME}
mkdir -p ${FOLDERNAME}
cp submit_greasy.sh settings.sh run-vracer.py run-dvracer.py ${FOLDERNAME}
cp -r _model/ ${FOLDERNAME}
cd ${FOLDERNAME}

# get default values needed for run
source settings.sh

# count number of runs
let NUMNODES=0

# Write continuous env tasks
for env in  Waterworld
do
    for model in 0 
    do
        for run in 0 
        do
            for mpupdate in 0 1
            do
                for multi in 0  
                do
                    for masampling in 0 1 2
                    do

                        RUNFOLDER=${FOLDERNAME}/${env}_${model}_${multi}_${mpupdate}_${masampling}
                        mkdir -p ${RUNFOLDER}
                        cp run-vracer.py ${RUNFOLDER}
                        cp -r _model/ ${RUNFOLDER}
                        cat << EOF >> tasks.txt
[@ ${RUNFOLDER}/ @] python3 run-vracer.py --env "$env" --dis "$DIS" --l2 $L2 --opt $OPT --lr $LR --model '$model' --run $run --multpolicies $multi --mpupdate $mpupdate --masampling $masampling >  log_${run}.txt
EOF
                        let NUMNODES++
                    done
                done
            done
        done
    done
done

# Write discrete env tasks
for env in  Pursuit
do
    for model in 0 
    do
        for run in 0 
        do
            for mpupdate in 0 1
            do
                for multi in 0  
                do
                    for masampling in 0 1 2
                    do

                        RUNFOLDER=${FOLDERNAME}/${env}_${model}_${multi}_${mpupdate}_${masampling}
                        mkdir -p ${RUNFOLDER}
                        cp run-dvracer.py ${RUNFOLDER}
                        cp -r _model/ ${RUNFOLDER}
                        cat << EOF >> tasks.txt
[@ ${RUNFOLDER}/ @] python3 run-dvracer.py --env "$env"  --l2 $L2 --opt $OPT --lr $LR --model '$model' --run $run --multpolicies $multi --mpupdate $mpupdate --masampling $masampling >  log_${run}.txt
EOF
                        let NUMNODES++
                    done
                done
            done
        done
    done
done

# Write and submit sbatch script
cat << EOF > daint_sbatch
#!/bin/bash -l
#SBATCH --account=s929
#SBATCH --job-name=${RUNNAME}
#SBATCH --output=${RUNNAME}_out_%j.txt
#SBATCH --error=${RUNNAME}_err_%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=${NUMNODES}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:0,craynetwork:4

module load GREASY

export CRAY_CUDA_MPS=1
export CUDA_VISIBLE_DEVICES=0
export GPU_DEVICE_ORDINAL=0
export OMP_NUM_THREADS=12

greasy tasks.txt
EOF

sbatch daint_sbatch
