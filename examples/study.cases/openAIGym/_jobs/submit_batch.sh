for E in Ant-v2 HalfCheetah-v2 Hopper-v2 Humanoid-v2 HumanoidStandup-v2 Reacher-v2 Swimmer-v2 Walker2d-v2;  # full run
#for E in Ant-v2 HalfCheetah-v2 Hopper-v2 Walker2d-v2;  # small run 1
#for E in Humanoid-v2 HumanoidStandup-v2 Reacher-v2 Swimmer-v2;  # small run 2
do 
#    for D in "Normal" "Clipped Normal" "Squashed Normal" "Truncated Normal"; 
#    for D in "Normal" "Clipped Normal" 
    for D in "Squashed Normal" 
    do 
        export ENV=$E
        export DIS="$D" 
        export L2=0.0
        ./sbatch-vracer-openAI.sh 
    done; 
done
