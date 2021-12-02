for E in Ant-v2 HalfCheetah-v2 Hopper-v2 Humanoid-v2 HumanoidStandup-v2 Reacher-v2 Swimmer-v2 Walker2d-v2;
do 
    for D in "Normal" "Clipped Normal" "Squashed Normal"; #"Truncated Normal"; 
    do 
        export ENV=$E
        export DIS="$D" 
        export L2=0.0
        ./sbatch-vracer-openAI.sh 
    done; 
done
