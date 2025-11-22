#!/bin/bash
#SBATCH --job-name=Process_large_data
#SBATCH -t 8:00:00
#SBATCH -p fat_rome
#SBATCH -N 1         # 3 nodes
#SBATCH --get-user-env
#SBATCH --exclusive
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

echo "into the Scripts for my project"
export PATH=/home/ybhatti2/miniconda3/bin:$PATH
source activate myenv

export PYTHONPATH=$PYTHONPATH:/home/ybhatti2/HPC_Jupyter/Python_Scripts/
export NUMBER_OF_SAMPLES=50000
export BASE_DIR="/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PD/Emulated_data/Implausibility/Var"
mkdir -p "$BASE_DIR"
VARIABLES=(AOD AAOD ANG AOD_Mode_1 AOD_Mode_2 AOD_Mode_3 SSA AI AOD_Mode_Coarse CDNC CDNC_Filtered REFFL_CT ERF ERFaci ERFari)
#VARIABLES=(CDNC_Filtered)

#python -u Generate_New_Samples_Constraints.py

#wait
echo "Part 4 STARTED"

export NUMBER_OF_SAMPLES=50000 # This is the number for how many emulated i want
export NUMBER_OF_SAMPLES_Newly_Generated=100000 # This is the number for how many emulated i want



#VARIABLES=(AOD AAOD ANG AOD_Mode_1 AOD_Mode_2 AOD_Mode_3 SSA AI AOD_Mode_Coarse CDNC ERF ERFaci ERFari)
#MAX_PARALLEL=${VARIABLES[@]}

for VARIABLE_NAME in "${VARIABLES[@]}"
do
    echo ">>> Running for VARIABLE_NAME=${VARIABLE_NAME}"
    export VARIABLE_NAME

    srun --ntasks=1 --nodes=1 python -u Part_4_Emulated_Constraint.py > log_${VARIABLE_NAME}_part4_constraint.out 2>&1 &

    sleep 5
done 

wait 

echo "✅ All variable emulations completed."

