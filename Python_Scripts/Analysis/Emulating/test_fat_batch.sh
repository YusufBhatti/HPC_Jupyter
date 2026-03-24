#!/bin/bash
#SBATCH --job-name=Process_large_data
#SBATCH -t 7:00:00
#SBATCH -p fat_genoa
#SBATCH -N 1         # 3 nodes
#SBATCH --get-user-env
#SBATCH --exclusive
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

echo "into the Scripts for my project"
export PATH=/home/ybhatti2/miniconda3/bin:$PATH
source activate myenv
#source activate master

export PYTHONPATH=$PYTHONPATH:/home/ybhatti2/HPC_Jupyter/Python_Scripts/
export NUMBER_OF_SAMPLES=60000
export BASE_DIR="/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PD/Emulated_data/Implausibility/Var"
mkdir -p "$BASE_DIR"
VARIABLES=(AOD TAU355 AAOD ANG AOD_Mode_1 AOD_Mode_2 AOD_Mode_3 SSA AI AOD_Mode_Coarse CDNC CDNC_Filtered REFFL_CT ERF ERFaci ERFari)
VARIABLES=(AOD_Mode_1 REFFL_CT)
VARIABLES=(TAU355 AOD_Mode_1 SSA AI AOD_Mode_Coarse CDNC_Filtered_OCI REFFL_CT_OCI CDNC_Filtered_SPEXone REFFL_CT_SPEXone CDNC_Filtered_HARP ERF ERFaci ERFari )
VARIABLES=( TAU355_daily REFFL_CT_OCI )
VARIABLES=( CDNC_OCI_spx )

#python -u Generate_New_Samples.py

wait
#for VARIABLE_NAME in "${VARIABLES[@]}"; do
#    mkdir -p "$BASE_DIR/$VARIABLE_NAME"
#    export VARIABLE_NAME
#    echo "Part 1 for VARIABLE_NAME=${VARIABLE_NAME}"
#    srun --ntasks=1 --nodes=1 python -u Part_1_Emulated.py > log_${VARIABLE_NAME}_part1.out 2>&1 &

#done
wait

echo "Part 1 FINISHED"

# --- Variables to emulate ---

MAX_PARALLEL=${#VARIABLES[@]}
MAX_PARALLEL=1

#MAX_PARALLEL=1   # how many variables to run at once
count=0
echo "Part 2 STARTED"

# for VARIABLE_NAME in "${VARIABLES[@]}"
# do
#     echo ">>> Running for VARIABLE_NAME=${VARIABLE_NAME}"
#     export VARIABLE_NAME
#     srun --ntasks=1 --nodes=1 python -u Part_2_Emulated.py > log_${VARIABLE_NAME}_part2.out 2>&1 &

#     ((count++))
#     # When two jobs are running, wait for them to finish before continuing
#     if ((count % MAX_PARALLEL == 0)); then
#         echo ">>> Waiting for current batch to finish..."
#         wait
#         echo ">>> Continuing with next batch..."
#     fi
#     sleep 15
# done
# wait

echo "Part 2 FINISHED"
 echo "Part 3 STARTED"
 #VARIABLES=( CDNC_Filtered_OCI REFFL_CT_HARP REFFL_CT_OCI CDNC_Filtered_SPEXone REFFL_CT_SPEXone CDNC_Filtered_HARP )

 count=0
 for VARIABLE_NAME in "${VARIABLES[@]}"
 do
     echo ">>> Running for VARIABLE_NAME=${VARIABLE_NAME}"
     export VARIABLE_NAME
     srun --ntasks=1 --nodes=1 python -u Part_3_Emulate.py > log_${VARIABLE_NAME}_part3.out 2>&1 &
 #    srun --exclusive -N1 -n1 python -u Part_3_Emulate.py \
 #     > log_${VARIABLE_NAME}.out 2>&1 &

     ((count++))
     # When two jobs are running, wait for them to finish before continuing
     if ((count % MAX_PARALLEL == 0)); then
         echo ">>> Waiting for current batch to finish..."
         wait
         echo ">>> Continuing with next batch..."
     fi
     sleep 15
 done
 wait


echo "Part 3 FINISHED"
echo "Part 4 STARTED"

export NUMBER_OF_SAMPLES=2000000 # This is the number for how many emulated i want
export NUMBER_OF_SAMPLES_Newly_Generated=2000000 # This is the number for how many emulated i want


#python -u Generate_New_Samples_Constraints.py

export NUMBER_OF_SAMPLES=60000 # This is the masked number
#VARIABLES=(AOD TAU355 AAOD ANG AOD_Mode_1 AOD_Mode_2 AOD_Mode_3 SSA AI AOD_Mode_Coarse CDNC CDNC_Filtered REFFL_CT ERF ERFaci ERFari)
#MAX_PARALLEL=${VARIABLES[@]}
count=0

for VARIABLE_NAME in "${VARIABLES[@]}"
do
    echo ">>> Running for VARIABLE_NAME=${VARIABLE_NAME}"
    export VARIABLE_NAME

    srun --ntasks=1 --nodes=1 python -u Part_4_Emulated.py > log_${VARIABLE_NAME}_part4.out 2>&1 &

    ((count++))
    # When two jobs are running, wait for them to finish before continuing
    if ((count % MAX_PARALLEL == 0)); then
        echo ">>> Waiting for current batch to finish..."
        wait
        echo ">>> Continuing with next batch..."
    fi
    sleep 5
done 

wait 

echo "✅ All variable emulations completed."

