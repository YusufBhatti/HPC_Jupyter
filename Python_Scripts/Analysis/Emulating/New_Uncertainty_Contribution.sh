#!/bin/bash
#SBATCH --job-name=Process_large_data
#SBATCH -t 8:00:00
#SBATCH -p fat_genoa
#SBATCH -N 4         # 3 nodes
#SBATCH --get-user-env
#SBATCH --exclusive
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

echo "into the Scripts for my project"
export PATH=/home/ybhatti2/miniconda3/bin:$PATH
#source activate ESM
#export PATH=/home/ybhatti/miniconda2/bin:$PATH
source activate myenv

export PYTHONPATH=$PYTHONPATH:/home/ybhatti2/HPC_Jupyter/Python_Scripts/
export NUMBER_OF_SAMPLES=60000
export BASE_DIR="/home/ybhatti2/prjs1474/Datasets/PPE_Processed_Data/PD/Emulated_data/Implausibility/Var"
mkdir -p "$BASE_DIR"
VARIABLES=(TAU355 AOD_Mode_1 SSA AI AOD_Mode_Coarse CDNC_Filtered_OCI REFFL_CT_OCI CDNC_Filtered_SPEXone REFFL_CT_SPEXone CDNC_Filtered_HARP ERF ERFaci ERFari )
VARIABLES=( ERF ERFaci ERFari CDNC_OCI_spx TAU355_daily AOD_Mode_1 SSA AI AOD_Mode_Coarse ) #ERF ERFaci ERFari )
#VARIABLES=( CDNC_Filtered_OCI AOD_Mode_1 AI ) #ERF ERFaci ERFari )

echo "Part 4 STARTED"

export NUMBER_OF_SAMPLES=2000000 # This is the number for how many emulated i want
export NUMBER_OF_SAMPLES_Newly_Generated=2000000 # This is the number for how many emulated i want


#python -u Generate_New_Samples_Constraints.py

export NUMBER_OF_SAMPLES=60000 # This is the masked number
#VARIABLES=(AOD TAU355 AAOD ANG AOD_Mode_1 AOD_Mode_2 AOD_Mode_3 SSA AI AOD_Mode_Coarse CDNC CDNC_Filtered REFFL_CT ERF ERFaci ERFari)
#MAX_PARALLEL=${VARIABLES[@]}
count=0
MAX_PARALLEL=${#VARIABLES[@]}
MAX_PARALLEL=2

# for VARIABLE_NAME in "${VARIABLES[@]}"
# do
#     echo ">>> Running for VARIABLE_NAME=${VARIABLE_NAME}"
#     export VARIABLE_NAME
#     #srun --ntasks=1 --nodes=2 python -u Emulating_ERF.py > log_EMULATING_${VARIABLE_NAME}.out 2>&1 &
#     #srun --ntasks=2 python -m mpi4py Emulating_ERF.py > log_EMULATING_${VARIABLE_NAME}.out 2>&1
#     #srun --ntasks=1 python -m mpi4py Emulating_AI_CDNC.py > log_EMULATING_${VARIABLE_NAME}.out 2>&1
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

for VARIABLE_NAME in "${VARIABLES[@]}"
do
   echo ">>> Running for VARIABLE_NAME=${VARIABLE_NAME}"
   export VARIABLE_NAME

#    srun --ntasks=2 python -m mpi4py Part_4_Emulated_New_Uncertainty_Contribution.py > log_Uncertainty_Contribution_${VARIABLE_NAME}_part4.out 2>&1 &
   #python -u Part_4_Emulated_New_Uncertainty_Contribution.py > log_Uncertainty_Contribution_${VARIABLE_NAME}_part4.out 2>&1 &
   python -u Part_4_Emulated_Original_Uncertainty_Contributions.py > log_Uncertainty_Original_Contribution_${VARIABLE_NAME}_part4.out 2>&1 &

#    python -u Part_4_Emulated.py > log_${VARIABLE_NAME}_part4.out 2>&1 &
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

