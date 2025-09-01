# HPC_Jupyter
Setting up jupyter notebook using HPC nodes 
run:
sbatch batch

these scripts (batch scripts) will start up a slurm job with n nodes being occupied for python job. Different nodes are used relative to the depth of job and computational resources needed. These jobs also re-write the ssh_connect.sh file, which is useful to login to the respective node which is being used for the python job. 

The port will need to be changed to your login port, which would need to be modified. Then once running . ssh_connect.sh, you can connect to localhost.

