#!/bin/bash
#SBATCH --job-name="v3"
#SBATCH --ntasks 1            # -n
#SBATCH --cpus-per-task=1     # -c
#SBATCH --ntasks-per-node=1   # 
#SBATCH --nodes=1             # -N
#SBATCH -t 00:10:00           # Wall time (hh:mm:ss)
#SBATCH --error="my_job.err"
#SBATCH --output="my_job.output"
#SBATCH --partition=main
## SBATCH --nodelist=hydro


source /etc/profile.d/modules.sh
# module load anaconda
# module load python3 pmix-4.2.2 openmpi-4.0.2 cuda-toolkit
# . ./venv/bin/activate

source ~/.bashrc
mamba activate nrm
# Execute the MPI program
# srun --mpi=pmix python3 mapreduce.py
# taskset --cpu-list $1 python3 main.py
srun python3 main.py $SLURM_ARRAY_TASK_ID