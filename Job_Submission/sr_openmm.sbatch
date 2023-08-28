#!/bin/bash
#SBATCH --job-name OPC3_Bulk1
#SBATCH --out job.slurm.out
#SBATCH --nodes=1
#SBATCH --exclude=node001,node004,node006
#SBATCH --gpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=24:00:00


hostname

module load cuda11.8

# =========================================
# 1. Define directory names [DO NOT CHANGE]
# =========================================
# Get name of the user submitting the job
export PREFIX=`whoami`

# Get name of the temporary directory working directory, physically on the compute-node. If it doesnt exist yet, the script will make it
workdir="/local-scratch/$PREFIX.${SLURM_JOB_NAME}.${SLURM_JOB_ID}"

# Get submit directory (every file/folder below this directory is copied to the compute node)
submitdir="${SLURM_SUBMIT_DIR}"

# Get storage directory (where results will be saved at the end)
storingdir="/gibbs/$PREFIX"

echo "workdir path is:"
echo $workdir
echo "submitdir path is:"
echo $submitdir
echo "storingdir path is:"
echo $storingdir

# ===================================
# 2. Transfer to node [DO NOT CHANGE]
# ===================================

# create/empty the temporary directory on the compute node
if [ ! -d "${workdir}" ]; then
  mkdir -p "${workdir}"
else
  rm -rf "${workdir}"/*
fi

# change current directory to the location of the sbatch command. "submitdir" is somewhere in the home directory on the head node
cd "${submitdir}"

# copy all files/folders in "submitdir" to "workdir"
cp -prf * ${workdir}

# change directory to the temporary directory on the compute-node
cd ${workdir}

# ====================================================
# 3. Computing commands [CHANGE ACCORDING TO YOUR JOB]
# ====================================================

#######################################################################
#                           Python Setup                              #
#---------------------------------------------------------------------#
export PATH="/home/sramsey/miniconda3_node002/envs/openff/bin:$PATH"
#=====================================================================#

#######################################################################
#                           Run Calculation                           #
#---------------------------------------------------------------------#
/home/sramsey/miniconda3_node002/envs/openff/bin/python yourscript.py
#=====================================================================#

# ==================================================================================================
# 4. Function to transfer back to the /gibbs/user (where log-term storage should be) [DO NOT CHANGE]
# ==================================================================================================

# Define clean-up function
function clean_up {
  # - remove log-file on the compute-node, to avoid the one created by SLURM being overwritten
  rm job.slurm.out
  # - delete temporary files from the compute-node, before copying
  # rm -r ...
  # - change directory to where long-term storage is
  cd "${submitdir}"
  # - copy the temporary directory on the compute-node over
  cp -prf "${workdir}"/ .
  # - erase the temporary directory from the compute-node
  rm -rf "${workdir}"/*
  rm -rf "${workdir}"
  # - exit the script
  exit
}

trap 'clean_up' EXIT


