#!/bin/bash -l
#SBATCH -A ict25_esp_0
#SBATCH -p boost_usr_prod
##SBATCH --qos=boost_qos_dbg
#SBATCH --time 3:00:00       # format: HH:MM:SS
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --output=train_accelerate.out
#SBATCH --error=train_accelerate.err

# --------------------------------------------------------------------
# A script for running RoMA experiments on the Leonardo compute cluster.
#
# Environment variables that must be set:
# VIRTUALENV_LOC : The location of the virtual environment with all dependencies
# EXPERIMENT_NAME : Name of the experiment python package being run
#
# Any arguments passed to this script will be forwarded to the experiment
# --------------------------------------------------------------------
# Load .env file
set -a; source audio.env; set +a

if [[ -z "${VIRTUALENV_LOC}" ]]; then
  echo "Please set the VIRTUALENV_LOC environment variable in the .env file"
  exit
fi
if [[ -z "${EXPERIMENT_NAME}" ]]; then
  echo "Please set the EXPERIMENT_NAME environment variable in the .env file"
  exit
fi


module load profile/deeplrn
module load cuda/12.3 
module load gcc/12.2.0
module load python/3.11.7 
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
module load nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22
module load hdf5/1.14.3--openmpi--4.1.6--gcc--12.2.0-spack0.22 


# Load the virtual environment
# shellcheck source=.env
source "$VIRTUALENV_LOC"

export WANDB_MODE=offline
export WANDB_PROJECT="audio"
export WANDB_ENTITY="rmae"
export WANDB_RUN_GROUP="${EXPERIMENT_NAME}"
export ACCELERATE_CONFIG_FILE="/leonardo_work/ICT25_ESP/sdigioia/accelerate_config/leonardo.yaml"



# All command line arguments passed to the script
ARGS="$@"

# Number of GPUS on each booster node, change depending on the actual hardware
GPUS_PER_NODE=4
# Splitting 32 CPU's between 4 gpus gives 8 cpus per process
CPUS_PER_PROCESS=8

# Number of nodes and processes in the current job
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

# Tell the RoMA how many CPU's each dataloader should spawn
#export ROMA_TRAINER_NUM_DATASET_WORKERS=2

# Use the first node's hostname as the master node address
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=6000

export SLURM_TOTAL_GPUS=$(($SLURM_NNODES * $SLURM_GPUS_ON_NODE))

echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Machine rank: $SLURM_JOBID"
echo "Num processes: $NUM_PROCESSES"
echo "Num machines: $NNODES"

export LAUNCHER="accelerate launch \
    --config_file $ACCELERATE_CONFIG_FILE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_NODEID \
    --num_processes $SLURM_TOTAL_GPUS \
    --num_machines $NNODES \
    --multi_gpu \
    --enable_cpu_affinity \
    --num_cpu_threads_per_process $CPUS_PER_PROCESS \
    --mixed_precision no \
    --module \
    --rdzv_backend c10d \
    --dynamo_mode default \
    --dynamo_backend inductor \
    --dynamo_use_dynamic \
    "

export CMD="$LAUNCHER $EXPERIMENT_NAME $ARGS"

echo "Running command: $CMD"

srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_PATH

# Exit the virtualenv for posterity
deactivate
