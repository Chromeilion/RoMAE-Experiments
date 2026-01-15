#!/bin/bash -l
#SBATCH -A ict25_esp_0
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 0:30:00       # format: HH:MM:SS
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --output=run_compstats.out
#SBATCH --error=run_compstats.err



cd /leonardo_work/ICT25_ESP/sdigioia


module load profile/deeplrn
module load cuda/12.3 
module load gcc/12.2.0
module load python/3.11.7 
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
module load nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22
module load hdf5/1.14.3--openmpi--4.1.6--gcc--12.2.0-spack0.22 


# Load the virtual environment
# shellcheck source=.env
source /leonardo_work/ICT25_ESP/sdigioia/.Audioenv/bin/activate

cd Audio_data/AudioSet/

python compute_mean_std.py

# Exit the virtualenv for posterity
deactivate
