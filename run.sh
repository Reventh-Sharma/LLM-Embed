#!/usr/bin/env bash
#SBATCH --job-name=train_lmtutor
#SBATCH --account=ddp390
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=train-places365.o%j.%N.log


declare -xr SINGUALRITY_MODULE='singularitypro/3.5'
export CACHE_FOLDER='/scratch/'${USER}'/job_'${SLURM_JOBID}

module purge
module load "${SINGUALRITY_MODULE}"
module list
printenv

# cd /home/ddivyansh/projects/lfcbm

echo $CUDA_VISIBLE_DEVICES

python run.py --prepare_dataset --dataset_name "squad" --embedding_model "hf_lmsys/vicuna-7b-v1.3"  --embed_device $CUDA_VISIBLE_DEVICES
