#!/usr/bin/env bash
#SBATCH --job-name=train_lmtutor
#SBATCH --account=ddp390
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=train-lmtutor.o%j.%N.log

declare -xr SINGUALRITY_MODULE='singularitypro/3.5'
export CACHE_FOLDER='/scratch/'${USER}'/job_'${SLURM_JOBID}

module purge
module load "${SINGUALRITY_MODULE}"
module list
printenv

# cd /home/ddivyansh/projects/lfcbm

# echo $CUDA_VISIBLE_DEVICES

# python run.py --prepare_dataset --dataset_name "squad" --embedding_model "hf_lmsys/vicuna-7b-v1.3"  --embed_device $CUDA_VISIBLE_DEVICES --embedding_model_layer -1


# Loop over the following models and embedding layers
# MODEL_LIST=("hf_lmsys/vicuna-13b-v1.3" "hf_meta-llama/Llama-2-13b-chat-hf")
# EMBEDING_MODEL_LAYER=("-1")
# EMBEDING_MODEL_LAYER=("-1" "-2" "-3")
# TYPES=("mean")
# counter=0

## All datasets
# DATASET_LIST=("squad" "quac" "trivia_qa")

# for DATASET in "${DATASET_LIST[@]}"
# do
#     for MODEL in "${MODEL_LIST[@]}"
#     do
#         for LAYER in "${EMBEDING_MODEL_LAYER[@]}"
#         do
#             if [ $counter -eq $SLURM_ARRAY_TASK_ID ]
#             then
#                 echo "Running model: $MODEL with layer: $LAYER on GPU: $CUDA_VISIBLE_DEVICES for dataset: $DATASET"
#                 python run.py --prepare_dataset --dataset_name $DATASET --embedding_model $MODEL --embed_device $CUDA_VISIBLE_DEVICES --hidden_state_id $LAYER --aggregation "mean"
#             fi
#             ((counter++))
#         done
#     done
# done

MODEL_LIST=("hf_lmsys/vicuna-7b-v1.3" "hf_meta-llama/Llama-2-7b-chat-hf", "instruct_embedding")
RAND_CONTEXT_COUNT=("2500" "5000" "7500" "10000")

for MODEL in "${MODEL_LIST[@]}"
do
    for RAND_CONTEXT_COUNT in "${RAND_CONTEXT_COUNT[@]}"
    do
        if [ $counter -eq $SLURM_ARRAY_TASK_ID ]
            then
                echo "Running model: $MODEL with layer: $LAYER on GPU: $CUDA_VISIBLE_DEVICES for dataset: $DATASET"
                python run.py --prepare_dataset --dataset_name "squad" --dataset_split "validation" --embedding_model $MODEL --embed_device $CUDA_VISIBLE_DEVICES --hidden_state_id -1 --aggregation "mean" --use_random_contexts --random_contexts_count $RAND_CONTEXT_COUNT --llm_model $MODEL --llm_device $CUDA_VISIBLE_DEVICES
            fi
            ((counter++))
        done
    done
done

QUERY_CHOICE=("1", "2")
MODEL_LIST=("hf_lmsys/vicuna-7b-v1.3" "hf_meta-llama/Llama-2-7b-chat-hf")

for MODEL in "${MODEL_LIST[@]}"
do
    for QR_CHOICE in "${QUERY_CHOICE[@]}"
    do
        if [ $counter -eq $SLURM_ARRAY_TASK_ID ]
                then
                    echo "Running model: $MODEL with layer: $LAYER on GPU: $CUDA_VISIBLE_DEVICES for dataset: $DATASET"
                    python run.py --prepare_dataset --dataset_name "squad" --embedding_model $MODEL --embed_device $CUDA_VISIBLE_DEVICES --hidden_state_id -1 --aggregation "mean" --query_choice $QR_CHOICE
                fi
                ((counter++))
        done
    done
done





## Squad Data
#for MODEL in "${MODEL_LIST[@]}"
#do
#   for LAYER in "${EMBEDING_MODEL_LAYER[@]}"
#   do
#       if [ $counter -eq $SLURM_ARRAY_TASK_ID ]
#       then
#           echo "Running model: $MODEL with layer: $LAYER on GPU: $CUDA_VISIBLE_DEVICES"
#           python run.py --prepare_dataset --dataset_name "squad" --embedding_model $MODEL --embed_device $CUDA_VISIBLE_DEVICES --hidden_state_id $LAYER --aggregation "mean"
#       fi
#       ((counter++))
#   done
#done

# python run.py --prepare_dataset --dataset_name "squad" --embedding_model "instruct_embedding" --embed_device $CUDA_VISIBLE_DEVICES

# python run.py --prepare_dataset --dataset_name "squad" --embedding_model "hf_lmsys/vicuna-7b-v1.5" --embed_device $CUDA_VISIBLE_DEVICES --hidden_state_id -1 --aggregation "mean"
# python run.py --prepare_dataset --dataset_name "squad" --embedding_model "hf_lmsys/vicuna-7b-v1.5" --embed_device $CUDA_VISIBLE_DEVICES --hidden_state_id -1 --aggregation "max"
# python run.py --prepare_dataset --dataset_name "squad" --embedding_model "hf_lmsys/vicuna-7b-v1.5" --embed_device $CUDA_VISIBLE_DEVICES --hidden_state_id -8 --aggregation "mean"
# python run.py --prepare_dataset --dataset_name "squad" --embedding_model "hf_lmsys/vicuna-7b-v1.5" --embed_device $CUDA_VISIBLE_DEVICES --hidden_state_id -16 --aggregation "mean"
# python run.py --prepare_dataset --dataset_name "squad" --embedding_model "hf_lmsys/vicuna-7b-v1.5" --embed_device $CUDA_VISIBLE_DEVICES --hidden_state_id -32 --aggregation "mean"
# python run.py --prepare_dataset --dataset_name "squad" --embedding_model "hf_meta-llama/Llama-2-7b-chat-hf" --embed_device $CUDA_VISIBLE_DEVICES --hidden_state_id -1 --aggregation "mean"
# python run.py --prepare_dataset --dataset_name "squad" --embedding_model "hf_meta-llama/Llama-2-7b-chat-hf" --embed_device $CUDA_VISIBLE_DEVICES --hidden_state_id -1 --aggregation "max"
# python run.py --prepare_dataset --dataset_name "squad" --embedding_model "hf_meta-llama/Llama-2-7b-chat-hf" --embed_device $CUDA_VISIBLE_DEVICES --hidden_state_id -8 --aggregation "mean"
# python run.py --prepare_dataset --dataset_name "squad" --embedding_model "hf_meta-llama/Llama-2-7b-chat-hf" --embed_device $CUDA_VISIBLE_DEVICES --hidden_state_id -16 --aggregation "mean"
# python run.py --prepare_dataset --dataset_name "squad" --embedding_model "hf_meta-llama/Llama-2-7b-chat-hf" --embed_device $CUDA_VISIBLE_DEVICES --hidden_state_id -32 --aggregation "mean"
# python run.py --prepare_dataset --dataset_name "squad" --embedding_model "instruct_embedding" --embed_device $CUDA_VISIBLE_DEVICES
