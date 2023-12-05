#!/usr/bin/env bash
# #SBATCH --job-name=train_lmtutor
# #SBATCH --account=ddp390
# #SBATCH --partition=gpu-shared
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=16
# #SBATCH --mem=64G
# #SBATCH --gpus=1
# #SBATCH --time=03:00:00
# #SBATCH --array=0-1
# #SBATCH --output=train-lmtutor.o%j.%N.log
#SBATCH -N 1                                          
#SBATCH --job-name=test1                     
#SBATCH --time=12:00:00                       
#SBATCH --mem=72000                          
#SBATCH --qos=normal                           
#SBATCH --gres=gpu:1                            


# declare -xr SINGUALRITY_MODULE='singularitypro/3.5'
# export CACHE_FOLDER='/scratch/'${USER}'/job_'${SLURM_JOBID}

# module purge
# module load "${SINGUALRITY_MODULE}"
# module list
# printenv

# cd /home/ddivyansh/projects/lfcbm

# echo $CUDA_VISIBLE_DEVICES

# python run.py --prepare_dataset --dataset_name "squad" --embedding_model "hf_lmsys/vicuna-7b-v1.3"  --embed_device $CUDA_VISIBLE_DEVICES --hidden_state_id -1
export CUDA_VISIBLE_DEVICES=0


# Loop over the following models and embedding layers
EMBEDING_MODEL_LAYER=("-1" "-2" "-3")
MODEL_LIST=("hf_lmsys/vicuna-7b-v1.3" "hf_meta-llama/Llama-2-7b-chat-hf")
TYPES=("mean" "last_token")
# counter=0

for MODEL in "${MODEL_LIST[@]}"
do
   for LAYER in "${EMBEDING_MODEL_LAYER[@]}"
   do
        for TYPE in "${TYPES[@]}"
        do
            echo "python run.py --prepare_dataset --dataset_name \"squad\" --embedding_model \"${MODEL}\" --embed_device $CUDA_VISIBLE_DEVICES --hidden_state_id ${LAYER} --aggregation \"${TYPE}\""
            python run.py --prepare_dataset --dataset_name "squad" --embedding_model "${MODEL}" --embed_device $CUDA_VISIBLE_DEVICES --hidden_state_id ${LAYER} --aggregation "${TYPE}" --dataset_split "validation" --doc_prob 0.0
            # counter=$((counter+1))
        done
    done

    # token_embeddings
    echo "python run.py --prepare_dataset --dataset_name \"squad\" --embedding_model \"${MODEL}\" --embed_device $CUDA_VISIBLE_DEVICES --hidden_state_id -1 --aggregation \"token_embeddings\""
    python run.py --prepare_dataset --dataset_name "squad" --embedding_model "${MODEL}" --embed_device $CUDA_VISIBLE_DEVICES --hidden_state_id -1 --aggregation "token_embeddings" --dataset_split "validation" --doc_prob 0.0
done



MODEL_LIST=("hf_lmsys/vicuna-7b-v1.3" "hf_meta-llama/Llama-2-7b-chat-hf" "instruct_embedding")
DOC_PROB=(1.0 0.8 0.6 0.4 0.2 0.0)


for MODEL in "${MODEL_LIST[@]}"
do
    for PROB in "${DOC_PROB[@]}"
    do
        echo "python run.py --prepare_dataset --dataset_name \"squad\" --embedding_model \"${MODEL}\" --embed_device $CUDA_VISIBLE_DEVICES --hidden_state_id -1 --aggregation \"mean\" --dataset_split \"validation\" --doc_prob ${PROB}"
        python run.py --prepare_dataset --dataset_name "squad" --embedding_model "${MODEL}" --embed_device $CUDA_VISIBLE_DEVICES --hidden_state_id -1 --aggregation "mean" --dataset_split "validation" --doc_prob ${PROB}
    done
done


# #    if [ $counter -eq $SLURM_ARRAY_TASK_ID ]
#        then
#            echo "Running model: $MODEL with layer: $LAYER on GPU: $CUDA_VISIBLE_DEVICES"
#            python run.py --prepare_dataset --dataset_name "squad" --embedding_model $MODEL --embed_device $CUDA_VISIBLE_DEVICES --hidden_state_id $LAYER --aggregation "mean"
#        fi
#        ((counter++))
#    done
# done


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
