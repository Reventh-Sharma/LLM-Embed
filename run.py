import argparse
import os
import shutil

import numpy as np

from dataset import prepare_data, get_parsed_data
from model.llm_langchain_tutor import LLMLangChainTutor
from utils import get_cache_dir, get_document_folder, get_vector_file
from metrics import EmbeddingModelMetrics
from loguru import logger
from tqdm import tqdm
import torch


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset preparation arguments
    parser.add_argument("--prepare_dataset", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="squad")

    # conversation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default="what's the course about?",
        help="Prompt to start conversation",
    )
    parser.add_argument("--embedding_model", type=str, default="")
    parser.add_argument("--llm_model", type=str, default="hf_lmsys/vicuna-7b-v1.3")

    # runtime arguments
    parser.add_argument(
        "--base_data_dir",
        type=str,
        default=get_cache_dir(),
        help="Path to folder containing data",
    )
    parser.add_argument("--llm_device", type=int, default=0)
    parser.add_argument("--embed_device", type=int, default=0)
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="whether to prepare a small subset of the dataset",
    )
    parser.add_argument("--ext_type", type=str, default="*.txt")

    parser.add_argument("--hidden_state_id", type=int, default=-1)

    parser.add_argument("--aggregation", type=str, default='mean')

    # parse arguments
    args = parser.parse_args()
    return args


def main(
    dataset_name,
    embedding_model,
    llm_model,
    prompt,
    base_data_dir,
    prepare_dataset=False,
    llm_device="cuda:0" if torch.cuda.is_available() else "cpu",
    embed_device="cuda:0" if torch.cuda.is_available() else "cpu",
    debug=False,
    ext_type="*.txt",
    hidden_state_id=-1,
    aggregation="mean"
):
    # Prepare dataset
    if prepare_dataset:
        logger.info("Preparing dataset...")
        prepare_data(dataset_name, base_data_dir, debug)
    else:
        logger.info("Dataset preparation skipped.")

    # Create LLMLangChainTutor
    lmtutor = LLMLangChainTutor(
        embedding=embedding_model,
        llm=llm_model,
        embed_device=embed_device,
        llm_device=llm_device,
        cache_dir=base_data_dir,
        debug=debug,
        token="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",
        hidden_state_id=hidden_state_id,
        aggregation=aggregation
    )

    # Create vector store if not exists, otherwise load vector store
    doc_folder = get_document_folder(base_data_dir, dataset_name, debug)
    vec_file = get_vector_file(base_data_dir, dataset_name, debug)

    # Remove vec_file if it exists
    if os.path.exists(vec_file):
        shutil.rmtree(
            vec_file
        )  # Temp fix for vec_file bug i.e. if a different models or dataset is loaded then use vec_file associated with that data or model

    # Create vector store if it does not exist
    lmtutor.generate_vector_store(
        doc_folder, vec_file, glob=ext_type, chunk_size=2000, chunk_overlap=10
    )

    # Load dataset
    # Dataset format: [question, answer, context_id]
    logger.info("Loading dataset...")
    dataset = get_parsed_data(dataset_name, base_data_dir=base_data_dir, debug=debug)

    # Initialize instance of EmbeddingModelMetrics
    # iterate over (question, context_id) pairs
    true_label, pred_labels = [], []
    logger.info("Calculating metrics...")
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        question = row["question"]
        doc_id = row["doc_id"]

        # get context from context_id
        relevant_documents = lmtutor.similarity_search_thres(question, k=15)
        relevant_documents_ids = [
            int(doc.metadata["source"].split("/")[-1].split(".")[0])
            for doc in relevant_documents
        ]

        true_label.append(doc_id)
        pred_labels.append(relevant_documents_ids)

    # convert to numpy arrays
    true_label = np.array(true_label)
    pred_labels = np.array(pred_labels)

    # print metrics
    logger.info("Calculating metrics...")
    metrics_calculator = EmbeddingModelMetrics(true_label, pred_labels)
    logger.info(f"Recall@1: {metrics_calculator.calculate_recall(1)}")
    logger.info(f"Recall@5: {metrics_calculator.calculate_recall(5)}")
    logger.info(f"Recall@10: {metrics_calculator.calculate_recall(10)}")
    logger.info(f"Average document rank: {metrics_calculator.calculate_rank()}")

    # # td_rank = np.inf
    #     # if len(np.where(np.array(relevant_documents_ids)==doc_id)[0])>0:
    #     #     td_rank = np.where(np.array(relevant_documents_ids)==doc_id)[0][0] + 1
    #     # true_label_rank.append(td_rank)
    #     # true_label.append(1)

    # # Calculate metrics
    # recall_atk = []
    # for k in range(1, 17, 5):
    #     pred_labels = [1 if td_rank<=k else 0 for td_rank in true_label_rank]
    #     recall_atk.append(np.sum(pred_labels) / np.sum(true_label))

    # print("Compare", len([td_rank for td_rank in true_label_rank if td_rank>0]), len(true_label_rank))
    # avg_discounted_rank = np.mean(np.array([1/(np.log2(td_rank)) for td_rank in true_label_rank if td_rank>0]))
    # avg_recalled_rank = np.mean(np.array([td_rank for td_rank in true_label_rank if td_rank<np.inf]))

    # # Initialize and start conversation
    # lmtutor.conversational_qa_init()
    # output = lmtutor.conversational_qa(user_input=prompt)
    # logger.info(output)


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)
    main(**args.__dict__)
