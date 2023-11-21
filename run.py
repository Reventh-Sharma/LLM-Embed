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
    parser.add_argument("--llm_device", type=str, default=0)
    parser.add_argument("--embed_device", type=str, default=0)
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="whether to prepare a small subset of the dataset",
    )
    parser.add_argument("--ext_type", type=str, default="*.txt")

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
    ext_type="*.txt",
    debug=False,
):
    # Prepare dataset
    if prepare_dataset:
        print("Preparing dataset...")
        prepare_data(dataset_name, base_data_dir, debug)
    else:
        print("Dataset preparation skipped.")

    # Create LLMLangChainTutor
    lmtutor = LLMLangChainTutor(
        embedding=embedding_model,
        llm=llm_model,
        embed_device=embed_device,
        llm_device=llm_device,
        cache_dir=base_data_dir,
        debug=debug,
        token = "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",
    )

    # Create vector store if not exists, otherwise load vector store
    doc_folder = get_document_folder(base_data_dir, dataset_name, debug)
    vec_file = get_vector_file(base_data_dir, dataset_name, debug)

    # TODO: Generate Vector store using Quack and SQL Data

    if os.path.exists(vec_file):
        shutil.rmtree(vec_file) # Temp fix for vec_file bug i.e. if a different models or dataset is loaded then use vec_file associated with that data or model

    if not os.path.exists(vec_file):
        logger.info("Creating vector store...")
        lmtutor.generate_vector_store(
            doc_folder, vec_file, glob=ext_type, chunk_size=400, chunk_overlap=10
        )
    else:
        logger.info("Vector Store already exists. Proceeding to load it")
        lmtutor.load_vector_store(vec_file)

    # Dataset format: [question, answer, context_id]
    dataset = get_parsed_data(dataset_name, base_data_dir=base_data_dir, debug=debug)

    # # Analyze embeddings

    # Initialize instance of EmbeddingModelMetrics

    true_label, predicted_label = [], []
    true_document_rank = []
    # iterate over (question, context_id) pairs
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        question = row["question"]
        doc_id = row["doc_id"]

        # get context from context_id
        relevant_documents = lmtutor.similarity_search_topk(question, k=100)
        relevant_documents_ids = [
            int(doc.metadata["source"].split("/")[-1].split(".")[0])
            for doc in relevant_documents
        ]

        td_rank = np.inf
        if len(np.where(relevant_documents_ids==doc_id)[0])>0:
            print(np.where(relevant_documents_ids==doc_id))
            td_rank = np.where(relevant_documents_ids==doc_id)[0][0] + 1
        true_document_rank.append(td_rank)
        true_label.append(1)

    # Calculate metrics
    recall_atk = []
    for k in range(1, 16, 5):
        predicted_label = [1 if td_rank<=k else 0 for td_rank in true_document_rank]
        recall_atk.append(np.sum(predicted_label) / np.sum(true_label))

    avg_discounted_rank = np.mean([1/np.log2(td_rank) for td_rank in true_document_rank])
    avg_recalled_rank = np.mean([td_rank for td_rank in true_document_rank if td_rank<np.inf])

    metrics_calculator = EmbeddingModelMetrics(true_label, predicted_label)
    precision = metrics_calculator.calculate_precision()
    recall = metrics_calculator.calculate_recall()
    f1_score = metrics_calculator.calculate_f1_score()
    accuracy = metrics_calculator.calculate_accuracy()


    # print metrics
    logger.info(f"Top-k accuracy: {accuracy / len(dataset) * 100.0}%")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1 Score: {f1_score}")
    logger.info(f"Recall@K: {recall_atk}")
    logger.info(f"Average Discounted Rank: {avg_discounted_rank}")
    logger.info(f"Average Recalled Rank: {avg_recalled_rank}")

    # Initialize and start conversation
    lmtutor.conversational_qa_init()
    output = lmtutor.conversational_qa(user_input=prompt)
    logger.info(output)


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)
    main(**args.__dict__)
