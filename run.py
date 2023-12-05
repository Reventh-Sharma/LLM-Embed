import argparse
import os
import shutil

import numpy as np
import sys

from dataset import prepare_data, get_parsed_data
from model.llm_langchain_tutor import LLMLangChainTutor
from utils import get_cache_dir, get_document_folder, get_vector_file
from metrics import EmbeddingModelMetrics
from loguru import logger
from tqdm import tqdm
import torch

# set logging level
handler = {"sink": sys.stdout, "level": "INFO"}
logger.configure(handlers=[handler])


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset preparation arguments
    parser.add_argument("--prepare_dataset", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="squad")
    parser.add_argument(
        "--dataset_split", type=str, default="train", help="train or validation"
    )
    parser.add_argument(
        "--doc_prob",
        type=float,
        default=1.0,
        help="probability that a context will be added to its corresponding document. If 1.0, all contexts will be added to their corresponding document and if 0.0, no context will be added to their corresponding document and each context will be added as a separate document",
    )

    # conversation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default="what's the course about?",
        help="Prompt to start conversation",
    )
    parser.add_argument("--embedding_model", type=str, default="")
    parser.add_argument("--hidden_state_id", type=int, default=-1)
    parser.add_argument("--aggregation", type=str, default="mean")
    parser.add_argument(
        "--query_choice",
        type=str,
        default=None,
        help="Choice for query prefix to append to each document. Use (None): No prefix is added, (1) `Summarize the following in 10 word` prefix is added",
    )

    parser.add_argument(
        "--use_random_contexts",
        action="store_true",
        default=False,
        help="Add random contexts to each document datset",
    )

    parser.add_argument(
        "--random_contexts_count",
        type=int,
        default=10000,
        help="Number of random contexts to add to each document in the dataset",
    )

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

    parser.add_argument("--tutor", action="store_true", default=False, help="Initialize as tutor to answer prompts if True")

    parser.add_argument("--weblink", default="http://zhiting.ucsd.edu/teaching/dsc250fall2023/lectures.html", help="Source for course files")

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
    aggregation="mean",
    query_choice=None,
    dataset_split="train",
    doc_prob=1.0,
    use_random_contexts=False,
    random_contexts_count=10000,
    tutor = False,
    weblink = "http://zhiting.ucsd.edu/teaching/dsc250fall2023/lectures.html"
):
    # Prepare dataset
    if prepare_dataset:
        logger.info("Preparing dataset...")
        prepare_data(
            dataset_name, base_data_dir, debug, split=dataset_split, doc_prob=doc_prob,
            use_random_contexts=use_random_contexts, random_contexts_count=random_contexts_count,
            weblink=weblink
        )
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
        aggregation=aggregation,
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
        doc_folder,
        vec_file,
        glob=ext_type,
        chunk_size=2000,
        chunk_overlap=0,
        query_choice=query_choice,
    )

    # Load dataset
    # Dataset format: [question, answer, context_id]
    if not tutor:
        logger.info("Loading dataset...")
        dataset = get_parsed_data(dataset_name, base_data_dir=base_data_dir, debug=debug)

        # Initialize instance of EmbeddingModelMetrics
        # iterate over (question, context_id) pairs
        true_label, pred_labels = [], []
        logger.info("Calculating metrics...")
        for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
            question = row["question"]
            if query_choice is not None:
                if query_choice == "1":
                    query_prefix = "Answer this question in 10 words:"
                elif query_choice == "2":
                    query_prefix = "You are a teaching assistant. Answer this question asked by your student:"
                else:
                    query_prefix = ""   
                question = f"{query_prefix} {question}"
            else:
                query_prefix = ""   
                question = f"{query_prefix} {question}"


            doc_id = row["doc_id"]

            # get context from context_id
            relevant_documents = lmtutor.similarity_search_thres(question, k=20)
            relevant_documents_ids = [
                int(doc.metadata["source"].split("/")[-1].split(".")[0])
                for doc in relevant_documents
            ]

            true_label.append(doc_id)
            pred_labels.append(relevant_documents_ids)

        # convert to numpy arrays
        true_label = np.array(true_label)
        pred_labels = np.array(pred_labels)

        # np.save(f"{base_data_dir}/true_label.npy", true_label)
        # np.save(f"{base_data_dir}/pred_labels.npy", pred_labels)

        # print metrics
        logger.info("Calculating metrics...")
        metrics_calculator = EmbeddingModelMetrics(true_label, pred_labels)
        logger.info(f"Recall@1: {metrics_calculator.calculate_recall(1)}")
        logger.info(f"Recall@5: {metrics_calculator.calculate_recall(5)}")
        logger.info(f"Recall@10: {metrics_calculator.calculate_recall(10)}")
        logger.info(f"Average document rank: {metrics_calculator.calculate_rank()}")

    else:
        # Initialize and start conversation
        logger.info(f"Generating response for prompts: {prompt}")
        lmtutor.conversational_qa_init()
        prompts = prompt.split("?")
        for prompt_i in prompts:
            output = lmtutor.conversational_qa(user_input=prompt_i)
            logger.info(output)


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)
    main(**args.__dict__)
