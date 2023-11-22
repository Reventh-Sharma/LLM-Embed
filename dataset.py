import argparse
import hashlib
import os
import pandas as pd

from datasets import load_dataset
from tqdm import tqdm

from hf_data_prep import load_hf_dataset_to_pandas
from utils import get_cache_dir, get_context_id_file, get_document_folder, get_qa_file
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="standqa")
    parser.add_argument(
        "--base_data_dir",
        type=str,
        default=get_cache_dir(),
        help="Path to folder where dataset will be saved",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="whether to prepare a small subset of the dataset",
    )
    args = parser.parse_args()
    return args


def prepare_squad_dataset(base_data_dir, debug=False):
    """
    Prepare dataset: SQuAD
    features: ['id', 'title', 'context', 'question', 'answers']
    Parameters:
        debug: whether to use a small subset of the dataset for debugging
    """
    dataset_name = "squad"

    # load dataset
    logger.info("Preparing SQuAD dataset...")
    dataset = load_dataset("squad", data_dir=base_data_dir, cache_dir=base_data_dir)
    logger.info("Preparing SQuAD dataset in validation split...")
    train_data = dataset["validation"]
    if debug:
        logger.info("Preparing SQuAD dataset in debug mode...")
        train_data = train_data.select(range(100))

    # Save documents
    context_id = {}
    contexts = train_data["context"]
    unique_contexts = list(set(contexts))
    documents_dir = get_document_folder(base_data_dir, dataset_name, debug)
    logger.info(f"Saving documents to: {documents_dir}")
    for i, context in tqdm(enumerate(unique_contexts), total=len(unique_contexts)):
        with open(os.path.join(documents_dir, f"{i}.txt"), "w") as f:
            f.write(context)
        context_id[context] = i
    logger.info(
        f"Total Contexts: {len(unique_contexts)}, Unique Contexts: {len(context_id)}"
    )

    # Save questions, answers, and their corresponding context ids
    questions = train_data["question"]
    answers = train_data["answers"]
    qa_file = get_qa_file(base_data_dir, dataset_name, debug)
    logger.info(f"Saving QA pairs to: {qa_file}")
    with open(qa_file, "w") as f:
        for i, (question, answer, context) in tqdm(
            enumerate(zip(questions, answers, contexts)), total=len(questions)
        ):
            context_id_ = context_id[context]
            f.write(f"{question.strip()}\t{answer['text'][0].strip()}\t{context_id_}\n")

    # save unique-context-hash and context-id mapping
    context_id_file = get_context_id_file(base_data_dir, dataset_name, debug)
    logger.info(f"Saving context_id to: {context_id_file}")
    with open(context_id_file, "w") as f:
        for i, context in enumerate(unique_contexts):
            context_hash = hashlib.md5(context.encode("utf-8"))
            f.write(f"{context_hash}\t{i}\n")


def prepare_data(dataset_name, base_data_dir, debug=False):
    """
    Parameters:
        dataset_name: name of the dataset
        debug: whether to use a small subset of the dataset for debugging
    """
    if dataset_name == "squad":
        prepare_squad_dataset(base_data_dir, debug)
    elif dataset_name == "quac" or dataset_name == "b-mc2/sql-create-context":
        load_hf_dataset_to_pandas(dataset_name)
    else:
        raise ValueError("Dataset name not found")


def get_parsed_data(dataset_name, base_data_dir, debug=False):
    """
    Parameters:
        dataset_name: name of the dataset
        debug: whether to use a small subset of the dataset for debugging
    """
    if dataset_name == "squad":
        dataset = get_qa_file(base_data_dir, dataset_name, debug)
        df = pd.read_csv(dataset, names=["question", "answer", "doc_id"], sep="\t", on_bad_lines="warn")
        return df
    else:
        raise ValueError("Dataset name not found")


if __name__ == "__main__":
    args = parse_args()
    prepare_data(args.dataset, args.base_data_dir, args.debug)
