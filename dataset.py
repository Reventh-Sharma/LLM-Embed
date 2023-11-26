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


def prepare_squad_dataset(base_data_dir, debug=False, split="validation"):
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
    logger.info(f"Preparing SQuAD dataset in {split} split...")
    data_split = dataset[split]
    if debug:
        logger.info("Preparing SQuAD dataset in debug mode...")
        data_split = data_split.select(range(2000))

    # Save documents
    documents = []
    docid_map = {}  # maps doscument titles to doc ids
    context_docid_map = {}  # maps contexts to doc ids

    # Get unique documents
    titles = data_split["title"]
    contexts = data_split["context"]
    for i, (context, title) in tqdm(
        enumerate(zip(contexts, titles)), total=len(contexts)
    ):
        # Check if document already exists
        if title in docid_map:
            docid = docid_map[title]
        else:
            docid = len(documents)
            docid_map[title] = docid
            documents.append("")

        # Add context to document
        if not context in context_docid_map:
            context_docid_map[context] = docid
            documents[docid] += f"{context}\n"

    # Save documents
    documents_dir = get_document_folder(base_data_dir, dataset_name, debug, delete_if_exists=True)
    logger.info(f"Saving documents to: {documents_dir}")
    for i, document in tqdm(enumerate(documents), total=len(documents)):
        with open(os.path.join(documents_dir, f"{i}.txt"), "w") as f:
            f.write(document)
    logger.info(
        f"Total Documents: {len(documents)}, Total Contexts: {len(context_docid_map)}"
    )

    # Save questions, answers, and their corresponding context ids
    questions = data_split["question"]
    answers = data_split["answers"]
    qa_file = get_qa_file(base_data_dir, dataset_name, debug)
    logger.info(f"Saving QA pairs to: {qa_file}")
    with open(qa_file, "w") as f:
        for i, (question, answer, context) in tqdm(
            enumerate(zip(questions, answers, contexts)), total=len(questions)
        ):
            context_id_ = context_docid_map[context]
            f.write(f"{question.strip()}\t{answer['text'][0].strip()}\t{context_id_}\n")

    # # save unique-context-hash and context-id mapping
    # context_id_file = get_context_id_file(base_data_dir, dataset_name, debug)
    # logger.info(f"Saving context_id to: {context_id_file}")
    # with open(context_id_file, "w") as f:
    #     for i, context in enumerate(unique_contexts):
    #         context_hash = hashlib.md5(context.encode("utf-8"))
    #         f.write(f"{context_hash}\t{i}\n")


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
        df = pd.read_csv(
            dataset,
            names=["question", "answer", "doc_id"],
            sep="\t",
            on_bad_lines="warn",
        )
        return df
    else:
        raise ValueError("Dataset name not found")


if __name__ == "__main__":
    args = parse_args()
    prepare_data(args.dataset, args.base_data_dir, args.debug)
