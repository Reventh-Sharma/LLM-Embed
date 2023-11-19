import os
import argparse

from tqdm import tqdm
from datasets import load_dataset

from utils import get_cache_dir, get_document_folder, get_qa_file

from hf_data_prep import load_hf_dataset_to_pandas


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
    print("Preparing SQuAD dataset...")
    dataset = load_dataset("squad", data_dir=base_data_dir, cache_dir=base_data_dir)
    train_data = dataset["train"]
    if debug:
        print("Preparing SQuAD dataset in debug mode...")
        train_data = train_data.select(range(100))

    # Save documents
    contexts = train_data["context"]
    unique_contexts = list(set(contexts))
    context_id = {}
    print("Total Contexts: ", len(contexts), "Unique Contexts: ", len(unique_contexts))
    documents_dir = get_document_folder(base_data_dir, dataset_name, debug)
    for i, context in tqdm(enumerate(unique_contexts), total=len(unique_contexts)):
        with open(os.path.join(documents_dir, f"{i}.txt"), "w") as f:
            f.write(context)
        context_id[context] = i

    # Save questions, answers, and their corresponding context ids
    questions = train_data["question"]
    answers = train_data["answers"]
    qa_file = get_qa_file(base_data_dir, dataset_name, debug)
    with open(qa_file, "w") as f:
        for i, (question, answer, context) in tqdm(
            enumerate(zip(questions, answers, contexts)), total=len(questions)
        ):
            context_id_ = context_id[context]
            f.write(f"{question},{answer['text'][0]},{context_id_}\n")


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


if __name__ == "__main__":
    args = parse_args()
    prepare_data(args.dataset, args.base_data_dir, args.debug)
