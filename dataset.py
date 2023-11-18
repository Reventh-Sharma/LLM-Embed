import os
import argparse

from tqdm import tqdm
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="standqa")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to folder containing documents",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="whether to prepare a small subset of the dataset",
    )
    args = parser.parse_args()
    return args


def prepare_squad_dataset(debug=False):
    """
    Prepare dataset: SQuAD
    features: ['id', 'title', 'context', 'question', 'answers']
    Parameters:
        debug: whether to use a small subset of the dataset for debugging
    """
    dataset_name = "squad"
    data_dir = os.path.join(DATA_DIR, dataset_name, "train")
    os.makedirs(os.path.join(data_dir, "documents"), exist_ok=True)

    # load dataset
    print("Preparing SQuAD dataset...")
    dataset = load_dataset("squad", data_dir="data")
    train_data = dataset["train"]
    if debug:
        print("Preparing SQuAD dataset in debug mode...")
        train_data = train_data.select(range(100))

    # Save documents
    contexts = train_data["context"]
    unique_contexts = list(set(contexts))
    context_id = {}
    print("Total Contexts: ", len(contexts), "Unique Contexts: ", len(unique_contexts))
    for i, context in tqdm(enumerate(unique_contexts), total=len(unique_contexts)):
        with open(os.path.join(data_dir, "documents", f"{i}.txt"), "w") as f:
            f.write(context)
        context_id[context] = i

    # Save questions, answers, and their corresponding context ids
    questions = train_data["question"]
    answers = train_data["answers"]
    with open(os.path.join(data_dir, "train.csv"), "w") as f:
        for i, (question, answer, context) in tqdm(
            enumerate(zip(questions, answers, contexts)), total=len(questions)
        ):
            context_id_ = context_id[context]
            f.write(f"{question},{answer['text'][0]},{context_id_}\n")


def prepare_data(dataset_name, base_dir, debug=False):
    """
    Parameters:
        dataset_name: name of the dataset
        debug: whether to use a small subset of the dataset for debugging
    """
    if dataset_name == "squad":
        prepare_squad_dataset(debug)
    elif dataset_name == "mathdial":
        pass
    else:
        raise ValueError("Dataset name not found")


if __name__ == "__main__":
    args = parse_args()
    prepare_data(args.dataset, args.debug)
