import argparse
import os

from dataset import prepare_data
from model.llm_langchain_tutor import LLMLangChainTutor
from utils import get_cache_dir, get_document_folder, get_vector_file


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
    parser.add_argument("--embedding_model", type=str, default="instruct_embedding")
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
    llm_device="cuda:0",
    embed_device="cuda:0",
    debug=False,
):
    # Prepare dataset
    doc_folder = get_document_folder(base_data_dir, dataset_name, debug)
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
    )

    # If vector store is not created, load vector store
    vec_file = get_vector_file(base_data_dir, dataset_name, debug)
    if not os.path.exists(vec_file):
        print("Creating vector store...")
        lmtutor.load_document(
            doc_path=doc_folder, glob="*.txt", chunk_size=400, chunk_overlap=10
        )
        lmtutor.generate_vector_store()
        lmtutor.save_vector_store(vec_file)
    else:
        print("Loading vector store...")
        lmtutor.load_vector_store(vec_file)

    lmtutor.conversational_qa_init()
    output = lmtutor.conversational_qa(user_input=prompt)
    print(output)


if __name__ == "__main__":
    args = parse_args()
    main(**args.__dict__)
