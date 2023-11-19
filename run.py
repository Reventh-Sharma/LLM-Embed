import argparse
import os

from dataset import prepare_data, get_parsed_data
from model.llm_langchain_tutor import LLMLangChainTutor
from utils import get_cache_dir, get_document_folder, get_vector_file
from metrics import EmbeddingModelMetrics
from loguru import logger
from tqdm import tqdm


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
    )

    # Create vector store if not exists, otherwise load vector store
    doc_folder = get_document_folder(base_data_dir, dataset_name, debug)
    vec_file = get_vector_file(base_data_dir, dataset_name, debug)
    if not os.path.exists(vec_file):
        logger.info("Creating vector store...")
        lmtutor.generate_vector_store(
            doc_folder, vec_file, glob="*.txt", chunk_size=400, chunk_overlap=10
        )
    else:
        logger.info("Loading vector store...")
        lmtutor.load_vector_store(vec_file)

    # Dataset format: [question, answer, context_id]
    dataset = get_parsed_data(dataset_name, base_data_dir=base_data_dir, debug=debug)

    # Analyze embeddings
    # iterate over (question, context_id) pairs
    accuracy = 0
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        question = row["question"]
        doc_id = row["doc_id"]

        # get context from context_id
        relevant_documents = lmtutor.similarity_search_topk(question)
        relevant_documents_ids = [
            int(doc.metadata["source"].split("/")[-1].split(".")[0])
            for doc in relevant_documents
        ]
        accuracy += 1 if doc_id in relevant_documents_ids else 0

    # print accuracy
    logger.info(f"Top-k accuracy: {accuracy/len(dataset) * 100.0}%")

    # Initialize and start conversation
    lmtutor.conversational_qa_init()
    output = lmtutor.conversational_qa(user_input=prompt)
    logger.info(output)

    # change predicted and true to ground truth
    # true_labels = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]

    # predicted_labels = [1, 1, 0, 1, 0, 0, 1, 1, 0, 0]

    # # Create an instance of EmbeddingModelMetrics
    # metrics_calculator = EmbeddingModelMetrics(true_labels, predicted_labels)

    # # Calculate and print precision, recall, and F1-score
    # precision = metrics_calculator.calculate_precision()
    # recall = metrics_calculator.calculate_recall()
    # f1_score = metrics_calculator.calculate_f1_score()

    # print("Precision:", precision)
    # print("Recall:", recall)
    # print("F1 Score:", f1_score)


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)
    main(**args.__dict__)
