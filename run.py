import argparse
from model.llm_langchain_tutor import LLMLangChainTutor
from utils import get_cache_dir


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset preparation arguments
    parser.add_argument("--dataset", type=str, default="standqa")

    # vector store arguments
    parser.add_argument(
        "--create_vector_store",
        action="store_true",
        help="If true, create vector store from documents",
    )
    parser.add_argument(
        "--vec_folder",
        type=str,
        default="data/DSC-250-vector/",
        help="Path to folder which contains vector store or where vector store will be saved",
    )
    parser.add_argument(
        "--doc_folder",
        type=str,
        default="data/DSC-250/",
        help="Path to folder containing documents",
    )

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
        "--base_dir",
        type=str,
        default=get_cache_dir(),
        help="Path to folder containing documents",
    )
    parser.add_argument("--llm_device", type=str, default="cuda:0")
    parser.add_argument("--embed_device", type=str, default="cuda:0")
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
    prompt,
    embedding_model,
    llm_model,
    vec_folder,
    create_vector_store=False,
    doc_folder=None,
    llm_device="cuda:0",
    embed_device="cuda:0",
):
    # Prepare dataset
    


    # Create LLMLangChainTutor
    lmtutor = LLMLangChainTutor(
        embedding=embedding_model,
        llm=llm_model,
        embed_device=embed_device,
        llm_device=llm_device,
    )

    # If vector store is not created, load vector store
    if create_vector_store:
        lmtutor.load_document(
            doc_path=doc_folder, glob="*.pdf", chunk_size=400, chunk_overlap=10
        )
        lmtutor.generate_vector_store()
        lmtutor.save_vector_store(vec_folder)
    else:
        lmtutor.load_vector_store(vec_folder)

    lmtutor.conversational_qa_init()
    lmtutor.conversational_qa(user_input=prompt)


if __name__ == "__main__":
    args = parse_args()
    main(**args.__dict__)
