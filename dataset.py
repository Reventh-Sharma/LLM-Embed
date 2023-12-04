import argparse
import hashlib
import os
import pandas as pd
import numpy as np
import shutil
from datasets import load_dataset
from tqdm import tqdm
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


def prepare_squad_dataset(
    base_data_dir,
    debug=False,
    split="train",
    use_random_contexts=False,
    random_contexts_count=10000,
    doc_prob=1.0,
):
    """
    Prepare dataset: SQuAD
    features: ['id', 'title', 'context', 'question', 'answers']
    Parameters:
        base_data_dir: path to folder where dataset will be saved
        debug: whether to use a small subset of the dataset for debugging
        split: train or validation
        use_random_contexts: whether to add random unrelated contexts
        random_contexts_count: number of random contexts to add
        doc_prob: probability of mapping a context to a document
    """
    dataset_name = "squad"

    # load dataset
    logger.info(f"Prepared SQuAD dataset in {split} split and doc_prob = {doc_prob}")
    data_split = load_dataset(
        "squad", data_dir=base_data_dir, cache_dir=base_data_dir, split=split
    )
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
        if context in docid_map:
            docid = docid_map[context]
        elif context in context_docid_map:
            docid = docid_map[title]
        else:
            # with probability `doc_prob``, map context correctly to documents
            if np.random.binomial(1, doc_prob):
                if title in docid_map:
                    docid = docid_map[title]
                else:
                    docid = len(documents)
                    docid_map[title] = docid
                    documents.append("")

                # Add context to document
                if not context in context_docid_map:
                    context_docid_map[context] = docid
                    documents[docid] += f"{context}\n\n"
            else:
                docid = len(documents)
                docid_map[context] = docid
                documents.append(f"{context}\n\n")

    # Save documents
    documents_dir = get_document_folder(
        base_data_dir, dataset_name, debug, delete_if_exists=True
    )
    if os.path.exists(documents_dir):
        logger.info(f"Deleting existing documents in {documents_dir}")
        shutil.rmtree(documents_dir)
    os.makedirs(documents_dir)
    for i, document in tqdm(enumerate(documents), total=len(documents)):
        with open(os.path.join(documents_dir, f"{i}.txt"), "w") as f:
            f.write(document)
    logger.info(f"Saving documents to: {documents_dir}")
    logger.info(
        f"Total Documents: {len(documents)}, Total Contexts: {len(set(contexts))}"
    )

    # Add random contexts
    if use_random_contexts:
        logger.info("Adding random unrelated contexts...")
        rand_contexts = generate_random_unrelated_contexts(
            base_data_dir, from_existing_unrelated=True, debug=debug, split=split
        )[:random_contexts_count]
        total_rands = random_contexts_count
        startfrom = len(contexts)
        for i, rand_context in tqdm(enumerate(rand_contexts), total=total_rands):
            with open(os.path.join(documents_dir, f"{startfrom + i}.txt"), "w") as f:
                f.write(rand_context)
        logger.info(f"Total Random Contexts: {total_rands} added")

    # Save questions, answers, and their corresponding context ids
    questions = data_split["question"]
    answers = data_split["answers"]
    qa_file = get_qa_file(base_data_dir, dataset_name, debug)
    logger.info(f"Saving QA pairs to: {qa_file}")
    with open(qa_file, "w") as f:
        for i, (question, answer, context) in tqdm(
            enumerate(zip(questions, answers, contexts)), total=len(questions)
        ):
            if context in context_docid_map:
                context_id_ = context_docid_map[context]
            else:
                context_id_ = docid_map[context]
            f.write(f"{question.strip()}\t{answer['text'][0].strip()}\t{context_id_}\n")

    # # save unique-context-hash and context-id mapping
    # context_id_file = get_context_id_file(base_data_dir, dataset_name, debug)
    # logger.info(f"Saving context_id to: {context_id_file}")
    # with open(context_id_file, "w") as f:
    #     for i, context in enumerate(unique_contexts):
    #         context_hash = hashlib.md5(context.encode("utf-8"))
    #         f.write(f"{context_hash}\t{i}\n")


def prepare_quac_dataset(
    base_data_dir,
    debug=False,
    split="train",
    use_random_contexts=False,
    random_contexts_count=10000,
):
    """
    Prepare QUAC dataset
    Parameters:
        base_data_dir: path to folder where dataset will be saved
        debug: whether to use a small subset of the dataset for debugging
        split: train or validation
        use_random_contexts: whether to add random unrelated contexts
        random_contexts_count: number of random contexts to add
    """
    dataset_name = "quac"

    # Load your QUAC dataset
    # Adjust the following line based on your actual dataset structure
    logger.info("Loading QUAC dataset...")
    data_split = load_dataset("quac", split=split, cache_dir=base_data_dir)

    logger.info(f"Preparing QUAC dataset in {split} split...")
    if debug:
        logger.info("Preparing QUAC dataset in debug mode...")
        data_split = data_split.select(range(2000))

    # Extract necessary information for query engineering
    wikipedia_page_titles = data_split["wikipedia_page_title"]
    backgrounds = data_split["background"]
    sectiontitles = data_split["section_title"]
    contexts = data_split["context"]

    questions = data_split["questions"]
    answers = data_split["orig_answers"]

    # Save docs with query instruction
    documents = []
    docid_map = {}  # maps document titles to doc ids

    logger.info("Preparing documents...")
    for i, (wikipedia_page_title, background, sectiontitle, context, question, answer) in tqdm(
            enumerate(
                zip(wikipedia_page_titles, backgrounds, sectiontitles, contexts, questions, answers)),
            total=len(questions)
    ):
        # Extract the needed info
        title = wikipedia_page_title+":"+sectiontitle

        document = f"{wikipedia_page_title}:\n{background}\n\n{sectiontitle}:\n{context}"

        if title in docid_map:
            docid = docid_map[title]
        else:
            docid = len(documents)
            docid_map[title] = docid
            documents.append(document)
        
    documents_dir = get_document_folder(base_data_dir=base_data_dir, dataset_name=dataset_name, debug=debug, delete_if_exists=True)
    logger.info(f"Saving documents to: {documents_dir}")
    for i, document in tqdm(enumerate(documents), total=len(documents)):
        with open(os.path.join(documents_dir, f"{i}.txt"), "w") as f:
            f.write(document)
    logger.info(f"Total Documents: {len(documents)}")

    # Add random contexts
    if use_random_contexts:
        logger.info("Adding random unrelated contexts...")
        rand_contexts = generate_random_unrelated_contexts(base_data_dir, from_existing_unrelated=True, debug=debug, split=split)[:random_contexts_count]
        total_rands = random_contexts_count
        startfrom = len(contexts)
        for i, rand_context in tqdm(enumerate(rand_contexts), total=total_rands):
            with open(os.path.join(documents_dir, f"{startfrom + i}.txt"), "w") as f:
                f.write(rand_context)
        logger.info(f"Total Random Contexts: {total_rands} added")

    # Save QA pairs
    qa_file = get_qa_file(base_data_dir, dataset_name, debug)
    logger.info(f"Saving QA pairs to: {qa_file}")
    with open(qa_file, "w") as f:
        for i, (question, answer, wikipedia_page_title, sectiontitle) in tqdm(enumerate(zip(questions, answers, wikipedia_page_titles, sectiontitles)), total=len(questions)):
            context_id = docid_map[wikipedia_page_title+":"+sectiontitle]
            # Questions sometime lack context (e.g. What happened in 1893?, multiple things could have happened hence title is required), we add this context as wikipedia page title
            final_question = f'In regards to {wikipedia_page_title}:'
            final_answer = '' 
            # There are multiple questions in each row, we combine them together as a single question. Similarly multiple answers are combined together as a single answer
            for i, each_question in enumerate(question):
                if answer['texts'][i] != "CANNOTANSWER":
                    final_question += each_question.strip() + " "
                    final_answer += answer['texts'][i].strip() + " "
                    f.write(f"{each_question.strip()}\t{answer['texts'][i].strip()}\t{context_id}\n")
            f.write(f"{final_question.strip()}\t{final_answer.strip()}\t{context_id}\n")


def prepare_trivia_dataset(base_data_dir, debug=False, split="train", use_random_contexts=False, random_contexts_count=10000):
    """
    Prepare trivia_qa dataset
    Parameters:
        base_data_dir: path to folder where dataset will be saved
        debug: whether to use a small subset of the dataset for debugging
        split: train or validation
        use_random_contexts: whether to add random unrelated contexts
        random_contexts_count: number of random contexts to add
    """
    dataset_name = "trivia_qa"

    # Load your trivia dataset
    # Adjust the following line based on your actual dataset structure
    logger.info("Loading trivia dataset...")
    data_split = load_dataset("trivia_qa", name="rc", split=split,
                           cache_dir=base_data_dir)

    logger.info(f"Loaded trivia_qa dataset for {split} split...")

    if debug:
        logger.info("Preparing trivia_qa dataset in debug mode...")
        data_split = data_split.select(range(2000))

    # Save questions, answers, and their corresponding context ids
    questions = data_split["question"]
    question_ids = data_split["question_id"]
    context_ids = []
    questions_withdocuments = []
    answers_withdocuments = []

    # Save docs
    documents = []
    docid_map = {}  # maps document titles to doc ids
    context_docid_map = {}  # maps contexts to doc ids


    logger.info("Preparing documents...")
    for i, (question, entity_pages, answer) in tqdm(
            enumerate(zip(questions, data_split["entity_pages"], data_split["answer"])),
            total=len(questions)
    ):
        # Extract the needed info from entity_pages
        title = ' '.join(entity_pages["title"])
        wiki_context = ' '.join(entity_pages["wiki_context"])
        answer_text = answer['value']
        # Title, wiki_context and answer_text should not be empty
        if len(title)>0 and len(wiki_context)>0 and len(answer_text)>0:
            # Check if document already exists
            questions_withdocuments.append(question)
            answers_withdocuments.append(answer_text)
            if title in docid_map:
                docid = docid_map[title]
            else:
                docid = len(documents)
                docid_map[title] = docid
                documents.append("")

            # Add context to document
            if wiki_context not in context_docid_map:
                context_docid_map[wiki_context] = docid
                documents[docid] += f"{wiki_context}\n\n"

            context_ids.append(context_docid_map[wiki_context])

    # Save documents
    documents_dir = get_document_folder(
        base_data_dir, dataset_name, debug, delete_if_exists=True
    )
    logger.info(f"Saving documents to: {documents_dir}")
    for i, document in tqdm(enumerate(documents), total=len(documents)):
        with open(os.path.join(documents_dir, f"{i}.txt"), "w") as f:
            f.write(document)
    logger.info(
        f"Total Documents: {len(documents)}, Total Contexts: {len(context_docid_map)}"
    )

    # Add random contexts
    if use_random_contexts:
        logger.info("Adding random unrelated contexts...")
        rand_contexts = generate_random_unrelated_contexts(base_data_dir, from_existing_unrelated=True, debug=debug, split=split)[:random_contexts_count]
        total_rands = random_contexts_count
        startfrom = len(documents)
        for i, rand_context in tqdm(enumerate(rand_contexts), total=total_rands):
            with open(os.path.join(documents_dir, f"{startfrom + i}.txt"), "w") as f:
                f.write(rand_context)
        logger.info(f"Total Random Contexts: {total_rands} added")

    # Save QA pairs
    logger.info("Saving QA pairs...")
    qa_file = get_qa_file(base_data_dir, dataset_name, debug)
    logger.info(f"Saving QA pairs to: {qa_file}")
    with open(qa_file, "w") as f:
        for i, (question, answer_text, context_id) in tqdm(
                enumerate(zip(questions_withdocuments, answers_withdocuments, context_ids)),
                total=len(questions_withdocuments)
        ):
            f.write(f"{question.strip()}\t{answer_text}\t{context_id}\n")
    logger.info(f"Total QA pairs saved: {len(questions_withdocuments)}")


def prepare_data(
    dataset_name,
    base_data_dir,
    debug=False,
    split="train",
    doc_prob=1.0,
    use_random_contexts=False,
    random_contexts_count=10000):
    """
    Parameters:
        dataset_name: name of the dataset
        debug: whether to use a small subset of the dataset for debugging
    """
    if dataset_name == "squad":
        prepare_squad_dataset(
            base_data_dir,
            debug,
            split=split,
            doc_prob=doc_prob,
            use_random_contexts=use_random_contexts,
            random_contexts_count=random_contexts_count
        )
    
    elif dataset_name == "quac":
        prepare_quac_dataset(
            base_data_dir,
            debug,
            split=split,
            use_random_contexts=use_random_contexts,
            random_contexts_count=random_contexts_count
        )

    elif dataset_name == "trivia_qa":
        prepare_trivia_dataset(
            base_data_dir,
            debug,
            split=split,
            use_random_contexts=use_random_contexts,
            random_contexts_count=random_contexts_count
            )
    else:
        raise ValueError("Dataset name not found")


def generate_random_unrelated_contexts(
    base_data_dir,
    from_existing_unrelated=True,
    dataset_name="pubmed_qa",
    name="pqa_artificial",
    debug=False,
    split="train",
):
    """
    Prepare dataset: Unrelated Contexts (Default pubmed_qa), Since our QA documents don't have medical related questions
    Parameters:
        base_data_dir: path to folder where dataset will be saved
        from_existing_unrelated: whether to use existing unrelated contexts or generate new ones using LLM's
        dataset_name: name of the dataset
        debug: whether to use a small subset of the dataset for debugging
        split: train or test
    """

    if from_existing_unrelated:
        try:
            rand_data = load_dataset(dataset_name, name=name, cache_dir=base_data_dir)
            logger.info(
                f"Successfully loaded existing unrelated contexts using {dataset_name}"
            )

        except:
            raise ValueError(
                f"Existing unrelated contexts not found from {dataset_name}, use random context generation using LLMS"
            )

        # Code specific to only pubmed_qa dataset
        rand_data = rand_data[split]
        rand_contexts = rand_data["context"]
        for i, context in tqdm(enumerate(rand_contexts), total=len(rand_contexts)):
            rand_contexts[i] = " ".join(context["contexts"]).replace("\n", " ").strip()
        if debug:
            rand_contexts = rand_contexts[:2000]
            logger.info("Debug mode is active, using only 2000 unrelated contexts")
        return rand_contexts

    else:
        # Random context generation using LLMS can be added later
        return None


def get_parsed_data(dataset_name, base_data_dir, debug=False):
    """
    Parameters:
        dataset_name: name of the dataset
        debug: whether to use a small subset of the dataset for debugging
    """
    if dataset_name == "squad" or dataset_name == "quac" or dataset_name == "trivia_qa":
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
