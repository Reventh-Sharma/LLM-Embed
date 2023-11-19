import os

def get_scratch_dir(base_dir="/scratch"):
    """
    The cache directory for expanse has format:
        `/scratch/$USER/$PROJECT_NAME`.
    """

    whoami = os.environ.get("USER")
    assert whoami, "USER environment variable not set"

    # list all projects under /scratch/$USER
    projects = os.listdir(f"{base_dir}/{whoami}")
    assert projects, f"No projects found under {base_dir}/{whoami}"
    expanse_scratch_dir = f"{base_dir}/{whoami}/{projects[0]}"

    return expanse_scratch_dir


def get_cache_dir(base_dir="/scratch"):
    """
    Return the path to the cache directory for this project.
    """
    try:
        cache_dir = get_scratch_dir(base_dir=base_dir)
    except:
        cache_dir = f"~/.cache/"

    return cache_dir

def get_document_folder(base_data_dir, dataset_name, debug=False):
    if debug:
        file_name = os.path.join(base_data_dir, f"{dataset_name}-debug", "train", "documents")
    else:
        file_name = os.path.join(base_data_dir, dataset_name, "train", "documents")
    
    # create directory if it doesn't exist
    os.makedirs(file_name, exist_ok=True)
    return file_name

def get_qa_file(base_data_dir, dataset_name, debug=False):
    if debug:
        file_name = os.path.join(base_data_dir, f"{dataset_name}-debug", "train", "train.tsv")
    else:
        file_name = os.path.join(base_data_dir, dataset_name, "train", "train.tsv")

    # create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    return file_name

def get_vector_file(base_data_dir, dataset_name, debug=False):
    if debug:
        file_name = os.path.join(base_data_dir, f"{dataset_name}-vector-debug")
    else:
        file_name = os.path.join(base_data_dir, f"{dataset_name}-vector")

    # create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    return file_name

def get_context_id_file(base_data_dir, dataset_name, debug=False):
    if debug:
        file_name = os.path.join(base_data_dir, f"{dataset_name}-debug", "train", "context_id.tsv")
    else:
        file_name = os.path.join(base_data_dir, dataset_name, "train", "context_id.tsv")

    # create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    return file_name

