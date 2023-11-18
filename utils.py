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