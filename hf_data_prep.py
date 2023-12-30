import random

import pandas as pd
from datasets import list_datasets, load_dataset

datasets_list = list_datasets()


def load_hf_dataset_to_pandas(dataset_name, split="train", config_name=None):
    """
    Load a dataset from Hugging Face using the datasets library and convert it to a Pandas DataFrame.

    Parameters:
    - dataset_name (str): The name of the Hugging Face dataset.
    - split (str): The split of the dataset to load (e.g., 'train', 'test', 'validation'). Default is 'train'.
    - config_name (str or None): The name of the dataset configuration. Default is None.

    Returns:
    - pd.DataFrame: A Pandas DataFrame containing the specified dataset split.
    """
    # Load the dataset with the specified config if provided
    dataset = load_dataset(dataset_name, config_name)

    # Print available splits for better understanding
    available_splits = list(dataset.keys())
    print(f"Available splits: {available_splits}")

    # Access the specified split or default to 'train' if it doesn't exist
    if "train" in available_splits:
        dataset_split = dataset.get(split, dataset.get("train"))
        print("**Train** data is used")
    else:
        random_key = random.choice(list(dataset.keys()))
        dataset_split = dataset.get(split, dataset.get(random_key))
        print(f"**{random_key}** data is used")

    # Convert to Pandas DataFrame
    df = dataset_split.to_pandas()

    return df


# ## Quac Data

# quac_df = load_hf_dataset_to_pandas("quac")
# quac_df

# ## trivia_qa

# trivia_df = load_hf_dataset_to_pandas("trivia", "rc")
# trivia_df
