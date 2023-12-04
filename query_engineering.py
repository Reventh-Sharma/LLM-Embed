from tqdm import tqdm


def add_query_instruction(prefix, documents):
    """
    Add a query prefix to each document.

    Parameters:
        documents (list): langchain character splitted documents
        prefix (str): Query prefix
    """
    for document in tqdm(documents):
        document.page_content = prefix + document.page_content

    return documents
    
