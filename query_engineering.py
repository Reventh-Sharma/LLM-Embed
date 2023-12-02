from tqdm import tqdm


def add_query_instruction(documents, instruction, context_column):
    """
    Add a query instruction to each document.

    Parameters:
        documents (list): List to store the modified documents
        instruction (str): Query instruction
        context_column (list): List containing context information
        from the dataset.
    """
    for i, context in tqdm(enumerate(context_column), total=len(context_column)):
        # Check if document already exists
        if i < len(documents):
            documents[i] += f"{instruction}\n\n{context}\n\n"
        else:
            documents.append(f"{instruction}\n\n{context}\n\n")
