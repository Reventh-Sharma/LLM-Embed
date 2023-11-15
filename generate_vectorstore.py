from model.llm_langchain_tutor import LLMLangChainTutor


def generate_vectorstore(doc_folder, vec_folder):
    lmtutor = LLMLangChainTutor(embedding='instruct_embedding', llm='hf_lmsys/vicuna-7b-v1.3', embed_device='cuda:0')
    lmtutor.load_document(doc_path=doc_folder, glob='*.pdf', chunk_size=400, chunk_overlap=10)
    lmtutor.generate_vector_store()
    lmtutor.save_vector_store(vec_folder)


if __name__ == "__main__":
    doc_folder = "data/DSC-250/"
    vec_folder = "data/DSC-250-vector/"
    generate_vectorstore(doc_folder=doc_folder, vec_folder=vec_folder)
