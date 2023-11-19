import os

import torch
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import HuggingFacePipeline, OpenAI
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

from model.llm_encoder import LLMBasedEmbeddings

# Pipeline type dictionary
PIPELINE_TYPE = {"lmsys/vicuna-7b-v1.3": "text-generation"}


class LLMLangChainTutor:
    def __init__(
        self,
        doc_loader="dir",
        embedding="openai",
        llm="openai",
        vector_store="faiss",
        openai_key=None,
        token= None,
        embed_device="cuda",
        llm_device="cuda",
        cache_dir="~/.cache",
        debug=False,
    ) -> None:
        """
        Wrapper class for conversational retrieval chain.
        Args:
            doc_loader: Loader for documents. Default is 'dir'.
            embedding: Embedding model to embed document and queries. Default is 'openai'.
            llm: Language model for generating results for query output. Default is 'openai'.
            vector_store: Vector store to store embeddings and associated documents. Default is 'faiss'.
            openai_key: Key for openai, out of scope for now.
            embed_device: Device to use for embedding. Default is 'cuda'.
            llm_device: Device to use for llm. Default is 'cuda'.
            cache_dir: Directory to store cache files. Default is '~/.cache'.
        """
        self.openai_key = openai_key
        self.token = token
        self.llm_name = llm
        self.embed_device = embed_device
        self.llm_device = llm_device
        self.cache_dir = cache_dir
        self.debug = debug

        self._document_loader(doc_loader=doc_loader)
        self._embedding_loader(embedding=embedding)
        self._vectorstore_loader(vector_store=vector_store)
        self._memory_loader()

    def _document_loader(self, doc_loader):
        """
        Args:
            doc_loader: Loader for documents, currently only supports 'dir'.
        """
        if doc_loader == "dir":
            self.doc_loader = DirectoryLoader
        else:
            raise NotImplementedError

    def _embedding_loader(self, embedding):
        """
        This function initializes the embedding model, and is the key part of our project.
        Args:
            embedding: Embedding model to embed document and queries

        Returns:

        """
        if embedding == "openai":
            os.environ["OPENAI_API_KEY"] = self.openai_key
            self.embedding_model = OpenAIEmbeddings()
        elif embedding == "instruct_embedding":
            self.embedding_model = HuggingFaceInstructEmbeddings(
                query_instruction="Represent the query for retrieval: ",
                model_kwargs={
                    "device": self.embed_device,
                },
                encode_kwargs={"batch_size": 32},
                cache_folder=self.cache_dir,
            )
        elif embedding.startswith("hf"): # If an LLM is chosen from HuggingFace
            llm_name = embedding.split("_")[-1]
            self.embedding_model = LLMBasedEmbeddings(llm_name, device = self.llm_device,
                                                      aggr = self.aggregation,
                                                      token = self.token)
        else:
            raise NotImplementedError

    def _vectorstore_loader(self, vector_store):
        """
        Args:
            vector_store: Vector store to store embeddings and associated documents. Default is 'faiss'.
        """
        if vector_store == "faiss":
            self.vector_store = FAISS

    def _memory_loader(self):
        """
        Buffer to store conversation chatbot's converstational history
        """
        self.memory = ChatMessageHistory()

    def _load_document(self, doc_path, glob="*.pdf", chunk_size=400, chunk_overlap=0):
        """
        Loads document from the given path and splits it into chunks of given size and overlap.
        Args:
            doc_path: Path to the document
            glob: Glob pattern to use to find files. Defaults to "**/[!.]*"
               (all files except hidden).
            chunk_size: Size of tokens in each chunk
            chunk_overlap: Number of overlapping chunks within consecutive documents.
        """
        docs = self.doc_loader(
            doc_path,
            glob=glob,
            show_progress=True,
            use_multithreading=True,
            max_concurrency=16,
        ).load()  ### many doc loaders

        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(docs)

    def generate_vector_store(
        self, doc_path, vec_path, glob="*.pdf", chunk_size=400, chunk_overlap=0
    ):
        """
        Generates vector store from the documents and embedding model
        """
        splitted_documents = self._load_document(
            doc_path, glob, chunk_size, chunk_overlap
        )
        self.gen_vectorstore = self.vector_store.from_documents(
            splitted_documents, self.embedding_model
        )
        self.gen_vectorstore.save_local(folder_path=vec_path)

    def load_vector_store(self, vec_path):
        """Load vectors from existing folder_path"""
        self.gen_vectorstore = self.vector_store.load_local(
            folder_path=vec_path, embeddings=self.embedding_model
        )

    def similarity_search_topk(self, query, k=4):
        """Top k-similarity search"""
        retrieved_docs = self.gen_vectorstore.similarity_search(query, k=k)
        return retrieved_docs

    def similarity_search_thres(self, query, thres=0.8):
        """Similarity search with which qualify threshold"""
        retrieval_result = self.gen_vectorstore.similarity_search_with_score(
            query, k=10
        )
        retrieval_result = [d[0] for d in retrieval_result]

        return retrieval_result

    def conversational_qa_init(self):
        """
        Creates a 'qa' object of type ConversationalRetrievalChain, which creates response to given queries based on
        retreived documents from the vector store.
        """
        # mark first conversation as true and reset memory
        self.first_conversation = True
        self._memory_loader()

        # setup conversational qa chain
        if self.llm_name == "openai":
            self.llm = OpenAI(temperature=0)
        elif self.llm_name.startswith("hf"):
            llm_name = self.llm_name.split("_")[-1]
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_name,
                temperature=0.7,
                torch_dtype=torch.float16,
                cache_dir=self.cache_dir,
            ).to(self.llm_device)
            self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
            self.gen_pipe = pipeline(
                PIPELINE_TYPE[llm_name],
                model=self.llm,
                tokenizer=self.tokenizer,
                device=self.llm_device,
                max_new_tokens=512,
                return_full_text=False,
            )
        else:
            raise NotImplementedError

    def conversational_qa(self, user_input):
        """
        Return output of query given a user input.
        Args:
            user_input: User input query
        Returns:
            output: Output of the query using LLM and previous buffers
        """
        FIRST_PROMPT = "A chat between a student user and a teaching assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context.\n"
        PROMPT_TEMPLTATE = "CONTEXT: {context} \n USER: {user_input} \n ASSISTANT:"

        # retrieve relevant documents as context
        context = " \n ".join(
            [each.page_content for each in self.similarity_search_topk(user_input, k=5)]
        )

        # create prompt
        if self.first_conversation:
            prompt = FIRST_PROMPT + PROMPT_TEMPLTATE.format(
                context=context, user_input=user_input
            )
            self.first_conversation = False
        else:
            prompt = (
                self.memory.messages[-1]
                + "\n\n "
                + PROMPT_TEMPLTATE.format(context=context, user_input=user_input)
            )

        # query model and return output
        output = self.gen_pipe(prompt)[0]["generated_text"]
        self.memory.add_message(prompt + output)
        return output
