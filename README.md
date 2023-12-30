# DSC-250-LMTutor

Large language models (LLMs) trained on massive internet-scale datasets are increasingly being used to improve human efficiency on tasks including knowledge retrieval, coding, and teaching. These models provide an out-of-box method to boost performance on tasks that previously required manual collection of large amounts of domain-specific data. Recently, \textit{LMTutor} has been proposed as a platform to improve question-answering capability from pre-defined documents by leveraging LLMs. However, LMTutor requires training an embedding model on a limited dataset to select relevant documents for LLM prompt, limiting the embedding model's capability to capture intent from a diverse set of question-document pairs. In this project, we  show that LLM embedding can be effectively used to represent diverse range of documents and queries, thereby removing the need to train an embedding model. We demonstrate that embeddings from LLMs such as Vicuna and LLama are better than sentence transformer embeddings when the corpus contains similar documents. We also perform a large scale analysis of embeddings from various hidden layers in LLMs. Our final report is available [here](report.pdf).

# Installation
1. Create and activate conda environment
```bash
conda create -n dsc250 python=3.10 -y
conda activate dsc250
```

2. Install requirements
```bash
pip install -r requirements.txt
```

# Prepare dataset
```bash
python dataset.py --dataset <dataset_name>
```
For example, to prepare SQuAD 1.0 dataset, run
```bash
python dataset.py --dataset squad
```

## Currently supported datasets
1. [SQuAD 1.0](https://rajpurkar.github.io/SQuAD-explorer/) - Stanford Question Answering Dataset. This can be installed by setting `--dataset squad` in the above command. Dataset can be explored on HuggingFace [here](https://huggingface.co/datasets/squad).

Note: For debugging purposes, you can set `--debug` flag to prepare a small subset of the dataset.


# Run code
```bash
python run.py --prompt "When is the office hour during Thanksgiving holiday for DSC250?" --prepare_dataset
```

# Run in debug mode
```bash 
python run.py --prompt "When is the office hour during Thanksgiving holiday for DSC250?" --prepare_dataset --debug
```