# DSC-250-LMTutor
LMTutor: LLM based bot to answer all academic queries

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

## Currently supported datasets
1. `dsc250` - DSC 250 dataset. This is included in the `data` folder.
2. [SQuAD 1.0](https://rajpurkar.github.io/SQuAD-explorer/) - Stanford Question Answering Dataset. This can be installed by setting `--dataset squad` in the above command. Dataset can be explored on HuggingFace [here](https://huggingface.co/datasets/squad).
3. 

Note: For debugging purposes, you can set `--debug` flag to prepare a small subset of the dataset.
