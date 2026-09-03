# CrossPep

This repository contains the data and source code for the CrossPep experiments. The experiments are organized into four main parts: in-domain evaluation, transfer learning, single-cold evaluation, and dual-cold evaluation.

## Environment and Pretrained Models

The model requires BioBERT and ESM2 as pretrained input encoders:

- BioBERT is used to encode pathogen text or description features. You can deploy a Hugging Face-compatible BioBERT model locally under `./BioBert`, or modify the model path in the scripts to load an online model.
- ESM2 is used to encode peptide or protein sequence features. This project uses `esm2_t30_150M_UR50D`. For local deployment, place the downloaded model under `./ESM2`; for online loading, use the corresponding Hugging Face model id, such as `facebook/esm2_t30_150M_UR50D`.

Python dependencies are listed in `CrossPep(code)/requirements.txt`:

```bash
pip install -r "CrossPep(code)/requirements.txt"
```

Training scripts are located in `CrossPep(code)`. The command examples below assume they are executed from the project root directory.

## Experiment Modules

### 1. In-domain

Main script:

```bash
python "CrossPep(code)/train-in-domain.py" --cfg "CrossPep(code)/configs/DrugBAN.yaml"
```

The input data for this part are stored in `1-in-domain`.

### 2. Transfer

Main script:

```bash
python "CrossPep(code)/transfer_main_parasite_finetune.py" --cfg "CrossPep(code)/configs/DrugBAN.yaml"
```

The input data for the transfer-learning experiments are stored in `2-transfer`.


### 3. Single-cold

The corresponding data are provided in `3-single-cold-unseen-parasites`.

Related scripts:

```bash
python "CrossPep(code)/train-cold-CrossPep-1.py" --data_root "./3-single-cold-unseen-parasites/data/seed16" --output_dir "./result/single-cold/seed16" --bert_model "./BioBert" --esm_model "./ESM2"
python "CrossPep(code)/train-CrossPep-2-unseen-parasites.py" --data_root "./3-single-cold-unseen-parasites/data" --split_folders 16 18 --pretrained_path "./3-single-cold-unseen-parasites/source-domain-Bacteria+Virus--pth/best_model_epoch_172.pth" --output_dir "./result/single-cold" --bert_model "./BioBert" --esm_model "./ESM2"
python "CrossPep(code)/train-cold-CrossPep-3.py" --data_root "./3-single-cold-unseen-parasites/data/seed16" --output_dir "./result/single-cold" --bert_model_path "./BioBert" --esm_model_path "./ESM2"
```

### 4. Dual-cold

The corresponding data are provided in `4-dual-cold`.

Related scripts:

```bash
python "CrossPep(code)/train-cold-CrossPep-1.py" --data_root "./4-dual-cold/data/dual-cold" --output_dir "./result/dual-cold/CrossPep-1" --bert_model "./BioBert" --esm_model "./ESM2" --max_epoch 100
python "CrossPep(code)/train-CrossPep-2-unseen-parasites.py" --data_root "./4-dual-cold/data" --split_folders dual-cold --output_dir "./result/dual-cold/CrossPep-2" --bert_model "./BioBert" --esm_model "./ESM2" --max_epoch 100
python "CrossPep(code)/train-cold-CrossPep-3.py" --data_root "./4-dual-cold/data/dual-cold" --output_dir "./result/dual-cold/CrossPep-3" --bert_model_path "./BioBert" --esm_model_path "./ESM2" --max_epoch 100
```

Adjust `--device`, `--training_seeds`, `--seeds`, `--save_epochs`, and checkpoint paths according to the target GPU and reproduction setting.
