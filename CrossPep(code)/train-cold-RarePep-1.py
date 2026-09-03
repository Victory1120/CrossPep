"""
Run RarePep/DrugBAN baseline training on one fixed xlsx split with 5 training seeds.

Expected input directory:
  data_root/
    train.xlsx
    valid.xlsx   # val.xlsx is also accepted
    test.xlsx

This script is for the non-transfer / non-fine-tuning experiment:
  - no pretrained checkpoint is loaded;
  - no module is frozen;
  - all model parameters are trained from random initialization;
  - the same train/valid/test split is reused for each seed;
  - different training seeds change initialization, dataloader shuffle, dropout, etc.

Outputs per seed:
  best_model_epoch_*.pth, best.pth, 190.pth, 200.pth, model_epoch_200.pth,
  train.txt, valid.txt, test.txt, train_markdowntable.txt, valid_markdowntable.txt,
  test_markdowntable.txt, result_metrics.pt, run_info.json, and data_split/*.xlsx.
"""

import argparse
import json
import logging
import os
import random
import shutil
import warnings
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

from configs_virus import get_cfg_defaults
from dataloader import DTIDataset
from models_5 import DrugBAN
from trainer_transfer import Trainer
from utils import custom_collate_fn, mkdir, set_seed

logging.getLogger().setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="RarePep-1 baseline training on one xlsx split with 5 seeds"
    )
    parser.add_argument("--cfg", type=str, default="./configs/DrugBAN.yaml", help="Path to config file")
    parser.add_argument(
        "--data_root",
        type=str,
        default=r"./XXX/data",# if single-cold: please:I:./3-single-cold-unseen-parasites/data/seed16(or seed18)
        help="Directory containing train.xlsx, valid.xlsx/val.xlsx, and test.xlsx",
    )
    parser.add_argument(
        "--training_seeds",
        nargs="+",
        type=int,
        default=[2024, 2025, 2026, 2027, 2028],#  12, 16, 18, 20, 42
        help="Training seeds. The data split is fixed; only training randomness changes.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./result/single-cold/seed16", # if others, please: ./result/XXX
        help="Output root directory.",
    )
    parser.add_argument(
        "--bert_model",
        type=str,
        default="./BioBert",
        help="Local BioBERT model path used for pathogen-description encoding.",
    )
    parser.add_argument(
        "--esm_model",
        type=str,
        default="./ESM2",
        help="Local ESM2 model path used for peptide encoding.",
    )
    parser.add_argument("--device", type=str, default="cuda:2", help="Example: cuda:0, cuda:1, cuda:2, or cpu")
    parser.add_argument("--max_epoch", type=int, default=200, help="Override SOLVER.MAX_EPOCH. Default: 200")  # in-domain:200,single-cold:200,dual-cold:100
    parser.add_argument("--save_epochs", nargs="+", type=int, default=[190, 200], help="Epoch checkpoints to save")
    parser.add_argument("--lr", type=float, default=None, help="Optional override for cfg.SOLVER.LR")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Adam weight_decay for baseline training")
    parser.add_argument("--cache_features", action="store_true", help="Cache encoded train/valid/test dataframes under output_dir/feature_cache")
    parser.add_argument("--excel_sheet", default=0, help="Excel sheet to read. Default: 0. Use sheet name if needed.")
    return parser.parse_args()


def set_all_random_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cfg_set(cfg, key_path: str, value):
    """Set yacs-like config key, e.g. cfg_set(cfg, 'SOLVER.MAX_EPOCH', 200)."""
    try:
        cfg.defrost()
    except Exception:
        pass
    node = cfg
    keys = key_path.split(".")
    for key in keys[:-1]:
        node = getattr(node, key)
    setattr(node, keys[-1], value)
    try:
        cfg.freeze()
    except Exception:
        pass


def _normalize_sheet_name(sheet_arg):
    if isinstance(sheet_arg, int):
        return sheet_arg
    if isinstance(sheet_arg, str) and sheet_arg.isdigit():
        return int(sheet_arg)
    return sheet_arg


def _find_split_file(data_root, split_name: str):
    data_root = Path(data_root)
    candidates = [f"{split_name}.xlsx"]
    if split_name == "valid":
        candidates.append("val.xlsx")
    for filename in candidates:
        path = data_root / filename
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Required xlsx file not found in {data_root}. Tried: {', '.join(candidates)}"
    )


def check_one_split_dir(data_root):
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")
    _find_split_file(data_root, "train")
    _find_split_file(data_root, "valid")
    _find_split_file(data_root, "test")
    return data_root


def read_split_excel(path, sheet_name=0):
    path = Path(path)
    df = pd.read_excel(path, sheet_name=sheet_name)
    required_cols = {"SMILES", "Protein"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns {missing} in {path}. Existing columns: {list(df.columns)}"
        )
    df = df.dropna(subset=["SMILES", "Protein"]).reset_index(drop=True)
    return df


def get_pathogen_embeddings(df_unique, tokenizer_bert, model_bert, device):
    emb_list = []
    for _, row in tqdm(df_unique.iterrows(), total=df_unique.shape[0], leave=False, desc="BioBERT pathogen"):
        encodings = tokenizer_bert(
            str(row["SMILES"]),
            return_tensors="pt",
            padding="max_length",
            max_length=150,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            output = model_bert(**encodings)
            smiles_embeddings = output.last_hidden_state[0, 0, :].detach().cpu().numpy()
            emb_list.append(smiles_embeddings)
    return emb_list


def get_protein_features(p_list, tokenizer_esm, model_esm, device, batch_size=5, max_len=51):
    records = []
    for start in tqdm(range(0, len(p_list), batch_size), leave=False, desc="ESM peptide"):
        original_batch = [str(seq) for seq in p_list[start:start + batch_size]]
        truncated_batch = [seq[:max_len] for seq in original_batch]
        inputs = tokenizer_esm(truncated_batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model_esm(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
        for original_seq, emb in zip(original_batch, embeddings):
            # Keep original sequence as merge key so long sequences do not fail after truncation.
            records.append({"Protein": original_seq, "esm": emb})
    return pd.DataFrame(records)


def encode_one_dataframe(df, tokenizer_bert, model_bert, tokenizer_esm, model_esm, device):
    df = df.copy()
    df["SMILES"] = df["SMILES"].astype(str)
    df["Protein"] = df["Protein"].astype(str)

    pro_list = df["Protein"].drop_duplicates().tolist()
    protein_df = get_protein_features(pro_list, tokenizer_esm, model_esm, device)
    df = pd.merge(df, protein_df, on="Protein", how="left")

    pathogen_df = df.drop_duplicates(subset="SMILES").copy()
    pathogen_df["fcfp"] = get_pathogen_embeddings(pathogen_df, tokenizer_bert, model_bert, device)
    df = pd.merge(df, pathogen_df[["SMILES", "fcfp"]], on="SMILES", how="left")
    return df


def prepare_datasets(data_root, tokenizer_bert, model_bert, tokenizer_esm, model_esm, device, cache_dir=None, excel_sheet=0):
    data_root = Path(data_root)
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_files = {name: cache_dir / f"{name}_encoded.pkl" for name in ["train", "valid", "test"]}
        if all(path.exists() for path in cache_files.values()):
            print(f"Loading encoded feature cache from {cache_dir}")
            df_train = pd.read_pickle(cache_files["train"])
            df_val = pd.read_pickle(cache_files["valid"])
            df_test = pd.read_pickle(cache_files["test"])
            return (
                DTIDataset(df_train.index.values, df_train),
                DTIDataset(df_val.index.values, df_val),
                DTIDataset(df_test.index.values, df_test),
            )

    sheet_name = _normalize_sheet_name(excel_sheet)
    train_file = _find_split_file(data_root, "train")
    valid_file = _find_split_file(data_root, "valid")
    test_file = _find_split_file(data_root, "test")

    print(f"Reading split files from: {data_root}")
    df_train = read_split_excel(train_file, sheet_name=sheet_name)
    df_val = read_split_excel(valid_file, sheet_name=sheet_name)
    df_test = read_split_excel(test_file, sheet_name=sheet_name)

    df_train = encode_one_dataframe(df_train, tokenizer_bert, model_bert, tokenizer_esm, model_esm, device)
    print("train ESM/BioBERT is done!")
    df_val = encode_one_dataframe(df_val, tokenizer_bert, model_bert, tokenizer_esm, model_esm, device)
    print("valid ESM/BioBERT is done!")
    df_test = encode_one_dataframe(df_test, tokenizer_bert, model_bert, tokenizer_esm, model_esm, device)
    print("test ESM/BioBERT is done!")

    if cache_dir is not None:
        df_train.to_pickle(cache_files["train"])
        df_val.to_pickle(cache_files["valid"])
        df_test.to_pickle(cache_files["test"])

    return (
        DTIDataset(df_train.index.values, df_train),
        DTIDataset(df_val.index.values, df_val),
        DTIDataset(df_test.index.values, df_test),
    )


def make_dataloaders(train_dataset, val_dataset, test_dataset, cfg, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    params_train = {
        "batch_size": cfg.SOLVER.BATCH_SIZE,
        "shuffle": True,
        "num_workers": cfg.SOLVER.NUM_WORKERS,
        "drop_last": True,
        "collate_fn": custom_collate_fn,
        "generator": generator,
    }
    params_eval = {
        "batch_size": cfg.SOLVER.BATCH_SIZE,
        "shuffle": False,
        "num_workers": cfg.SOLVER.NUM_WORKERS,
        "drop_last": False,
        "collate_fn": custom_collate_fn,
    }
    return (
        DataLoader(train_dataset, **params_train),
        DataLoader(val_dataset, **params_eval),
        DataLoader(test_dataset, **params_eval),
    )


def copy_split_xlsx_files(data_root, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for split_name in ["train", "valid", "test"]:
        src = _find_split_file(data_root, split_name)
        shutil.copy2(src, Path(dst_dir) / f"{split_name}.xlsx")


def run_one_seed(seed, data_root, train_dataset, val_dataset, test_dataset, args, device):
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    set_all_random_seeds(seed)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg_set(cfg, "SOLVER.SEED", seed)
    cfg_set(cfg, "SOLVER.MAX_EPOCH", args.max_epoch)
    if args.lr is not None:
        cfg_set(cfg, "SOLVER.LR", args.lr)
    if cfg.DA.USE:
        raise ValueError("This script is for non-DA baseline training. Please set DA.USE=False in the config.")

    out_base = Path(args.output_dir) / "Basic"
    mkdir(str(out_base))
    cfg_set(cfg, "RESULT.OUTPUT_DIR", str(out_base))

    training_generator, val_generator, test_generator = make_dataloaders(
        train_dataset, val_dataset, test_dataset, cfg, seed
    )

    model = DrugBAN(**cfg).to(device)
    # No pretrained weights and no freezing: train all modules from random initialization.
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Baseline training | trainable parameters: {n_trainable:,}/{n_total:,} ({100*n_trainable/n_total:.2f}%)")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=args.weight_decay)

    extra_config = dict(cfg)
    extra_config["SAVE_EPOCHS"] = args.save_epochs
    trainer = Trainer(
        seed,
        model,
        opt,
        device,
        training_generator,
        val_generator,
        test_generator,
        opt_da=None,
        discriminator=None,
        experiment=None,
        **extra_config,
    )
    result = trainer.train()

    run_dir = out_base / str(seed)
    copy_split_xlsx_files(data_root, run_dir / "data_split")
    with open(run_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_root": str(data_root),
                "experiment": "Basic/no-transfer/no-fine-tuning",
                "seed": seed,
                "save_epochs": args.save_epochs,
                "max_epoch": args.max_epoch,
                "excel_sheet": args.excel_sheet,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "experiment": "Basic",
        "seed": seed,
        "best_epoch": result.get("best_epoch"),
        "best_val_auroc": result.get("best_val_auroc"),
        "test_auroc": result.get("auroc"),
        "test_auprc": result.get("auprc"),
        "test_F1": result.get("F1"),
        "test_recall": result.get("recall"),
        "test_accuracy": result.get("accuracy"),
        "test_precision": result.get("Precision"),
        "test_loss": result.get("test_loss"),
        "run_dir": str(run_dir),
    }


def main():
    args = parse_args()
    data_root = check_one_split_dir(args.data_root)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"Running on: {device}")
    print(f"Data root: {data_root}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading BioBERT and ESM2 once...")
    tokenizer_bert = AutoTokenizer.from_pretrained(args.bert_model)
    model_bert = AutoModel.from_pretrained(args.bert_model).eval().to(device)
    tokenizer_esm = AutoTokenizer.from_pretrained(args.esm_model)
    model_esm = AutoModel.from_pretrained(args.esm_model).eval().to(device)

    cache_dir = output_dir / "feature_cache" if args.cache_features else None
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        data_root,
        tokenizer_bert,
        model_bert,
        tokenizer_esm,
        model_esm,
        device,
        cache_dir=cache_dir,
        excel_sheet=args.excel_sheet,
    )

    # Free pretrained language models before model training.
    del model_bert, model_esm
    torch.cuda.empty_cache()

    all_rows = []
    start_time = time()
    for seed in args.training_seeds:
        print("\n" + "=" * 100)
        print(f"Start baseline run | seed={seed}")
        print("=" * 100)
        row = run_one_seed(seed, data_root, train_dataset, val_dataset, test_dataset, args, device)
        all_rows.append(row)
        summary_df = pd.DataFrame(all_rows)
        summary_df.to_csv(output_dir / "all_run_results.csv", index=False)
        summary_df.to_excel(output_dir / "all_run_results.xlsx", index=False)

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(output_dir / "all_run_results.csv", index=False)
    summary_df.to_excel(output_dir / "all_run_results.xlsx", index=False)

    metric_cols = ["test_auroc", "test_auprc", "test_F1", "test_recall", "test_accuracy", "test_precision", "test_loss"]
    grouped = summary_df[metric_cols].agg(["mean", "std"])
    grouped.to_csv(output_dir / "summary_basic.csv")
    grouped.to_excel(output_dir / "summary_basic.xlsx")

    print("\nFinished all baseline runs.")
    print(f"Total running time: {round(time() - start_time, 2)} s")
    print(f"Results saved to: {output_dir}")
    print(summary_df)
    print("\nMean ± std:")
    print(grouped)


if __name__ == "__main__":
    main()
