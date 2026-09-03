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
    parser = argparse.ArgumentParser(description="Batch transfer fine-tuning for RarePep/DrugBAN with xlsx split files")
    parser.add_argument("--cfg", type=str, default="./configs/DrugBAN.yaml", help="Path to config file")
    parser.add_argument(
        "--data_root",
        type=str,
        default=r"./3-single-cold-unseen-parasites/data",  # single-cold if dual-cold, please: ./4-dual-cold/data
        help="Root directory containing split folders, e.g. /path/to/root/16/train.xlsx, valid.xlsx, test.xlsx",
    )
    parser.add_argument(
        "--split_folders",
        nargs="+",
        default=["16,18"],  # single-cold:16,18 dual-cold:16
        help="Only these split folders under data_root will be used. Default: 16 18",
    )
    parser.add_argument(
        "--training_seeds",
        nargs="+",
        type=int,
        default=[2026, 2027, 2028],  # 2024, 2025, 2026, 2027, 2028 12, 16, 18, 20, 42
        help="Training seeds. Data split is fixed; only training randomness changes.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["FT-(Pat+Cls)"],
        choices=["FT-Cls", "FT-(Pat+Cls)", "FT-(Pep+Cls)", "Full"],
        help="Fine-tuning strategies to run.",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="./3-single-cold-unseen-parasites/source-domain-Bacteria+Virus--pth/best_model_epoch_172.pth",
        help="Pretrained weight path used before fine-tuning.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./result/single-cold",
        help="Output root directory.",
    )
    parser.add_argument("--bert_model", type=str, default="./BioBert")
    parser.add_argument("--esm_model", type=str, default="./ESM2")
    parser.add_argument("--device", type=str, default="cuda:2", help="Example: cuda:0, cuda:1, cuda:2, or cpu")
    parser.add_argument("--max_epoch", type=int, default=200, help="Override SOLVER.MAX_EPOCH. Default: 200")  # single-cold:200 dual-cold:100
    parser.add_argument("--save_epochs", nargs="+", type=int, default=[90, 100], help="Epoch checkpoints to save, e.g. 190 200")
    parser.add_argument("--lr", type=float, default=None, help="Optional override for cfg.SOLVER.LR")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Adam weight_decay for fine-tuning")
    parser.add_argument("--cache_features", action="store_true", help="Cache encoded train/valid/test dataframes under output_dir/feature_cache")
    parser.add_argument("--excel_sheet", default=0, help="Excel sheet to read. Default: 0, i.e. the first sheet. Use a sheet name if needed.")
    return parser.parse_args()


def set_all_random_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cfg_set(cfg, key_path, value):
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
    """argparse returns strings; convert numeric sheet index strings to int for pandas."""
    if isinstance(sheet_arg, int):
        return sheet_arg
    if isinstance(sheet_arg, str) and sheet_arg.isdigit():
        return int(sheet_arg)
    return sheet_arg


def _find_split_file(split_dir, split_name):
    """Find xlsx split files: train.xlsx, valid.xlsx/test.xlsx. val.xlsx is accepted for valid."""
    split_dir = Path(split_dir)
    candidates = [f"{split_name}.xlsx"]
    if split_name == "valid":
        candidates.append("val.xlsx")
    for filename in candidates:
        path = split_dir / filename
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Required xlsx file not found in {split_dir}. Tried: {', '.join(candidates)}"
    )


def discover_split_dirs(data_root, split_folders):
    root = Path(data_root)
    split_dirs = []

    # If the user directly passes one split folder containing train/valid/test .xlsx files, use it.
    if (root / "train.xlsx").exists() and ((root / "valid.xlsx").exists() or (root / "val.xlsx").exists()) and (root / "test.xlsx").exists():
        return [root]

    for split_name in split_folders:
        split_dir = root / str(split_name)
        if not split_dir.exists():
            raise FileNotFoundError(f"Split folder not found: {split_dir}")
        _find_split_file(split_dir, "train")
        _find_split_file(split_dir, "valid")
        _find_split_file(split_dir, "test")
        split_dirs.append(split_dir)
    return split_dirs


def read_split_excel(path, sheet_name=0):
    path = Path(path)
    df = pd.read_excel(path, sheet_name=sheet_name)
    required_cols = {"SMILES", "Protein"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}. Existing columns: {list(df.columns)}")
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
            # Keep the original Protein key so that merge never fails for sequences longer than max_len.
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


def prepare_datasets(split_dir, tokenizer_bert, model_bert, tokenizer_esm, model_esm, device, cache_dir=None, excel_sheet=0):
    split_dir = Path(split_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_files = {name: cache_dir / f"{split_dir.name}_{name}_encoded.pkl" for name in ["train", "valid", "test"]}
        if all(path.exists() for path in cache_files.values()):
            print(f"Loading cached features for split {split_dir.name}: {cache_dir}")
            dfs = {name: pd.read_pickle(path) for name, path in cache_files.items()}
            return (
                DTIDataset(dfs["train"].index.values, dfs["train"]),
                DTIDataset(dfs["valid"].index.values, dfs["valid"]),
                DTIDataset(dfs["test"].index.values, dfs["test"]),
            )

    print(f"Encoding data split: {split_dir}")
    sheet_name = _normalize_sheet_name(excel_sheet)
    train_file = _find_split_file(split_dir, "train")
    valid_file = _find_split_file(split_dir, "valid")
    test_file = _find_split_file(split_dir, "test")
    df_train = read_split_excel(train_file, sheet_name=sheet_name)
    df_val = read_split_excel(valid_file, sheet_name=sheet_name)
    df_test = read_split_excel(test_file, sheet_name=sheet_name)

    df_train = encode_one_dataframe(df_train, tokenizer_bert, model_bert, tokenizer_esm, model_esm, device)
    print("train esm/BioBERT is done!")
    df_val = encode_one_dataframe(df_val, tokenizer_bert, model_bert, tokenizer_esm, model_esm, device)
    print("valid esm/BioBERT is done!")
    df_test = encode_one_dataframe(df_test, tokenizer_bert, model_bert, tokenizer_esm, model_esm, device)
    print("test esm/BioBERT is done!")

    if cache_dir is not None:
        df_train.to_pickle(cache_files["train"])
        df_val.to_pickle(cache_files["valid"])
        df_test.to_pickle(cache_files["test"])

    return (
        DTIDataset(df_train.index.values, df_train),
        DTIDataset(df_val.index.values, df_val),
        DTIDataset(df_test.index.values, df_test),
    )


def load_pretrained_weights(model, pretrained_path, device):
    if pretrained_path is None or str(pretrained_path).lower() in ["", "none"]:
        print("No pretrained_path provided; training from random initialization.")
        return model
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained weights: {pretrained_path}")
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    return model


def set_requires_grad(module, flag):
    for param in module.parameters():
        param.requires_grad = flag


def apply_finetune_strategy(model, strategy):
    """
    Module mapping in this codebase:
    - PepEncoder: model.protein_extractor
    - PatEncoder: model.drug_extractor
    - Classifier/fusion head: model.bcn + model.mlp_classifier
    """
    for param in model.parameters():
        param.requires_grad = False

    
    if strategy == "FT-Cls":
        train_modules = ["bcn", "mlp_classifier"]
    elif strategy == "FT-(Pat+Cls)":
        train_modules = ["drug_extractor", "bcn", "mlp_classifier"]
    elif strategy == "FT-(Pep+Cls)":
        train_modules = ["protein_extractor", "bcn", "mlp_classifier"]
    elif strategy == "Full":
        train_modules = ["protein_extractor", "drug_extractor", "bcn", "mlp_classifier"]
    else:
        raise ValueError(f"Unknown fine-tuning strategy: {strategy}")

    for name in train_modules:
        if not hasattr(model, name):
            raise AttributeError(f"Model does not have module `{name}` required by strategy `{strategy}`")
        set_requires_grad(getattr(model, name), True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Fine-tuning strategy: {strategy} | trainable parameters: {n_trainable:,}/{n_total:,} ({100*n_trainable/n_total:.2f}%)")
    return model


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


def copy_split_xlsx_files(split_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for split_name in ["train", "valid", "test"]:
        src = _find_split_file(split_dir, split_name)
        # Normalize val.xlsx to valid.xlsx in the output copy.
        shutil.copy2(src, Path(dst_dir) / f"{split_name}.xlsx")


def run_one(seed, strategy, split_dir, train_dataset, val_dataset, test_dataset, args, device):
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
        raise ValueError("This batch script is for transfer fine-tuning strategies. Please set DA.USE=False in the config.")

    split_name = Path(split_dir).name
    safe_strategy = strategy.replace("(", "").replace(")", "").replace("+", "_")
    out_base = Path(args.output_dir) / safe_strategy / f"split_{split_name}"
    mkdir(str(out_base))
    cfg_set(cfg, "RESULT.OUTPUT_DIR", str(out_base))

    training_generator, val_generator, test_generator = make_dataloaders(train_dataset, val_dataset, test_dataset, cfg, seed)

    model = DrugBAN(**cfg).to(device)
    model = load_pretrained_weights(model, args.pretrained_path, device)
    model = apply_finetune_strategy(model, strategy)

    optim_params = [p for p in model.parameters() if p.requires_grad]
    if len(optim_params) == 0:
        raise RuntimeError(f"No trainable parameters for strategy {strategy}")
    opt = torch.optim.Adam(optim_params, lr=cfg.SOLVER.LR, weight_decay=args.weight_decay)

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
    copy_split_xlsx_files(split_dir, run_dir / "data_split")
    with open(run_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "split_dir": str(split_dir),
                "split_name": split_name,
                "strategy": strategy,
                "seed": seed,
                "pretrained_path": args.pretrained_path,
                "save_epochs": args.save_epochs,
                "max_epoch": args.max_epoch,
                "excel_sheet": args.excel_sheet,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    row = {
        "strategy": strategy,
        "split": split_name,
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
    return row


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"Running on: {device}")

    split_dirs = discover_split_dirs(args.data_root, args.split_folders)
    print("Selected split folders:")
    for sd in split_dirs:
        print(f"  - {sd}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading BioBERT and ESM2 once...")
    tokenizer_bert = AutoTokenizer.from_pretrained(args.bert_model)
    model_bert = AutoModel.from_pretrained(args.bert_model).eval().to(device)
    tokenizer_esm = AutoTokenizer.from_pretrained(args.esm_model)
    model_esm = AutoModel.from_pretrained(args.esm_model).eval().to(device)

    cache_dir = output_dir / "feature_cache" if args.cache_features else None
    dataset_cache = {}
    for split_dir in split_dirs:
        dataset_cache[str(split_dir)] = prepare_datasets(
            split_dir,
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
    for strategy in args.strategies:
        for split_dir in split_dirs:
            train_dataset, val_dataset, test_dataset = dataset_cache[str(split_dir)]
            for seed in args.training_seeds:
                print("\n" + "=" * 100)
                print(f"Start run | strategy={strategy} | split={Path(split_dir).name} | seed={seed}")
                print("=" * 100)
                row = run_one(seed, strategy, split_dir, train_dataset, val_dataset, test_dataset, args, device)
                all_rows.append(row)
                summary_df = pd.DataFrame(all_rows)
                summary_df.to_csv(output_dir / "all_run_results.csv", index=False)
                summary_df.to_excel(output_dir / "all_run_results.xlsx", index=False)

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(output_dir / "all_run_results.csv", index=False)
    summary_df.to_excel(output_dir / "all_run_results.xlsx", index=False)

    metric_cols = ["test_auroc", "test_auprc", "test_F1", "test_recall", "test_accuracy", "test_precision", "test_loss"]
    grouped = summary_df.groupby("strategy")[metric_cols].agg(["mean", "std"])
    grouped.to_csv(output_dir / "summary_by_strategy.csv")
    grouped.to_excel(output_dir / "summary_by_strategy.xlsx")
    grouped2 = summary_df.groupby(["strategy", "split"])[metric_cols].agg(["mean", "std"])
    grouped2.to_csv(output_dir / "summary_by_strategy_and_split.csv")
    grouped2.to_excel(output_dir / "summary_by_strategy_and_split.xlsx")

    print("\nFinished all runs.")
    print(f"Total running time: {round(time() - start_time, 2)} s")
    print(f"All-run results: {output_dir / 'all_run_results.xlsx'}")
    print(f"Strategy summary: {output_dir / 'summary_by_strategy.xlsx'}")


if __name__ == "__main__":
    main()
