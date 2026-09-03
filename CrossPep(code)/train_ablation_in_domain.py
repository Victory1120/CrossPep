import argparse
import os
from time import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from configs_bacteria import get_cfg_defaults
from dataloader import DTIDataset
from models_5_ablation import SUPPORTED_ABLATIONS, DrugBAN
from trainer import Trainer
from utils import custom_collate_fn, mkdir, set_seed


DEFAULT_SEEDS = [12, 16, 18, 20, 42]
DEFAULT_DATASETS = ["Bacteria", "parasite"]
DEFAULT_ABLATIONS = ["no_pep_encoder", "no_pat_encoder", "no_pam"]


def parse_args():
    parser = argparse.ArgumentParser(description="DrugBAN in-domain ablation training")
    parser.add_argument("--cfg", type=str, default="./configs/DrugBAN.yaml", help="path to config yaml")
    parser.add_argument("--data-root", type=str, default=None, help="path containing Bacteria/parasite data folders")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="datasets to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="random split seeds")
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=DEFAULT_ABLATIONS,
        choices=sorted(SUPPORTED_ABLATIONS),
        help="ablation variants to run",
    )
    parser.add_argument("--output-root", type=str, default="./result/ablation/in-domain/random-splits")
    parser.add_argument("--biobert-path", type=str, default="./BioBert", help="local BioBERT path used for SMILES/text")
    parser.add_argument("--esm2-path", type=str, default="./ESM2", help="local ESM2 path used for peptide sequence")
    parser.add_argument("--cache-dir", type=str, default="./feature_cache/ablation_in_domain")
    parser.add_argument("--device", type=str, default="cuda:2" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-epoch", type=int, default=None, help="override SOLVER.MAX_EPOCH")
    parser.add_argument("--batch-size", type=int, default=None, help="override SOLVER.BATCH_SIZE")
    parser.add_argument("--lr", type=float, default=None, help="override SOLVER.LR")
    parser.add_argument("--num-workers", type=int, default=None, help="override SOLVER.NUM_WORKERS")
    parser.add_argument("--no-save-model", action="store_true", help="save only metrics/tables")
    parser.add_argument("--rebuild-cache", action="store_true", help="recompute BioBERT/ESM2 features")
    return parser.parse_args()


def resolve_data_root(data_root):
    candidates = []
    if data_root:
        candidates.append(data_root)
    candidates.extend(
        [
            "./1-in-domain/1-random-splits",
            "../1-in-domain/1-random-splits",
        ]
    )
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        "Cannot find in-domain data root. Pass --data-root, for example "
        "../1-in-domain/1-random-splits"
    )


def require_transformers():
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required when train/valid/test files do not already contain "
            "`fcfp` and `esm` columns."
        ) from exc
    return AutoTokenizer, AutoModel


def get_text_embeddings(df_unique, tokenizer, model, device):
    embeddings = []
    for _, row in tqdm(df_unique.iterrows(), total=df_unique.shape[0], leave=False, desc="BioBERT"):
        encodings = tokenizer(
            row["SMILES"],
            return_tensors="pt",
            padding="max_length",
            max_length=150,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            output = model(**encodings)
            embeddings.append(output.last_hidden_state[0, 0, :].cpu().numpy())
    return embeddings


def get_protein_features(protein_list, tokenizer, model, device):
    records = []
    data_tmp = [(f"protein{i}", protein[:51]) for i, protein in enumerate(protein_list)]
    for i in tqdm(range(len(data_tmp) // 5 + 1), leave=False, desc="ESM2"):
        data_part = data_tmp[i * 5 :] if i == len(data_tmp) // 5 else data_tmp[i * 5 : (i + 1) * 5]
        if not data_part:
            continue
        inputs = tokenizer(
            [seq for _, seq in data_part],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        for j, (_, seq) in enumerate(data_part):
            records.append((seq, embeddings[j]))
    return pd.DataFrame(records, columns=["Protein", "esm"]).drop_duplicates(subset="Protein")


def add_features(df, tokenizer_bert, model_bert, tokenizer_esm, model_esm, device):
    if {"fcfp", "esm"}.issubset(df.columns):
        return df

    protein_features = get_protein_features(df["Protein"].unique().tolist(), tokenizer_esm, model_esm, device)
    featured_df = pd.merge(df, protein_features, on="Protein", how="left")

    unique_text = featured_df.drop_duplicates(subset="SMILES").copy()
    unique_text["fcfp"] = get_text_embeddings(unique_text, tokenizer_bert, model_bert, device)
    featured_df = pd.merge(featured_df, unique_text[["SMILES", "fcfp"]], on="SMILES", how="left")
    return featured_df


def load_split_with_features(
    split_path,
    cache_path,
    tokenizer_bert,
    model_bert,
    tokenizer_esm,
    model_esm,
    device,
    rebuild_cache=False,
):
    if os.path.exists(cache_path) and not rebuild_cache:
        return pd.read_pickle(cache_path)

    df = pd.read_excel(split_path)
    df = add_features(df, tokenizer_bert, model_bert, tokenizer_esm, model_esm, device)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_pickle(cache_path)
    return df


def load_dataset_splits(args, dataset, seed, data_root, device):
    split_dir = os.path.join(data_root, dataset, "data", str(seed))
    paths = {
        "train": os.path.join(split_dir, "train.xlsx"),
        "valid": os.path.join(split_dir, "valid.xlsx"),
        "test": os.path.join(split_dir, "test.xlsx"),
    }
    missing = [path for path in paths.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing split files for dataset={dataset}, seed={seed}: {missing}")

    sample = pd.read_excel(paths["train"], nrows=1)
    need_extract = not {"fcfp", "esm"}.issubset(sample.columns)
    tokenizer_bert = model_bert = tokenizer_esm = model_esm = None
    if need_extract:
        AutoTokenizer, AutoModel = require_transformers()
        tokenizer_bert = AutoTokenizer.from_pretrained(args.biobert_path)
        model_bert = AutoModel.from_pretrained(args.biobert_path).eval().to(device)
        tokenizer_esm = AutoTokenizer.from_pretrained(args.esm2_path)
        model_esm = AutoModel.from_pretrained(args.esm2_path).eval().to(device)

    split_frames = {}
    for split_name, split_path in paths.items():
        cache_path = os.path.join(args.cache_dir, dataset, str(seed), f"{split_name}.pkl")
        split_frames[split_name] = load_split_with_features(
            split_path,
            cache_path,
            tokenizer_bert,
            model_bert,
            tokenizer_esm,
            model_esm,
            device,
            rebuild_cache=args.rebuild_cache,
        )
    return split_frames["train"], split_frames["valid"], split_frames["test"]


def build_config(args, dataset, ablation, seed):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.DA.TASK = False
    cfg.DA.USE = False
    cfg.SOLVER.SEED = seed
    if args.max_epoch is not None:
        cfg.SOLVER.MAX_EPOCH = args.max_epoch
    if args.batch_size is not None:
        cfg.SOLVER.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        cfg.SOLVER.LR = args.lr
    if args.num_workers is not None:
        cfg.SOLVER.NUM_WORKERS = args.num_workers
    if args.no_save_model:
        cfg.RESULT.SAVE_MODEL = False
    cfg.RESULT.OUTPUT_DIR = os.path.join(args.output_root, dataset, ablation)
    mkdir(cfg.RESULT.OUTPUT_DIR)
    return cfg


def train_one(args, dataset, ablation, seed, data_root, device):
    cfg = build_config(args, dataset, ablation, seed)
    set_seed(seed)

    print(f"\n=== dataset={dataset} ablation={ablation} seed={seed} ===")
    df_train, df_val, df_test = load_dataset_splits(args, dataset, seed, data_root, device)

    train_dataset = DTIDataset(df_train.index.values, df_train)
    val_dataset = DTIDataset(df_val.index.values, df_val)
    test_dataset = DTIDataset(df_test.index.values, df_test)

    train_params = {
        "batch_size": cfg.SOLVER.BATCH_SIZE,
        "shuffle": True,
        "num_workers": cfg.SOLVER.NUM_WORKERS,
        "drop_last": True,
        "collate_fn": custom_collate_fn,
    }
    eval_params = dict(train_params)
    eval_params["shuffle"] = False
    eval_params["drop_last"] = False

    train_loader = DataLoader(train_dataset, **train_params)
    val_loader = DataLoader(val_dataset, **eval_params)
    test_loader = DataLoader(test_dataset, **eval_params)

    model = DrugBAN(ablation=ablation, **cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    trainer = Trainer(seed, model, optimizer, device, train_loader, val_loader, test_loader, **cfg)
    return trainer.train()


def summarize_results(results):
    if not results:
        return
    df = pd.DataFrame(results)
    metric_cols = ["auroc", "auprc", "F1", "Precision", "recall", "accuracy"]
    print("\n=== Summary by dataset/ablation ===")
    for (dataset, ablation), group in df.groupby(["dataset", "ablation"]):
        print(f"\n{dataset} / {ablation}")
        for metric in metric_cols:
            values = group[metric].astype(float).values
            print(f"{metric}: {np.mean(values):.4f} +- {np.var(values):.4f}")


def main():
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Data root: {data_root}")
    print(f"Running on: {device}")

    results = []
    for dataset in args.datasets:
        for ablation in args.ablations:
            for seed in args.seeds:
                torch.cuda.empty_cache()
                metrics = train_one(args, dataset, ablation, seed, data_root, device)
                metrics.update({"dataset": dataset, "ablation": ablation, "seed": seed})
                results.append(metrics)
    summarize_results(results)
    return results


if __name__ == "__main__":
    start = time()
    main()
    end = time()
    print(f"Total running time: {round(end - start, 2)}s")
