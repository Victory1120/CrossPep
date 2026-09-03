import argparse
import json
import logging
import os
import shutil
import warnings
from time import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from configs_fungi import get_cfg_defaults
from dataloader import DTIDataset, MultiDataLoader
from domain_adaptator import Discriminator
from models_5 import DrugBAN
from DA_trainer_dann_5seeds import Trainer
from utils import custom_collate_fn, mkdir, set_seed

logging.getLogger().setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(description="DANN/RarePep-3 training on one xlsx split with 5 training seeds")
    parser.add_argument("--cfg", type=str, default="./configs/DrugBAN_DA.yaml", help="path to config file")
    parser.add_argument("--data_root", type=str, default=r"./3-single-cold-unseen-parasites/data/seed16", help="folder containing train.xlsx, valid.xlsx and test.xlsx")  # # single-cold if dual-cold, please: ./4-dual-cold/data, 
    parser.add_argument("--output_dir", type=str, default="./result/single-cold", help="directory to save all DANN results")
    parser.add_argument("--device", type=str, default="cuda:2", help="cuda device, e.g. cuda:0 / cuda:2 / cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])  # 2024, 2025, 2026, 2027, 2028 12, 16, 18, 20, 42
    parser.add_argument("--max_epoch", type=int, default=100) # single-cold:200 dual-cold:100
    parser.add_argument("--da_init_epoch", type=int, default=30, help="start DANN/domain adversarial loss from this epoch") # 200: 40; 100: 30
    parser.add_argument("--save_epochs", type=int, nargs="+", default=[90, 100])
    parser.add_argument("--bert_model_path", type=str, default="./BioBert")
    parser.add_argument("--esm_model_path", type=str, default="./ESM2")
    parser.add_argument("--excel_sheet", type=str, default=None)
    parser.add_argument("--cache_features", action="store_true", help="cache encoded dataframe features to save time")
    parser.add_argument("--source_file", type=str, default="train.xlsx")
    parser.add_argument("--target_file", type=str, default="valid.xlsx", help="target-domain adaptation file; usually valid.xlsx")
    parser.add_argument("--test_file", type=str, default="test.xlsx")
    return parser.parse_args()


def cfg_set(cfg, key_path, value):
    node = cfg
    for k in key_path[:-1]:
        node = node[k]
    node[key_path[-1]] = value


def load_xlsx(path, sheet_name=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    if sheet_name is None:
        return pd.read_excel(path)
    return pd.read_excel(path, sheet_name=sheet_name)


def validate_columns(df, name):
    required = {"SMILES", "Protein"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def get_text_embeddings(df_unique, tokenizer_bert, model_bert, device):
    emblist = []
    for _, row in tqdm(df_unique.iterrows(), total=df_unique.shape[0], leave=False, desc="BioBERT"):
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
            emblist.append(smiles_embeddings)
    return emblist


def get_protein_features(p_list, tokenizer_esm, model_esm, device, batch_size=5):
    data_tmp = []
    for i, p in enumerate(p_list):
        seq = str(p)[:51]
        data_tmp.append(("protein" + str(i), seq))

    dictionary = {}
    for i in tqdm(range(len(data_tmp) // batch_size + 1), leave=False, desc="ESM2"):
        if i == len(data_tmp) // batch_size:
            data_part = data_tmp[i * batch_size:]
        else:
            data_part = data_tmp[i * batch_size:(i + 1) * batch_size]
        if not data_part:
            continue
        inputs = tokenizer_esm([seq for _, seq in data_part], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model_esm(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
        for j, (_, seq) in enumerate(data_part):
            dictionary[seq] = embeddings[j]

    return pd.DataFrame(dictionary.items(), columns=["Protein", "esm"])


def encode_dataframe(df, split_name, tokenizer_bert, model_bert, tokenizer_esm, model_esm, device):
    validate_columns(df, split_name)
    df = df.copy()
    df["Protein"] = df["Protein"].astype(str).str.slice(0, 51)
    df["SMILES"] = df["SMILES"].astype(str)

    pro_list = df["Protein"].unique()
    x_pro = get_protein_features(list(pro_list), tokenizer_esm, model_esm, device)
    df = pd.merge(df, x_pro, on="Protein", how="left")
    print(f"{split_name} esm is done!")

    df_unique = df.drop_duplicates(subset="SMILES").copy()
    df_unique["fcfp"] = get_text_embeddings(df_unique, tokenizer_bert, model_bert, device)
    df = pd.merge(df, df_unique[["SMILES", "fcfp"]], on="SMILES", how="left")
    print(f"{split_name} BioBERT feature extraction is done!")
    return df


def maybe_cache_encoded(args, device):
    cache_dir = os.path.join(args.output_dir, "feature_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "encoded_source_target_test.pkl")
    if args.cache_features and os.path.exists(cache_file):
        print(f"Loading cached encoded features from: {cache_file}")
        payload = pd.read_pickle(cache_file)
        return payload["source"], payload["target"], payload["test"]

    source_path = os.path.join(args.data_root, args.source_file)
    target_path = os.path.join(args.data_root, args.target_file)
    test_path = os.path.join(args.data_root, args.test_file)

    df_source = load_xlsx(source_path, args.excel_sheet)
    df_target = load_xlsx(target_path, args.excel_sheet)
    df_test = load_xlsx(test_path, args.excel_sheet)

    tokenizer_bert = AutoTokenizer.from_pretrained(args.bert_model_path)
    model_bert = AutoModel.from_pretrained(args.bert_model_path).eval().to(device)
    tokenizer_esm = AutoTokenizer.from_pretrained(args.esm_model_path)
    model_esm = AutoModel.from_pretrained(args.esm_model_path).eval().to(device)

    df_source = encode_dataframe(df_source, "source_train", tokenizer_bert, model_bert, tokenizer_esm, model_esm, device)
    df_target = encode_dataframe(df_target, "target_train", tokenizer_bert, model_bert, tokenizer_esm, model_esm, device)
    df_test = encode_dataframe(df_test, "target_test", tokenizer_bert, model_bert, tokenizer_esm, model_esm, device)

    del model_bert, model_esm
    torch.cuda.empty_cache()

    if args.cache_features:
        pd.to_pickle({"source": df_source, "target": df_target, "test": df_test}, cache_file)
        print(f"Cached encoded features to: {cache_file}")
    return df_source, df_target, df_test


def prepare_cfg(args, seed, run_output_dir):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    try:
        cfg.defrost()
    except Exception:
        pass
    cfg_set(cfg, ["SOLVER", "SEED"], seed)
    cfg_set(cfg, ["SOLVER", "MAX_EPOCH"], args.max_epoch)
    cfg_set(cfg, ["RESULT", "OUTPUT_DIR"], run_output_dir)
    cfg_set(cfg, ["DA", "USE"], True)
    cfg_set(cfg, ["DA", "INIT_EPOCH"], args.da_init_epoch)
    # Keep compatibility with your original config/trainer naming.
    if "METHOD" in cfg["DA"]:
        cfg_set(cfg, ["DA", "METHOD"], "CDAN")
    try:
        cfg.freeze()
    except Exception:
        pass
    return cfg


def copy_input_files(args, seed_output_dir):
    split_dir = os.path.join(seed_output_dir, "data_split")
    os.makedirs(split_dir, exist_ok=True)
    mapping = {
        args.source_file: "train.xlsx",
        args.target_file: "valid.xlsx",
        args.test_file: "test.xlsx",
    }
    for src_name, dst_name in mapping.items():
        src = os.path.join(args.data_root, src_name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(split_dir, dst_name))


def main():
    args = parse_args()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    mkdir(args.output_dir)
    print(f"Running on: {device}")
    print(f"DANN/domain adversarial loss will start from epoch = {args.da_init_epoch}")

    df_source, df_target, df_test = maybe_cache_encoded(args, device)

    all_results = []
    for seed in args.seeds:
        print("\n" + "=" * 80)
        print(f"Starting DANN run | seed={seed}")
        print("=" * 80)
        torch.cuda.empty_cache()
        set_seed(seed)

        cfg = prepare_cfg(args, seed, args.output_dir)

        train_dataset = DTIDataset(df_source.index.values, df_source)
        train_target_dataset = DTIDataset(df_target.index.values, df_target)
        test_target_dataset = DTIDataset(df_test.index.values, df_test)

        params = {
            "batch_size": cfg.SOLVER.BATCH_SIZE,
            "shuffle": True,
            "num_workers": cfg.SOLVER.NUM_WORKERS,
            "drop_last": True,
            "collate_fn": custom_collate_fn,
        }
        source_generator = DataLoader(train_dataset, **params)
        target_generator = DataLoader(train_target_dataset, **params)
        n_batches = max(len(source_generator), len(target_generator))
        multi_generator = MultiDataLoader(dataloaders=[source_generator, target_generator], n_batches=n_batches)

        eval_params = dict(params)
        eval_params["shuffle"] = False
        eval_params["drop_last"] = False
        # This follows your original DANN script: test.xlsx is used for validation and final test.
        # If you later have a separate target_valid.xlsx, replace val_generator accordingly.
        val_generator = DataLoader(test_target_dataset, **eval_params)
        test_generator = DataLoader(test_target_dataset, **eval_params)

        model = DrugBAN(**cfg).to(device)
        if cfg["DA"].get("RANDOM_LAYER", False):
            domain_dmm = Discriminator(input_size=cfg["DA"]["RANDOM_DIM"], n_class=cfg["DECODER"]["BINARY"]).to(device)
        else:
            domain_dmm = Discriminator(
                input_size=cfg["DECODER"]["IN_DIM"] * cfg["DECODER"]["BINARY"],
                n_class=cfg["DECODER"]["BINARY"],
            ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
        opt_da = torch.optim.Adam(domain_dmm.parameters(), lr=cfg.SOLVER.DA_LR)

        trainer = Trainer(
            seed,
            model,
            opt,
            device,
            multi_generator,
            val_generator,
            test_generator,
            opt_da=opt_da,
            discriminator=domain_dmm,
            experiment=None,
            save_epochs=args.save_epochs,
            **cfg,
        )
        result = trainer.train()

        seed_output_dir = os.path.join(args.output_dir, str(seed))
        copy_input_files(args, seed_output_dir)
        run_info = {
            "seed": seed,
            "cfg": args.cfg,
            "data_root": args.data_root,
            "source_file": args.source_file,
            "target_file": args.target_file,
            "test_file": args.test_file,
            "max_epoch": args.max_epoch,
            "da_init_epoch": args.da_init_epoch,
            "save_epochs": args.save_epochs,
            "note": "Feature-level DANN/domain-adversarial training; target_file is used as target-domain adaptation data.",
        }
        with open(os.path.join(seed_output_dir, "run_info.json"), "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2, ensure_ascii=False)

        row = {"seed": seed, **result}
        all_results.append(row)
        pd.DataFrame(all_results).to_csv(os.path.join(args.output_dir, "all_run_results.csv"), index=False)
        pd.DataFrame(all_results).to_excel(os.path.join(args.output_dir, "all_run_results.xlsx"), index=False)

    results_df = pd.DataFrame(all_results)
    metric_cols = [c for c in ["auroc", "auprc", "F1", "recall", "accuracy", "Precision", "test_loss", "best_epoch", "best_val_auroc"] if c in results_df]
    summary_rows = []
    for col in metric_cols:
        summary_rows.append({"metric": col, "mean": results_df[col].mean(), "std": results_df[col].std(ddof=1)})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.output_dir, "summary_dann.csv"), index=False)
    summary_df.to_excel(os.path.join(args.output_dir, "summary_dann.xlsx"), index=False)
    print("\nFinished all DANN runs.")
    print(summary_df)


if __name__ == "__main__":
    s = time()
    main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
