comet_support = True
try:
    from comet_ml import Experiment
except ImportError as e:
    print("Comet ML is not installed, ignore the comet experiment monitor")
    comet_support = False
from models_5 import DrugBAN
from time import time
from utils import set_seed, custom_collate_fn, mkdir  # graph_collate_func,
from configs_fungi import get_cfg_defaults
from dataloader import DTIDataset, MultiDataLoader
from torch.utils.data import DataLoader
from trainer import Trainer
from domain_adaptator import Discriminator
import torch
import argparse
import warnings, os
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

import logging
logging.getLogger().setLevel(logging.ERROR)   # 或 logging.CRITICAL

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="DrugBAN for DTI prediction")
parser.add_argument('--cfg', type=str, default='./configs/DrugBAN1.yaml', help="path to config file")
parser.add_argument('--data', type=str, default = 'Microbe',metavar='TASK',
                    help='dataset')
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task", choices=['random', 'cold', 'cluster'])
args = parser.parse_args()




# 定义获取分子特征的函数
def get_embeddings(df,tokenizer_bert,model_bert):
    emblist = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], leave=False):
        # 使用 BERT 分词器和模型
        encodings = tokenizer_bert(row['SMILES'], return_tensors='pt', padding="max_length", max_length=150, truncation=True)
        encodings = encodings.to(device)
        with torch.no_grad():
            output = model_bert(**encodings)
            smiles_embeddings = output.last_hidden_state[0, 0, :].cpu().numpy()
            emblist.append(smiles_embeddings)
    return emblist

 # 冻结策略实现 -----------------------------------
def freeze_feature_extractors(model):
    """冻结两个特征提取器"""
    # 冻结肽序列特征提取器 True-FT-PepEncoder
    for param in model.protein_extractor.parameters():
        param.requires_grad = True

    # 冻结文本特征提取器 True-FT-PatEncoder
    for param in model.drug_extractor.parameters():
        param.requires_grad = False

    # 双线性层和分类器保持可训练状态 True-FT-Cls
    for param in model.bcn.parameters():
        param.requires_grad = True
    for param in model.mlp_classifier.parameters():
        param.requires_grad = True
    return model


def main():
    aa = [12,16,18,20,42]  # ,16,18,20,42
    best_epoch = [151,152,196,182,125]  # best-epoch-source domain results.pth
    # bacteria:[58,184,187,167,45]  fungi:[116,52,90,85,177]  virus: [76,175,91,68,171]
    # text:  bacteria:[122,83,173,163,167]  fungi:[131,154,114,85,156]  virus:[71,164,197,163,148]
    for seed, current_epoch in zip(aa, best_epoch):
    # for seed in aa:
        torch.cuda.empty_cache()
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        cfg = get_cfg_defaults()
        cfg.merge_from_file(args.cfg)
        set_seed(cfg.SOLVER.SEED)
        suffix = str(int(time() * 1000))[6:]
        mkdir(cfg.RESULT.OUTPUT_DIR)
        experiment = None
        print(f"Config yaml: {args.cfg}")
        print(f"Hyperparameters: {dict(cfg)}")
        print(f"Running on: {device}", end="\n\n")

        dataFolder = './1-in-domain/1-random-splits/parasite/data'

        if not cfg.DA.TASK:
            train_path = os.path.join(dataFolder, str(seed), 'train.xlsx')
            val_path = os.path.join(dataFolder, str(seed), 'valid.xlsx')
            test_path = os.path.join(dataFolder, str(seed), 'test.xlsx')
            df_train = pd.read_excel(train_path)  # 在inductive中，train:source_train
            df_val = pd.read_excel(val_path)  # 在inductive中，val:target_train
            df_test = pd.read_excel(test_path)  # 在inductive中，test:target_test
            # 对数据进行处理，将寄生虫的大模型编码放进来，取代药物那一块的编码
            # 加载 BERT 模型和分词器
            bert_model_name = "./BioBert"  # BERT 模型名称
            tokenizer_bert = AutoTokenizer.from_pretrained(bert_model_name)
            model_bert = AutoModel.from_pretrained(bert_model_name)
            model_bert = model_bert.eval().to(device)

            # 加载 ESM 模型和分词器
            esm_model_name = "./ESM2"  # ESM 模型名称
            tokenizer_esm = AutoTokenizer.from_pretrained(esm_model_name)
            model_esm = AutoModel.from_pretrained(esm_model_name)
            model_esm = model_esm.eval().to(device)

            # 定义获取蛋白质特征的函数
            def Get_Protein_Feature(p_list):
                feature = []
                data_tmp = []
                dictionary = {}
                i = 0
                for p in p_list:  # 截断蛋白质序列
                    p = p[0:51]  # 截断蛋白质序列
                    data_tmp.append(("protein" + str(i), p))
                    i = i + 1
                sequence_representations = []

                for i in range(len(data_tmp) // 5 + 1):
                    if i == len(data_tmp) // 5:
                        data_part = data_tmp[i * 5:]
                    else:
                        data_part = data_tmp[i * 5:(i + 1) * 5]  # 拿出五个来list:5

                    if not data_part:  # 检查 data_part 是否为空
                        continue

                    # 使用 ESM 分词器和模型
                    inputs = tokenizer_esm([seq for _, seq in data_part], return_tensors='pt', padding=True,
                                           truncation=True)
                    inputs = inputs.to(device)  # input_ids:(5,447) attention_mask:(5,447) 5:batch,447:length?
                    with torch.no_grad():
                        outputs = model_esm(**inputs)
                        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # 进行平均了，(5,1280)

                    for j, (_, seq) in enumerate(data_part):
                        dictionary[seq] = embeddings[j]  # 让蛋白质序列与其编码放在一起，放在dictionary中，key(protein 序列):value(编码值)

                df = pd.DataFrame(dictionary.items(), columns=['Protein', 'esm'])
                return df

            # 处理训练集、验证集和测试集
            pro_list_train = df_train['Protein'].unique()  # (1387) 训练集中的蛋白质序列
            x_train = Get_Protein_Feature(list(pro_list_train))  # 是蛋白质和esm编码后的对应的  key(蛋白质序列):value(esm编码)
            df_train = pd.merge(df_train, x_train, on='Protein', how='left')  # 在原train中的那个表里加入了一列
            print('train esm is done!\n')

            pro_list_val = df_val['Protein'].unique()  # (478)
            x_val = Get_Protein_Feature(list(pro_list_val))
            df_val = pd.merge(df_val, x_val, on='Protein', how='left')
            print('val esm is done!\n')

            pro_list_test = df_test['Protein'].unique()  # (162)
            x_test = Get_Protein_Feature(list(pro_list_test))
            df_test = pd.merge(df_test, x_test, on='Protein', how='left')
            print('test esm is done!\n')  # 训练/验证/测试中的蛋白质序列都需要进行esm的编码


            # 获取分子特征
            df_trainu = df_train.drop_duplicates(subset='SMILES')  # (1880,7)
            df_valu = df_val.drop_duplicates(subset='SMILES')  # (552,7)
            df_testu = df_test.drop_duplicates(subset='SMILES')  # (173,7)

            emblist_train = get_embeddings(df_trainu, tokenizer_bert, model_bert)  # 每一个是384
            df_trainu['fcfp'] = emblist_train  # (1880,7)

            emblist_val = get_embeddings(df_valu, tokenizer_bert, model_bert)
            df_valu['fcfp'] = emblist_val  # (552,7)

            emblist_test = get_embeddings(df_testu, tokenizer_bert, model_bert)
            df_testu['fcfp'] = emblist_test  # (173,7)  # 在这里面加入了

            # 合并特征
            df_train = pd.merge(df_train, df_trainu[['SMILES', 'fcfp']], on='SMILES', how='left')  # (3589,7)
            df_val = pd.merge(df_val, df_valu[['SMILES', 'fcfp']], on='SMILES', how='left')  # (759,7)
            df_test = pd.merge(df_test, df_testu[['SMILES', 'fcfp']], on='SMILES', how='left')  # (190,7)
            print('chemBERTa feature extraction: pass\n')

            train_dataset = DTIDataset(df_train.index.values, df_train)
            val_dataset = DTIDataset(df_val.index.values, df_val)
            test_dataset = DTIDataset(df_test.index.values, df_test)
        else:
            train_source_path = os.path.join(dataFolder, 'source_train.csv')
            train_target_path = os.path.join(dataFolder, 'target_train.csv')
            test_target_path = os.path.join(dataFolder, 'target_test.csv')
            df_train_source = pd.read_csv(train_source_path)  # (14928,5)
            df_train_target = pd.read_csv(train_target_path)  # (7114,5)
            df_test_target = pd.read_csv(test_target_path)  # (1779.5)
            train_dataset = DTIDataset(df_train_source.index.values, df_train_source)
            train_target_dataset = DTIDataset(df_train_target.index.values, df_train_target)
            test_target_dataset = DTIDataset(df_test_target.index.values, df_test_target)

        # 定义列表
        f1_list = []
        precision_list = []
        auc_list = []
        auprc_list = []
        recall_list = []
        accuracy_list = []

        if cfg.COMET.USE and comet_support:
            experiment = Experiment(
                project_name=cfg.COMET.PROJECT_NAME,
                workspace=cfg.COMET.WORKSPACE,
                auto_output_logging="simple",
                log_graph=True,
                log_code=False,
                log_git_metadata=False,
                log_git_patch=False,
                auto_param_logging=False,
                auto_metric_logging=False
            )
            hyper_params = {
                "LR": cfg.SOLVER.LR,
                "Output_dir": cfg.RESULT.OUTPUT_DIR,
                "DA_use": cfg.DA.USE,
                "DA_task": cfg.DA.TASK,
            }
            if cfg.DA.USE:
                da_hyper_params = {
                    "DA_init_epoch": cfg.DA.INIT_EPOCH,
                    "Use_DA_entropy": cfg.DA.USE_ENTROPY,
                    "Random_layer": cfg.DA.RANDOM_LAYER,
                    "Original_random": cfg.DA.ORIGINAL_RANDOM,
                    "DA_optim_lr": cfg.SOLVER.DA_LR
                }
                hyper_params.update(da_hyper_params)
            experiment.log_parameters(hyper_params)
            if cfg.COMET.TAG is not None:
                experiment.add_tag(cfg.COMET.TAG)
            experiment.set_name(f"{args.data}_{suffix}")

        params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
                  'drop_last': True, 'collate_fn': custom_collate_fn}

        if not cfg.DA.USE:
            training_generator = DataLoader(train_dataset, **params)
            params['shuffle'] = False
            params['drop_last'] = False
            if not cfg.DA.TASK:
                val_generator = DataLoader(val_dataset, **params)
                test_generator = DataLoader(test_dataset, **params)
            else:
                val_generator = DataLoader(test_target_dataset, **params)
                test_generator = DataLoader(test_target_dataset, **params)
        else:
            source_generator = DataLoader(train_dataset, **params)
            target_generator = DataLoader(train_target_dataset, **params)
            n_batches = max(len(source_generator), len(target_generator))
            multi_generator = MultiDataLoader(dataloaders=[source_generator, target_generator], n_batches=n_batches)
            params['shuffle'] = False
            params['drop_last'] = False
            val_generator = DataLoader(test_target_dataset, **params)
            test_generator = DataLoader(test_target_dataset, **params)

        model = DrugBAN(**cfg).to(device)
        model.load_state_dict(
        torch.load(f"./2-source-domain-results\bacteria+virus-parasite/{seed}/best_model_epoch_{current_epoch}.pth"))
        # # 20:best_model_epoch_167.pth   42:best_model_epoch_45.pth  --bacteria
        # # fungi: 12-best_model_epoch_116.pth  16-best_model_epoch_52.pth  18-best_model_epoch_90.pth  20-best_model_epoch_85.pth  42-best_model_epoch_177.pth
        model = freeze_feature_extractors(model)
        # 5. 微调配置（仅优化未冻结参数）
        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3,  # 通常比预训练学习率大
            weight_decay=1e-5
            )

        # opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

        torch.backends.cudnn.benchmark = True

        trainer = Trainer(seed,model, opt, device, training_generator, val_generator, test_generator, opt_da=None,
                          discriminator=None,
                          experiment=experiment, **cfg)

        result = trainer.train()
        # 将 result 字典中的值添加到相应的列表中
        f1_list.append(result['F1'])
        precision_list.append(result['Precision'])
        auc_list.append(result['auroc'])
        auprc_list.append(result['auprc'])
        recall_list.append(result['recall'])
        accuracy_list.append(result['accuracy'])

        # 计算均值和方差
    f1_mean = np.mean(f1_list)
    f1_var = np.var(f1_list)
    precision_mean = np.mean(precision_list)
    precision_var = np.var(precision_list)
    auc_mean = np.mean(auc_list)
    auc_var = np.var(auc_list)
    auprc_mean = np.mean(auprc_list)
    auprc_var = np.var(auprc_list)
    recall_mean = np.mean(recall_list)
    recall_var = np.var(recall_list)
    accuracy_mean = np.mean(accuracy_list)
    accuracy_var = np.var(accuracy_list)

    # 打印结果，保留四位小数
    print("F1 Mean: {:.4f} ± {:.4f}".format(f1_mean, f1_var))
    print("Precision Mean: {:.4f} ± {:.4f}".format(precision_mean, precision_var))
    print("AUROC Mean: {:.4f} ± {:.4f}".format(auc_mean, auc_var))
    print("AUPRC Mean: {:.4f} ± {:.4f}".format(auprc_mean, auprc_var))
    print("Recall Mean: {:.4f} ± {:.4f}".format(recall_mean, recall_var))
    print("Accuracy Mean: {:.4f} ± {:.4f}".format(accuracy_mean, accuracy_var))

    print()
    print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")

    return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
