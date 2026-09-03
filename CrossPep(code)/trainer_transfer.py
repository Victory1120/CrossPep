import copy
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from prettytable import PrettyTable
from tqdm import tqdm

from models_5 import binary_cross_entropy, cross_entropy_logits, entropy_logits, RandomLayer
from domain_adaptator import ReverseLayerF


class Trainer(object):
    """
    Fine-tuning trainer for RarePep/DrugBAN-style peptide-pathogen prediction.

    Main changes relative to the original trainer:
    1) save user-specified epochs, e.g. 190.pth and 200.pth;
    2) keep best validation model and save best_model_epoch_*.pth;
    3) keep train/valid/test markdown txt outputs, and additionally write train.txt/valid.txt/test.txt aliases;
    4) evaluate intermediate checkpoints using the current epoch model instead of best_model.
    """

    def __init__(self, seed, model, optim, device, train_dataloader, val_dataloader, test_dataloader,
                 opt_da=None, discriminator=None, experiment=None, alpha=1, **config):
        self.seed = seed
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.is_da = config["DA"]["USE"]
        self.alpha = alpha
        self.n_class = config["DECODER"]["BINARY"]
        self.save_epochs = set(config.get("SAVE_EPOCHS", [190, 200]))

        if opt_da:
            self.optim_da = opt_da
        if self.is_da:
            self.da_method = config["DA"]["METHOD"]
            self.domain_dmm = discriminator
            if config["DA"]["RANDOM_LAYER"] and not config["DA"]["ORIGINAL_RANDOM"]:
                self.random_layer = nn.Linear(
                    in_features=config["DECODER"]["IN_DIM"] * self.n_class,
                    out_features=config["DA"]["RANDOM_DIM"],
                    bias=False,
                ).to(self.device)
                torch.nn.init.normal_(self.random_layer.weight, mean=0, std=1)
                for param in self.random_layer.parameters():
                    param.requires_grad = False
            elif config["DA"]["RANDOM_LAYER"] and config["DA"]["ORIGINAL_RANDOM"]:
                self.random_layer = RandomLayer([config["DECODER"]["IN_DIM"], self.n_class], config["DA"]["RANDOM_DIM"])
                if torch.cuda.is_available():
                    self.random_layer.cuda()
            else:
                self.random_layer = False

        self.da_init_epoch = config["DA"]["INIT_EPOCH"]
        self.init_lamb_da = config["DA"]["LAMB_DA"]
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.use_da_entropy = config["DA"]["USE_ENTROPY"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0
        self.experiment = experiment

        self.best_model = None
        self.best_epoch = None
        self.best_auroc = -float("inf")

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch, self.val_auprc_epoch = [], [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]

        valid_metric_header = ["# Seed", "# Epoch", "AUROC", "AUPRC", "Val_loss"]
        test_metric_header = ["# Seed", "# Epoch", "AUROC", "AUPRC", "F1", "Recall", "Accuracy", "Precision", "Test_loss"]
        if not self.is_da:
            train_metric_header = ["# Seed", "# Epoch", "Train_loss"]
        else:
            train_metric_header = ["# Epoch", "Train_loss", "Model_loss", "epoch_lamb_da", "da_loss"]
        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)

        self.original_random = config["DA"]["ORIGINAL_RANDOM"]

    def _to_float_tensor(self, x):
        if torch.is_tensor(x):
            return x.float().to(self.device)
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def da_lambda_decay(self):
        delta_epoch = self.current_epoch - self.da_init_epoch
        non_init_epoch = self.epochs - self.da_init_epoch
        p = (self.current_epoch + delta_epoch * self.nb_training) / (non_init_epoch * self.nb_training)
        grow_fact = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        return self.init_lamb_da * grow_fact

    def train(self):
        float2str = lambda x: "%0.4f" % x
        output_path = os.path.join(self.output_dir, str(self.seed))
        os.makedirs(output_path, exist_ok=True)

        for _ in range(self.epochs):
            self.current_epoch += 1
            if not self.is_da:
                train_loss = self.train_epoch()
                train_lst = ["seed " + str(self.seed), "epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
                if self.experiment:
                    self.experiment.log_metric("train_epoch model loss", train_loss, epoch=self.current_epoch)
            else:
                train_loss, model_loss, da_loss, epoch_lamb = self.train_da_epoch()
                train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss, model_loss, epoch_lamb, da_loss]))
                self.train_model_loss_epoch.append(model_loss)
                self.train_da_loss_epoch.append(da_loss)
                if self.experiment:
                    self.experiment.log_metric("train_epoch total loss", train_loss, epoch=self.current_epoch)
                    self.experiment.log_metric("train_epoch model loss", model_loss, epoch=self.current_epoch)
                    if self.current_epoch >= self.da_init_epoch:
                        self.experiment.log_metric("train_epoch da loss", da_loss, epoch=self.current_epoch)

            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)

            auroc, auprc, val_loss = self.evaluate(dataloader="val", model_to_eval=self.model)
            if self.experiment:
                self.experiment.log_metric("valid_epoch model loss", val_loss, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auroc", auroc, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auprc", auprc, epoch=self.current_epoch)

            val_lst = ["seed " + str(self.seed), "epoch " + str(self.current_epoch)] + list(map(float2str, [auroc, auprc, val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(auroc)
            self.val_auprc_epoch.append(auprc)

            if auroc >= self.best_auroc:
                self.best_model = copy.deepcopy(self.model)
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch

            print(
                "Validation at Epoch " + str(self.current_epoch) +
                " with validation loss " + str(val_loss) +
                " AUROC " + str(auroc) + " AUPRC " + str(auprc)
            )

            if self.current_epoch in self.save_epochs:
                ckpt_name = f"last_model_epoch_{self.current_epoch}.pth"
                torch.save(self.model.state_dict(), os.path.join(output_path, ckpt_name))
                # Alias required by the user, e.g. 190.pth and 200.pth
                torch.save(self.model.state_dict(), os.path.join(output_path, f"{self.current_epoch}.pth"))

                tauroc, tauprc, tf1, trecall, tacc, t_loss, tprecision = self.evaluate(
                    dataloader="test", model_to_eval=self.model
                )
                test_lst = ["seed " + str(self.seed), "epoch " + str(self.current_epoch)] + list(
                    map(float2str, [tauroc, tauprc, tf1, trecall, tacc, tprecision, t_loss])
                )
                self.test_table.add_row(test_lst)
                print(
                    "Test at Model of Epoch " + str(self.current_epoch) +
                    " with test loss " + str(t_loss) +
                    " AUROC " + str(tauroc) + " AUPRC " + str(tauprc) +
                    " f1-score " + str(tf1) + " Recall " + str(trecall) +
                    " Accuracy " + str(tacc) + " Precision " + str(tprecision)
                )

        if self.best_model is None:
            self.best_model = copy.deepcopy(self.model)
            self.best_epoch = self.current_epoch

        auroc, auprc, f1, recall, accuracy, test_loss, precision = self.evaluate(
            dataloader="test", model_to_eval=self.best_model
        )
        test_lst = ["seed " + str(self.seed), "best epoch " + str(self.best_epoch)] + list(
            map(float2str, [auroc, auprc, f1, recall, accuracy, precision, test_loss])
        )
        self.test_table.add_row(test_lst)
        print(
            "Test at Best Model of Epoch " + str(self.best_epoch) +
            " with test loss " + str(test_loss) +
            " AUROC " + str(auroc) + " AUPRC " + str(auprc) +
            " f1-score " + str(f1) + " Recall " + str(recall) +
            " Accuracy " + str(accuracy) + " Precision " + str(precision)
        )

        self.test_metrics["auroc"] = auroc
        self.test_metrics["auprc"] = auprc
        self.test_metrics["test_loss"] = test_loss
        self.test_metrics["recall"] = recall
        self.test_metrics["accuracy"] = accuracy
        self.test_metrics["best_epoch"] = self.best_epoch
        self.test_metrics["best_val_auroc"] = self.best_auroc
        self.test_metrics["F1"] = f1
        self.test_metrics["Precision"] = precision
        self.save_result()
        return self.test_metrics

    def save_result(self):
        output_path = os.path.join(self.output_dir, str(self.seed))
        os.makedirs(output_path, exist_ok=True)
        if self.config["RESULT"]["SAVE_MODEL"]:
            torch.save(self.best_model.state_dict(), os.path.join(output_path, f"best_model_epoch_{self.best_epoch}.pth"))
            torch.save(self.best_model.state_dict(), os.path.join(output_path, "best.pth"))
            torch.save(self.model.state_dict(), os.path.join(output_path, f"model_epoch_{self.current_epoch}.pth"))

        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "val_epoch_auroc": self.val_auroc_epoch,
            "val_epoch_auprc": self.val_auprc_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config,
        }
        if self.is_da:
            state["train_model_loss"] = self.train_model_loss_epoch
            state["train_da_loss"] = self.train_da_loss_epoch
            state["da_init_epoch"] = self.da_init_epoch
        torch.save(state, os.path.join(output_path, "result_metrics.pt"))

        files = {
            "valid_markdowntable.txt": self.val_table.get_string(),
            "test_markdowntable.txt": self.test_table.get_string(),
            "train_markdowntable.txt": self.train_table.get_string(),
            # aliases requested by the user
            "valid.txt": self.val_table.get_string(),
            "test.txt": self.test_table.get_string(),
            "train.txt": self.train_table.get_string(),
        }
        for filename, content in files.items():
            with open(os.path.join(output_path, filename), "w", encoding="utf-8") as fp:
                fp.write(content)

    def _compute_entropy_weights(self, logits):
        entropy = entropy_logits(logits)
        entropy = ReverseLayerF.apply(entropy, self.alpha)
        entropy_w = 1.0 + torch.exp(-entropy)
        return entropy_w

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        for _, (v_d, v_p, labels) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            v_d = self._to_float_tensor(v_d)
            v_p, labels = v_p.to(self.device), labels.float().to(self.device)
            self.optim.zero_grad()
            _, _, _, score = self.model(v_d, v_p)
            if self.n_class == 1:
                _, loss = binary_cross_entropy(score, labels)
            else:
                _, loss = cross_entropy_logits(score, labels)
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
            if self.experiment:
                self.experiment.log_metric("train_step model loss", loss.item(), step=self.step)
        loss_epoch = loss_epoch / max(num_batches, 1)
        print("Training at Epoch " + str(self.current_epoch) + " with training loss " + str(loss_epoch))
        return loss_epoch

    def train_da_epoch(self):
        self.model.train()
        total_loss_epoch = 0
        model_loss_epoch = 0
        da_loss_epoch = 0
        epoch_lamb_da = 0
        if self.current_epoch >= self.da_init_epoch:
            epoch_lamb_da = 1
            if self.experiment:
                self.experiment.log_metric("D loss lambda", epoch_lamb_da, epoch=self.current_epoch)
        num_batches = len(self.train_dataloader)
        for _, (batch_s, batch_t) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            v_d, v_p, labels = self._to_float_tensor(batch_s[0]), batch_s[1].to(self.device), batch_s[2].float().to(self.device)
            v_d_t, v_p_t = self._to_float_tensor(batch_t[0]), batch_t[1].to(self.device)
            self.optim.zero_grad()
            self.optim_da.zero_grad()
            _, _, f, score = self.model(v_d, v_p)
            if self.n_class == 1:
                _, model_loss = binary_cross_entropy(score, labels)
            else:
                _, model_loss = cross_entropy_logits(score, labels)
            _, _, f_t, t_score = self.model(v_d_t, v_p_t)
            if self.da_method == "CDAN":
                reverse_f = ReverseLayerF.apply(f, self.alpha)
                softmax_output = torch.nn.Softmax(dim=1)(score).detach()
                if self.original_random:
                    random_out = self.random_layer.forward([reverse_f, softmax_output])
                    adv_output_src_score = self.domain_dmm(random_out.view(-1, random_out.size(1)))
                else:
                    feature = torch.bmm(softmax_output.unsqueeze(2), reverse_f.unsqueeze(1))
                    feature = feature.view(-1, softmax_output.size(1) * reverse_f.size(1))
                    adv_output_src_score = self.domain_dmm(self.random_layer.forward(feature) if self.random_layer else feature)

                reverse_f_t = ReverseLayerF.apply(f_t, self.alpha)
                softmax_output_t = torch.nn.Softmax(dim=1)(t_score).detach()
                if self.original_random:
                    random_out_t = self.random_layer.forward([reverse_f_t, softmax_output_t])
                    adv_output_tgt_score = self.domain_dmm(random_out_t.view(-1, random_out_t.size(1)))
                else:
                    feature_t = torch.bmm(softmax_output_t.unsqueeze(2), reverse_f_t.unsqueeze(1))
                    feature_t = feature_t.view(-1, softmax_output_t.size(1) * reverse_f_t.size(1))
                    adv_output_tgt_score = self.domain_dmm(self.random_layer.forward(feature_t) if self.random_layer else feature_t)

                if self.use_da_entropy:
                    entropy_src = self._compute_entropy_weights(score)
                    entropy_tgt = self._compute_entropy_weights(t_score)
                    src_weight = entropy_src / torch.sum(entropy_src)
                    tgt_weight = entropy_tgt / torch.sum(entropy_tgt)
                else:
                    src_weight = None
                    tgt_weight = None
                _, loss_cdan_src = cross_entropy_logits(adv_output_src_score, torch.zeros(self.batch_size).to(self.device), src_weight)
                _, loss_cdan_tgt = cross_entropy_logits(adv_output_tgt_score, torch.ones(self.batch_size).to(self.device), tgt_weight)
                da_loss = loss_cdan_src + loss_cdan_tgt
            else:
                raise ValueError(f"The da method {self.da_method} is not supported")
            loss = model_loss + da_loss
            loss.backward()
            self.optim.step()
            self.optim_da.step()
            total_loss_epoch += loss.item()
            model_loss_epoch += model_loss.item()
            da_loss_epoch += da_loss.item()
        total_loss_epoch = total_loss_epoch / max(num_batches, 1)
        model_loss_epoch = model_loss_epoch / max(num_batches, 1)
        da_loss_epoch = da_loss_epoch / max(num_batches, 1)
        print(
            "Training at Epoch " + str(self.current_epoch) +
            " model training loss " + str(model_loss_epoch) +
            ", da loss " + str(da_loss_epoch) +
            ", total training loss " + str(total_loss_epoch) +
            ", D lambda " + str(epoch_lamb_da)
        )
        return total_loss_epoch, model_loss_epoch, da_loss_epoch, epoch_lamb_da

    def evaluate(self, dataloader="test", model_to_eval=None):
        test_loss = 0
        y_label, y_pred = [], []
        if dataloader == "test" or dataloader == "tval":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        if model_to_eval is None:
            model_to_eval = self.model
        num_batches = len(data_loader)
        with torch.no_grad():
            model_to_eval.eval()
            for _, (v_d, v_p, labels) in enumerate(data_loader):
                v_d = self._to_float_tensor(v_d)
                v_p, labels = v_p.to(self.device), labels.float().to(self.device)
                _, _, _, score = model_to_eval(v_d, v_p)
                if self.n_class == 1:
                    n, loss = binary_cross_entropy(score, labels)
                else:
                    n, loss = cross_entropy_logits(score, labels)
                test_loss += loss.item()
                y_label += labels.to("cpu").tolist()
                y_pred += n.to("cpu").tolist()
        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        test_loss = test_loss / max(num_batches, 1)

        if dataloader == "test" or dataloader == "tval":
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            precision_curve = tpr / (tpr + fpr + 1e-10)
            f1_curve = 2 * precision_curve * tpr / (tpr + precision_curve + 1e-5)
            if len(thresholds) > 5:
                thred_optim = thresholds[5:][np.argmax(f1_curve[5:])]
                best_f1 = np.max(f1_curve[5:])
            else:
                thred_optim = thresholds[np.argmax(f1_curve)]
                best_f1 = np.max(f1_curve)
            y_pred_arr = np.asarray(y_pred)
            y_pred_s = [1 if i >= thred_optim else 0 for i in y_pred_arr]
            cm1 = confusion_matrix(y_label, y_pred_s)
            recall1 = recall_score(y_label, y_pred_s, zero_division=0)
            accuracy = (cm1[0, 0] + cm1[1, 1]) / np.sum(cm1)
            precision1 = precision_score(y_label, y_pred_s, zero_division=0)
            return auroc, auprc, best_f1, recall1, accuracy, test_loss, precision1
        else:
            return auroc, auprc, test_loss

    # Backward-compatible name used by older code.
    def test(self, dataloader="test"):
        if dataloader == "test":
            return self.evaluate(dataloader="test", model_to_eval=self.best_model)
        return self.evaluate(dataloader=dataloader, model_to_eval=self.model)
