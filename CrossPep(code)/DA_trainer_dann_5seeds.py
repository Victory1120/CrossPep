import copy
import os
import numpy as np
import torch
import torch.nn as nn
from prettytable import PrettyTable
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from domain_adaptator import ReverseLayerF
from models_5 import binary_cross_entropy, cross_entropy_logits, entropy_logits, RandomLayer


class Trainer(object):
    """
    DANN-style domain-adversarial trainer.

    Notes:
    - Source-domain batches provide supervised labels for model_loss.
    - Target-domain batches are used only for domain-adversarial alignment.
    - Domain-adversarial loss is activated from config["DA"]["INIT_EPOCH"].
    - The implementation keeps compatibility with config["DA"]["METHOD"] == "CDAN",
      but the actual domain loss here is feature-level DANN, because the discriminator
      receives only feature representations f and f_t after gradient reversal.
    """

    def __init__(
        self,
        seed,
        model,
        optim,
        device,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        opt_da=None,
        discriminator=None,
        experiment=None,
        alpha=1,
        save_epochs=None,
        **config,
    ):
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
        self.optim_da = opt_da
        self.domain_dmm = discriminator
        self.save_epochs = set(save_epochs or [190, 200])

        if self.is_da:
            self.da_method = config["DA"]["METHOD"]
            if config["DA"].get("RANDOM_LAYER", False) and not config["DA"].get("ORIGINAL_RANDOM", False):
                self.random_layer = nn.Linear(
                    in_features=config["DECODER"]["IN_DIM"] * self.n_class,
                    out_features=config["DA"]["RANDOM_DIM"],
                    bias=False,
                ).to(self.device)
                torch.nn.init.normal_(self.random_layer.weight, mean=0, std=1)
                for param in self.random_layer.parameters():
                    param.requires_grad = False
            elif config["DA"].get("RANDOM_LAYER", False) and config["DA"].get("ORIGINAL_RANDOM", False):
                self.random_layer = RandomLayer([config["DECODER"]["IN_DIM"], self.n_class], config["DA"]["RANDOM_DIM"])
                if torch.cuda.is_available():
                    self.random_layer.cuda()
            else:
                self.random_layer = False

        self.da_init_epoch = config["DA"]["INIT_EPOCH"]
        self.init_lamb_da = config["DA"].get("LAMB_DA", 1.0)
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.use_da_entropy = config["DA"].get("USE_ENTROPY", False)
        self.nb_training = len(self.train_dataloader)
        self.step = 0
        self.experiment = experiment

        self.best_model = None
        self.best_epoch = None
        self.best_auroc = -1.0

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]

        valid_metric_header = ["# Seed", "# Epoch", "AUROC", "AUPRC", "F1", "Accuracy", "Val_loss"]
        test_metric_header = [
            "# Seed",
            "# Epoch",
            "Mode",
            "AUROC",
            "AUPRC",
            "F1",
            "Recall",
            "Accuracy",
            "Precision",
            "Test_loss",
        ]
        train_metric_header = ["# Seed", "# Epoch", "Train_loss", "Model_loss", "epoch_lamb_da", "da_loss"]
        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)

    @staticmethod
    def _as_tensor(x, device, dtype=None):
        if torch.is_tensor(x):
            x = x.to(device)
            if dtype is not None:
                x = x.to(dtype=dtype)
            return x
        return torch.tensor(x, dtype=dtype if dtype is not None else torch.float32, device=device)

    def da_lambda_decay(self):
        delta_epoch = self.current_epoch - self.da_init_epoch
        non_init_epoch = max(1, self.epochs - self.da_init_epoch)
        p = (self.current_epoch + delta_epoch * self.nb_training) / (non_init_epoch * self.nb_training)
        grow_fact = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        return self.init_lamb_da * grow_fact

    def train(self):
        float2str = lambda x: "%0.4f" % x
        output_path = os.path.join(self.output_dir, str(self.seed))
        os.makedirs(output_path, exist_ok=True)

        for _ in range(self.epochs):
            self.current_epoch += 1
            train_loss, model_loss, da_loss, epoch_lamb = self.train_da_epoch()
            train_lst = ["seed " + str(self.seed), "epoch " + str(self.current_epoch)] + list(
                map(float2str, [train_loss, model_loss, epoch_lamb, da_loss])
            )
            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            self.train_model_loss_epoch.append(model_loss)
            self.train_da_loss_epoch.append(da_loss)

            auroc, auprc, f1, acc, val_loss = self.test(dataloader="val", use_best=False)
            val_lst = ["seed " + str(self.seed), "epoch " + str(self.current_epoch)] + list(
                map(float2str, [auroc, auprc, f1, acc, val_loss])
            )
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(auroc)

            if auroc >= self.best_auroc:
                self.best_model = copy.deepcopy(self.model)
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch

            print(
                "Validation at Epoch " + str(self.current_epoch) + " with validation loss " + str(val_loss),
                " AUROC " + str(auroc) + " AUPRC " + str(auprc) + " F1 " + str(f1) + " ACC " + str(acc),
            )

            if self.current_epoch in self.save_epochs:
                epoch_name = str(self.current_epoch)
                torch.save(self.model.state_dict(), os.path.join(output_path, f"last_model_epoch_{epoch_name}.pth"))
                torch.save(self.model.state_dict(), os.path.join(output_path, f"{epoch_name}.pth"))
                auroc, auprc, f1, recall, accuracy, test_loss, precision = self.test(dataloader="test", use_best=False)
                test_lst = ["seed " + str(self.seed), "epoch " + epoch_name, "current"] + list(
                    map(float2str, [auroc, auprc, f1, recall, accuracy, precision, test_loss])
                )
                self.test_table.add_row(test_lst)
                print(
                    "Test at Current Model of Epoch " + epoch_name + " with test loss " + str(test_loss),
                    " AUROC " + str(auroc) + " AUPRC " + str(auprc) + " F1 " + str(f1)
                    + " Recall " + str(recall) + " Accuracy " + str(accuracy) + " Precision " + str(precision),
                )

        if self.best_model is None:
            self.best_model = copy.deepcopy(self.model)
            self.best_epoch = self.current_epoch
            self.best_auroc = self.val_auroc_epoch[-1] if self.val_auroc_epoch else 0.0

        auroc, auprc, f1, recall, accuracy, test_loss, precision = self.test(dataloader="test", use_best=True)
        test_lst = ["seed " + str(self.seed), "epoch " + str(self.best_epoch), "best"] + list(
            map(float2str, [auroc, auprc, f1, recall, accuracy, precision, test_loss])
        )
        self.test_table.add_row(test_lst)
        print(
            "Test at Best Model of Epoch " + str(self.best_epoch) + " with test loss " + str(test_loss),
            " AUROC " + str(auroc) + " AUPRC " + str(auprc) + " F1 " + str(f1)
            + " Recall " + str(recall) + " Accuracy " + str(accuracy) + " Precision " + str(precision),
        )

        self.test_metrics = {
            "auroc": auroc,
            "auprc": auprc,
            "test_loss": test_loss,
            "recall": recall,
            "accuracy": accuracy,
            "best_epoch": self.best_epoch,
            "best_val_auroc": self.best_auroc,
            "F1": f1,
            "Precision": precision,
        }
        self.save_result()
        return self.test_metrics

    def save_result(self):
        output_path = os.path.join(self.output_dir, str(self.seed))
        os.makedirs(output_path, exist_ok=True)
        if self.config["RESULT"].get("SAVE_MODEL", True):
            torch.save(self.best_model.state_dict(), os.path.join(output_path, f"best_model_epoch_{self.best_epoch}.pth"))
            torch.save(self.best_model.state_dict(), os.path.join(output_path, "best.pth"))
            torch.save(self.model.state_dict(), os.path.join(output_path, f"model_epoch_{self.current_epoch}.pth"))

        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "train_model_loss": self.train_model_loss_epoch,
            "train_da_loss": self.train_da_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "val_epoch_auroc": self.val_auroc_epoch,
            "test_metrics": self.test_metrics,
            "da_init_epoch": self.da_init_epoch,
            "config": self.config,
        }
        torch.save(state, os.path.join(output_path, "result_metrics.pt"))

        files = {
            "valid_markdowntable.txt": self.val_table,
            "test_markdowntable.txt": self.test_table,
            "train_markdowntable.txt": self.train_table,
            "valid.txt": self.val_table,
            "test.txt": self.test_table,
            "train.txt": self.train_table,
        }
        for filename, table in files.items():
            with open(os.path.join(output_path, filename), "w", encoding="utf-8") as fp:
                fp.write(table.get_string())

    def train_da_epoch(self):
        self.model.train()
        total_loss_epoch = 0.0
        model_loss_epoch = 0.0
        da_loss_epoch = 0.0
        epoch_lamb_da = 0.0
        if self.current_epoch >= self.da_init_epoch:
            epoch_lamb_da = 1.0
            # Use the following line instead if you want a smooth DANN ramp-up:
            # epoch_lamb_da = self.da_lambda_decay()

        num_batches = len(self.train_dataloader)
        for batch_s, batch_t in tqdm(self.train_dataloader):
            self.step += 1

            v_d = self._as_tensor(batch_s[0], self.device, dtype=torch.float32)
            v_p = self._as_tensor(batch_s[1], self.device)
            labels = self._as_tensor(batch_s[2], self.device, dtype=torch.float32)
            v_d_t = self._as_tensor(batch_t[0], self.device, dtype=torch.float32)
            v_p_t = self._as_tensor(batch_t[1], self.device)

            self.optim.zero_grad()
            if self.optim_da is not None:
                self.optim_da.zero_grad()

            _, _, f, score = self.model(v_d, v_p)
            if self.n_class == 1:
                _, model_loss = binary_cross_entropy(score, labels)
            else:
                _, model_loss = cross_entropy_logits(score, labels)

            if self.current_epoch >= self.da_init_epoch:
                _, _, f_t, _ = self.model(v_d_t, v_p_t)
                reverse_f = ReverseLayerF.apply(f, self.alpha)
                reverse_f_t = ReverseLayerF.apply(f_t, self.alpha)
                domain_feature = torch.cat((reverse_f, reverse_f_t), dim=0)
                domain_labels = torch.cat(
                    [
                        torch.zeros(reverse_f.size(0), dtype=torch.long, device=self.device),
                        torch.ones(reverse_f_t.size(0), dtype=torch.long, device=self.device),
                    ],
                    dim=0,
                )
                domain_pred = self.domain_dmm(domain_feature)
                _, da_loss = cross_entropy_logits(domain_pred, domain_labels)
                loss = model_loss + epoch_lamb_da * da_loss
            else:
                da_loss = torch.tensor(0.0, device=self.device)
                loss = model_loss

            loss.backward()
            self.optim.step()
            if self.optim_da is not None and self.current_epoch >= self.da_init_epoch:
                self.optim_da.step()

            total_loss_epoch += loss.item()
            model_loss_epoch += model_loss.item()
            da_loss_epoch += da_loss.item()

        total_loss_epoch /= max(1, num_batches)
        model_loss_epoch /= max(1, num_batches)
        da_loss_epoch /= max(1, num_batches)

        if self.current_epoch < self.da_init_epoch:
            print(f"Training at Epoch {self.current_epoch} with model training loss {total_loss_epoch}")
        else:
            print(
                f"Training at Epoch {self.current_epoch} model training loss {model_loss_epoch}, "
                f"da loss {da_loss_epoch}, total training loss {total_loss_epoch}, D lambda {epoch_lamb_da}"
            )
        return total_loss_epoch, model_loss_epoch, da_loss_epoch, epoch_lamb_da

    def test(self, dataloader="test", use_best=True):
        test_loss = 0.0
        y_label, y_pred = [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")

        eval_model = self.best_model if (dataloader == "test" and use_best and self.best_model is not None) else self.model
        eval_model.eval()
        num_batches = len(data_loader)
        with torch.no_grad():
            for v_d, v_p, labels in data_loader:
                v_d = self._as_tensor(v_d, self.device, dtype=torch.float32)
                v_p = self._as_tensor(v_p, self.device)
                labels = self._as_tensor(labels, self.device, dtype=torch.float32)
                _, _, _, score = eval_model(v_d, v_p)
                if self.n_class == 1:
                    n, loss = binary_cross_entropy(score, labels)
                else:
                    n, loss = cross_entropy_logits(score, labels)
                test_loss += loss.item()
                y_label += labels.detach().cpu().tolist()
                y_pred += n.detach().cpu().tolist()

        y_label = np.asarray(y_label)
        y_pred = np.asarray(y_pred)
        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        test_loss = test_loss / max(1, num_batches)

        fpr, tpr, thresholds = roc_curve(y_label, y_pred)
        precision_curve = tpr / (tpr + fpr + 1e-10)
        f1_curve = 2 * precision_curve * tpr / (tpr + precision_curve + 1e-5)
        if len(thresholds) > 5:
            thred_optim = thresholds[5:][np.argmax(f1_curve[5:])]
            f1_best = np.max(f1_curve[5:])
        else:
            thred_optim = 0.5
            f1_best = np.max(f1_curve)
        y_pred_s = (y_pred >= thred_optim).astype(int)
        cm1 = confusion_matrix(y_label, y_pred_s)
        recall1 = recall_score(y_label, y_pred_s, zero_division=0)
        accuracy = (cm1[0, 0] + cm1[1, 1]) / np.sum(cm1)
        precision1 = precision_score(y_label, y_pred_s, zero_division=0)

        if dataloader == "val":
            return auroc, auprc, f1_best, accuracy, test_loss
        return auroc, auprc, f1_best, recall1, accuracy, test_loss, precision1
