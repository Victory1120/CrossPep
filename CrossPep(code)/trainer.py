import torch
import torch.nn as nn
import copy
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score, recall_score
from models_5 import binary_cross_entropy, cross_entropy_logits, entropy_logits, RandomLayer
from prettytable import PrettyTable
from domain_adaptator import ReverseLayerF
from tqdm import tqdm


class Trainer(object):
    def __init__(self,  seed, model, optim, device, train_dataloader, val_dataloader, test_dataloader, opt_da=None, discriminator=None,
                 experiment=None, alpha=1, **config):
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
        if opt_da:
            self.optim_da = opt_da
        if self.is_da:
            self.da_method = config["DA"]["METHOD"]
            self.domain_dmm = discriminator
            if config["DA"]["RANDOM_LAYER"] and not config["DA"]["ORIGINAL_RANDOM"]:
                self.random_layer = nn.Linear(in_features=config["DECODER"]["IN_DIM"]*self.n_class, out_features=config["DA"]
                ["RANDOM_DIM"], bias=False).to(self.device)
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
        self.best_auroc = 0

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]

        valid_metric_header = ["# Seed", "# Epoch", "AUROC", "AUPRC", "Val_loss"]
        test_metric_header = ["# Seed", "# Best Epoch", "AUROC", "AUPRC", "F1", "Recall", "Accuracy", "Precision",
                              "Test_loss"]  # "Threshold",
        if not self.is_da:
            train_metric_header = ["# Seed", "# Epoch", "Train_loss"]

        else:
            train_metric_header = ["# Epoch", "Train_loss", "Model_loss", "epoch_lamb_da", "da_loss"]
        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)

        self.original_random = config["DA"]["ORIGINAL_RANDOM"]

    def da_lambda_decay(self):
        delta_epoch = self.current_epoch - self.da_init_epoch
        non_init_epoch = self.epochs - self.da_init_epoch
        p = (self.current_epoch + delta_epoch * self.nb_training) / (
                non_init_epoch * self.nb_training
        )
        grow_fact = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        return self.init_lamb_da * grow_fact

    def train(self):
        float2str = lambda x: '%0.4f' % x
        pth_last3_pth = os.path.join(self.output_dir, str(self.seed))
        os.makedirs(pth_last3_pth, exist_ok=True)  # 如果目录不存在，创建它
        for i in range(self.epochs):
            self.current_epoch += 1
            if not self.is_da:
                train_loss = self.train_epoch()
                train_lst = ["seed " + str(self.seed)] + ["epoch " + str(self.current_epoch)] + list(
                    map(float2str, [train_loss]))
                if self.experiment:
                    self.experiment.log_metric("train_epoch model loss", train_loss, epoch=self.current_epoch)
            else:
                train_loss, model_loss, da_loss, epoch_lamb = self.train_da_epoch()
                train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss, model_loss,
                                                                                        epoch_lamb, da_loss]))
                self.train_model_loss_epoch.append(model_loss)
                self.train_da_loss_epoch.append(da_loss)
                if self.experiment:
                    self.experiment.log_metric("train_epoch total loss", train_loss, epoch=self.current_epoch)
                    self.experiment.log_metric("train_epoch model loss", model_loss, epoch=self.current_epoch)
                    if self.current_epoch >= self.da_init_epoch:
                        self.experiment.log_metric("train_epoch da loss", da_loss, epoch=self.current_epoch)
            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            auroc, auprc, val_loss = self.test(dataloader="val")
            if self.experiment:
                self.experiment.log_metric("valid_epoch model loss", val_loss, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auroc", auroc, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auprc", auprc, epoch=self.current_epoch)
            val_lst = ["seed " + str(self.seed)] + ["epoch " + str(self.current_epoch)] + list(
                map(float2str, [auroc, auprc, val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(auroc)
            if auroc >= self.best_auroc:
                self.best_model = copy.deepcopy(self.model)
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch
            print('Validation at Epoch ' + str(self.current_epoch) + ' with validation loss ' + str(val_loss), " AUROC "
                  + str(auroc) + " AUPRC " + str(auprc))
            tauroc, tauprc, tf1, tacc, t_loss = self.test(dataloader="tval")
            print('TTValidation at Epoch ' + str(self.current_epoch) + ' with validation loss ' + str(t_loss), " TAUROC "
                  + str(tauroc) + " TAUPRC " + str(tauprc) + " Tf1 " + str(tf1) + " Tacc " + str(tacc))
            
            if self.current_epoch == 180 or self.current_epoch == 190 or self.current_epoch ==195:
                torch.save(self.model.state_dict(), os.path.join(pth_last3_pth, f"last_model_epoch_{self.current_epoch}.pth"))
                auroc, auprc, f1, recall, accuracy, test_loss, precision = self.test(
                    dataloader="test")  # thred_optim, precision
                test_lst = ["seed " + str(self.seed)] + ["epoch " + str(self.current_epoch)] + list(
                    map(float2str, [auroc, auprc, f1, recall,
                                    accuracy, precision, test_loss]))  # thred_optim,
                self.test_table.add_row(test_lst)
                print('Test at Model of Epoch ' + str(self.current_epoch) + ' with test loss ' + str(test_loss),
                      " AUROC "
                      + str(auroc) + " AUPRC " + str(auprc) + " f1-score " + str(f1) + " Recall " +
                      str(recall) + " Accuracy " + str(accuracy) + " Precision " + str(
                          precision))  # + + " Thred_optim " str(thred_optim)
        auroc, auprc, f1, recall, accuracy, test_loss, precision = self.test(
            dataloader="test")  # thred_optim, precision
        test_lst = ["seed " + str(self.seed)] + ["epoch " + str(self.best_epoch)] + list(
            map(float2str, [auroc, auprc, f1, recall,
                            accuracy, precision, test_loss]))  # thred_optim,
        self.test_table.add_row(test_lst)
        print('Test at Best Model of Epoch ' + str(self.best_epoch) + ' with test loss ' + str(test_loss), " AUROC "
              + str(auroc) + " AUPRC " + str(auprc) + " f1-score " + str(f1) + " Recall " +
              str(recall) + " Accuracy " + str(accuracy) + " Precision " + str(
            precision))  # + + " Thred_optim " str(thred_optim)
        self.test_metrics["auroc"] = auroc
        self.test_metrics["auprc"] = auprc
        self.test_metrics["test_loss"] = test_loss
        self.test_metrics["recall"] = recall
        self.test_metrics["accuracy"] = accuracy
        # self.test_metrics["thred_optim"] = thred_optim
        self.test_metrics["best_epoch"] = self.best_epoch
        self.test_metrics["F1"] = f1
        self.test_metrics["Precision"] = precision
        self.save_result()
        if self.experiment:
            self.experiment.log_metric("valid_best_auroc", self.best_auroc)
            self.experiment.log_metric("valid_best_epoch", self.best_epoch)
            self.experiment.log_metric("test_auroc", self.test_metrics["auroc"])
            self.experiment.log_metric("test_auprc", self.test_metrics["auprc"])
            self.experiment.log_metric("test_sensitivity", self.test_metrics["sensitivity"])
            self.experiment.log_metric("test_specificity", self.test_metrics["specificity"])
            self.experiment.log_metric("test_accuracy", self.test_metrics["accuracy"])
            self.experiment.log_metric("test_threshold", self.test_metrics["thred_optim"])
            self.experiment.log_metric("test_f1", self.test_metrics["F1"])
            self.experiment.log_metric("test_precision", self.test_metrics["Precision"])
        return self.test_metrics

    def save_result(self):
        output_path = os.path.join(self.output_dir, str(self.seed))
        os.makedirs(output_path, exist_ok=True)  # 如果目录不存在，创建它
        if self.config["RESULT"]["SAVE_MODEL"]:
            # 构造完整的文件路径
            torch.save(self.best_model.state_dict(),
                       os.path.join(output_path, f"best_model_epoch_{self.best_epoch}.pth"))
            torch.save(self.model.state_dict(), os.path.join(output_path, f"model_epoch_{self.current_epoch}.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }
        if self.is_da:
            state["train_model_loss"] = self.train_model_loss_epoch
            state["train_da_loss"] = self.train_da_loss_epoch
            state["da_init_epoch"] = self.da_init_epoch
        torch.save(state, os.path.join(output_path, f"result_metrics.pt"))

        val_prettytable_file = os.path.join(output_path, "valid_markdowntable.txt")
        test_prettytable_file = os.path.join(output_path, "test_markdowntable.txt")
        train_prettytable_file = os.path.join(output_path, "train_markdowntable.txt")
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())

    def _compute_entropy_weights(self, logits):
        entropy = entropy_logits(logits)
        entropy = ReverseLayerF.apply(entropy, self.alpha)
        entropy_w = 1.0 + torch.exp(-entropy)
        return entropy_w

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        for i, (v_d, v_p, labels) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1  # v_d是64个图
            v_d = torch.tensor(v_d, dtype=torch.float32)
            # v_d = torch.reshape(v_d, (v_d.shape[0], 1, 384))
            v_p, labels = v_p.to(self.device), labels.float().to(self.device)
            self.optim.zero_grad()
            v_d, v_p, f, score = self.model(v_d, v_p)
            if self.n_class == 1:
                n, loss = binary_cross_entropy(score, labels)
            else:
                n, loss = cross_entropy_logits(score, labels)
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
            if self.experiment:
                self.experiment.log_metric("train_step model loss", loss.item(), step=self.step)
        loss_epoch = loss_epoch / num_batches
        print('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(loss_epoch))
        return loss_epoch

    def train_da_epoch(self):
        self.model.train()
        total_loss_epoch = 0
        model_loss_epoch = 0
        da_loss_epoch = 0
        epoch_lamb_da = 0
        if self.current_epoch >= self.da_init_epoch:
            # epoch_lamb_da = self.da_lambda_decay()
            epoch_lamb_da = 1
            if self.experiment:
                self.experiment.log_metric("D loss lambda", epoch_lamb_da, epoch=self.current_epoch)
        num_batches = len(self.train_dataloader)  # 466
        for i, (batch_s, batch_t) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            v_d, v_p, labels = batch_s[0].to(self.device), batch_s[1].to(self.device), batch_s[2].float().to(
                self.device)
            v_d_t, v_p_t = batch_t[0].to(self.device), batch_t[1].to(self.device)
            self.optim.zero_grad()
            self.optim_da.zero_grad()
            v_d, v_p, f, score = self.model(v_d, v_p)  # 源域数据集进行模型训练
            if self.n_class == 1:
                n, model_loss = binary_cross_entropy(score, labels)
            else:
                n, model_loss = cross_entropy_logits(score, labels)
            # if self.current_epoch >= self.da_init_epoch:  # 源域先训练多少轮之后，再进行目标域训练
            v_d_t, v_p_t, f_t, t_score = self.model(v_d_t, v_p_t)  # 目标域数据集进行模型训练
            if self.da_method == "CDAN":
                reverse_f = ReverseLayerF.apply(f, self.alpha)
                softmax_output = torch.nn.Softmax(dim=1)(score)
                softmax_output = softmax_output.detach()
                # reverse_output = ReverseLayerF.apply(softmax_output, self.alpha)
                if self.original_random:
                    random_out = self.random_layer.forward([reverse_f, softmax_output])  # 源域的数据
                    adv_output_src_score = self.domain_dmm(random_out.view(-1, random_out.size(1)))  # (32,2)
                else:
                    feature = torch.bmm(softmax_output.unsqueeze(2), reverse_f.unsqueeze(1))
                    feature = feature.view(-1, softmax_output.size(1) * reverse_f.size(1))
                    if self.random_layer:
                        random_out = self.random_layer.forward(feature)
                        adv_output_src_score = self.domain_dmm(random_out)
                    else:
                        adv_output_src_score = self.domain_dmm(feature)

                reverse_f_t = ReverseLayerF.apply(f_t, self.alpha)
                softmax_output_t = torch.nn.Softmax(dim=1)(t_score)
                softmax_output_t = softmax_output_t.detach()
                # reverse_output_t = ReverseLayerF.apply(softmax_output_t, self.alpha)
                if self.original_random:
                    random_out_t = self.random_layer.forward([reverse_f_t, softmax_output_t])
                    adv_output_tgt_score = self.domain_dmm(random_out_t.view(-1, random_out_t.size(1)))  # (32,2)
                else:
                    feature_t = torch.bmm(softmax_output_t.unsqueeze(2), reverse_f_t.unsqueeze(1))
                    feature_t = feature_t.view(-1, softmax_output_t.size(1) * reverse_f_t.size(1))
                    if self.random_layer:
                        random_out_t = self.random_layer.forward(feature_t)
                        adv_output_tgt_score = self.domain_dmm(random_out_t)
                    else:
                        adv_output_tgt_score = self.domain_dmm(feature_t)

                if self.use_da_entropy:
                    entropy_src = self._compute_entropy_weights(score)
                    entropy_tgt = self._compute_entropy_weights(t_score)
                    src_weight = entropy_src / torch.sum(entropy_src)
                    tgt_weight = entropy_tgt / torch.sum(entropy_tgt)
                else:
                    src_weight = None
                    tgt_weight = None

                n_src, loss_cdan_src = cross_entropy_logits(adv_output_src_score, torch.zeros(self.batch_size).to(self.device),
                                                            src_weight)
                n_tgt, loss_cdan_tgt = cross_entropy_logits(adv_output_tgt_score, torch.ones(self.batch_size).to(self.device),
                                                            tgt_weight)
                da_loss = loss_cdan_src + loss_cdan_tgt
            else:
                raise ValueError(f"The da method {self.da_method} is not supported")
            loss = model_loss + da_loss
            # else:
            #     loss = model_loss
            loss.backward()
            self.optim.step()
            self.optim_da.step()
            total_loss_epoch += loss.item()
            model_loss_epoch += model_loss.item()
            if self.experiment:
                self.experiment.log_metric("train_step model loss", model_loss.item(), step=self.step)
                self.experiment.log_metric("train_step total loss", loss.item(), step=self.step)
            if self.current_epoch >= self.da_init_epoch:
                da_loss_epoch += da_loss.item()
                if self.experiment:
                    self.experiment.log_metric("train_step da loss", da_loss.item(), step=self.step)
        total_loss_epoch = total_loss_epoch / num_batches
        model_loss_epoch = model_loss_epoch / num_batches
        da_loss_epoch = da_loss_epoch / num_batches
        if self.current_epoch < self.da_init_epoch:
            print('Training at Epoch ' + str(self.current_epoch) + ' with model training loss ' + str(total_loss_epoch))
        else:
            print('Training at Epoch ' + str(self.current_epoch) + ' model training loss ' + str(model_loss_epoch)
                  + ", da loss " + str(da_loss_epoch) + ", total training loss " + str(total_loss_epoch) + ", D lambda " +
                  str(epoch_lamb_da))
        return total_loss_epoch, model_loss_epoch, da_loss_epoch, epoch_lamb_da

    def test(self, dataloader="test"):
        test_loss = 0
        y_label, y_pred = [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        elif dataloader == "tval":
            data_loader = self.test_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        with torch.no_grad():
            self.model.eval()
            for i, (v_d, v_p, labels) in enumerate(data_loader):
                v_d, v_p, labels = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device)
                if dataloader == "val":
                    v_d, v_p, f, score = self.model(v_d, v_p)
                elif dataloader == "test":
                    v_d, v_p, f, score = self.best_model(v_d, v_p)
                elif dataloader == "tval":
                    v_d, v_p, f, score = self.model(v_d, v_p)
                if self.n_class == 1:
                    n, loss = binary_cross_entropy(score, labels)
                else:
                    n, loss = cross_entropy_logits(score, labels)
                test_loss += loss.item()
                y_label = y_label + labels.to("cpu").tolist()
                y_pred = y_pred + n.to("cpu").tolist()
        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        test_loss = test_loss / num_batches

        if dataloader == "test":
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec, recall, _ = precision_recall_curve(y_label, y_pred)
            precision = tpr / (tpr + fpr)
            f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
            thred_optim = thresholds[5:][np.argmax(f1[5:])]
            y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
            cm1 = confusion_matrix(y_label, y_pred_s)
            recall1 = recall_score(y_label, y_pred_s)
            accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
            precision1 = precision_score(y_label, y_pred_s)
            return auroc, auprc, np.max(f1[5:]), recall1, accuracy, test_loss, precision1
        elif dataloader == "tval":
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec, recall, _ = precision_recall_curve(y_label, y_pred)
            precision = tpr / (tpr + fpr)
            f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
            thred_optim = thresholds[5:][np.argmax(f1[5:])]
            y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
            cm1 = confusion_matrix(y_label, y_pred_s)
            accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
            return auroc, auprc, np.max(f1[5:]), accuracy, test_loss
            
        else:
            return auroc, auprc, test_loss


