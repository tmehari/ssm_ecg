
import yaml
import tensorboard
import torch
import torch.nn as nn
import os
import shutil
import sys
import csv
import argparse
import pickle

from dl_models.cpc import CPCModel
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score

from clinical_ts.ecg_dataset_wrapper import ECGDataSetWrapper
from clinical_ts.eval_utils_cafa import eval_scores, eval_scores_bootstrap
from clinical_ts.timeseries_utils import aggregate_predictions
# from clinical_ts.pesg_auc import auc_loss, PESG_AUC
import pdb
from copy import deepcopy
from os.path import join, isdir
from dl_models.s4_model import S4Model
# from clinical_ts.data_modifiers import ToOneHot
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ptb_num_classes = {"label_all": 71, "label_diag": 44, "label_form": 19,
                   "label_rhythm": 12, "label_diag_subclass": 23, "label_diag_superclass": 5}
chapman_num_classes = {"label_all": 67,
                     "label_condition": 56, "label_rhythm": 11, "label_rhythm_filtered":9}


def parse_args():
    parser = argparse.ArgumentParser("Finetuning tests")
    parser.add_argument("--model_file")
    parser.add_argument("--method")
    parser.add_argument("--target_folder",
                        default="./data/ptb_xl_fs100")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--discriminative_lr",
                        default=False, action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--hidden", default=False, action="store_true")
    parser.add_argument("--lr_schedule", default="{}")
    parser.add_argument("--use_pretrained", default=False, action="store_true")
    parser.add_argument("--linear_evaluation",
                        default=False, action="store_true", help="use linear evaluation")
    parser.add_argument("--test_noised", default=False,
                        action="store_true", help="validate also on a noisy dataset")
    parser.add_argument("--noise_level", default=1, type=int,
                        help="level of noise induced to the second validations set")
    parser.add_argument("--folds", default=8, type=int,
                        help="number of folds used in finetuning (between 1-8)")
    parser.add_argument("--tag", default="")
    parser.add_argument("--eval_only", action="store_true",
                        default=False, help="only evaluate mode")
    parser.add_argument("--load_finetuned", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--cpc", action="store_true", default=False)
    parser.add_argument("--model_location")
    parser.add_argument("--l_epochs", type=int, default=0,
                        help="number of head-only epochs (these are performed first)")
    parser.add_argument("--f_epochs", type=int, default=0,
                        help="number of finetuning epochs (these are perfomed after head-only training")
    parser.add_argument("--normalize", action="store_true", default=False,
                        help="normalize dataset with ptbxl mean and std")
    parser.add_argument("--bn_head", action="store_true", default=False)
    parser.add_argument("--ps_head", type=float, default=0.0)
    parser.add_argument("--conv_encoder", action="store_true", default=False)
    parser.add_argument("--base_model", default="xresnet1d50")
    parser.add_argument("--widen", default=1, type=int,
                        help="use wide xresnet1d50")
    parser.add_argument("--auc_maximization",
                        action="store_true", default=False)
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--label_class", default="label_all")
    parser.add_argument("--num_classes", type=int, default=71)
    parser.add_argument("--ptb_to_zheng", default=False, action="store_true",
                        help="use when evaluating a ptb-trained model on the zheng dataset, takes charge of mapping")
    parser.add_argument("--filter_labels", nargs="+", default=[])
    parser.add_argument("--trial_nr", default="1")
    parser.add_argument("--n_hidden", default=512, type=int)
    parser.add_argument("--input_size", default=250, type=int)
    parser.add_argument("--save_model_at")
    parser.add_argument("--use_meta_information_in_head", action="store_true", default=False)
    args = parser.parse_args()
    return args


def get_new_state_dict(init_state_dict, lightning_state_dict, method="simclr", model='resnet'):
    # in case of moco model

    from collections import OrderedDict
    # lightning_state_dict = lightning_state_dict["state_dict"]
    new_state_dict = OrderedDict()

    if method != "cpc" and method != "s4":
        if method == "moco":
            for key in init_state_dict:
                l_key = "encoder_q." + key
                if l_key in lightning_state_dict.keys():
                    new_state_dict[key] = lightning_state_dict[l_key]
        elif method == "simclr":
            for key in init_state_dict:
                if "features" in key:
                    l_key = key.replace("features", "encoder.features")
                if l_key in lightning_state_dict.keys():
                    new_state_dict[key] = lightning_state_dict[l_key]
        elif method == "swav" or method == "stab":
            if model == "lenet":
                for key in init_state_dict:
                    l_key = "model." + key
                    #l_key = "model.features." + key
                    if l_key in lightning_state_dict.keys():
                        new_state_dict[key] = lightning_state_dict[l_key]

                return new_state_dict

            for key in init_state_dict:

                if "features" in key:
                    l_key = key.replace("features", "model.features")
                if l_key in lightning_state_dict.keys():
                    new_state_dict[key] = lightning_state_dict[l_key]
            if method == "stab":
                new_state_dict["l1.weight"] = lightning_state_dict["model.l1.weight"]
                new_state_dict["l1.bias"] = lightning_state_dict["model.l1.bias"]
                if "l2.weight" in init_state_dict.keys():
                    new_state_dict["l2.weight"] = lightning_state_dict["model.l2.weight"]
                    new_state_dict["l2.bias"] = lightning_state_dict["model.l2.bias"]
                return new_state_dict
        elif method == "byol":
            for key in init_state_dict:
                l_key = "online_network.encoder." + key
                if l_key in lightning_state_dict.keys():
                    new_state_dict[key] = lightning_state_dict[l_key]
        else:
            raise("method unknown")
        new_state_dict["l1.weight"] = init_state_dict["l1.weight"]
        new_state_dict["l1.bias"] = init_state_dict["l1.bias"]
        if "l2.weight" in init_state_dict.keys():
            new_state_dict["l2.weight"] = init_state_dict["l2.weight"]
            new_state_dict["l2.bias"] = init_state_dict["l2.bias"]

        assert(len(init_state_dict) == len(new_state_dict))
    else:
        for key in init_state_dict:
            l_key = "model_cpc." + key
            # l_key = "model." + key
            if l_key in lightning_state_dict.keys():
                new_state_dict[key] = lightning_state_dict[l_key]
            if "head" in key:
                new_state_dict[key] = init_state_dict[key]

    return new_state_dict


def adjust(model, num_classes, hidden=False):

    in_features = model.l1.in_features
    last_layer = torch.nn.modules.linear.Linear(
        in_features, num_classes).to(device)
    if hidden:
        model.l1 = torch.nn.modules.linear.Linear(
            in_features, in_features).to(device)
        model.l2 = last_layer
    else:
        model.l1 = last_layer

    def def_forward(self):
        def new_forward(x):
            h = self.features(x)
            h = h.squeeze()

            x = self.l1(h)
            if hidden:
                x = F.relu(x)
                x = self.l2(x)
            return x
        return new_forward

    model.forward = def_forward(model)


def configure_optimizer(model, batch_size, head_only=False, lr=None, discriminative_lr=False,
                        base_model="xresnet1d", optimizer="adamw", 
                        discriminative_lr_factor=1, auc_maximization=False, 
                        num_classes=71, use_meta_information_in_head=False):

    if auc_maximization:
        loss_fn = auc_loss([1./num_classes]*num_classes).to(device)
    else:
        loss_fn = F.binary_cross_entropy_with_logits
    if(optimizer == "sgd"):
        opt = torch.optim.SGD
        print("use SGD optimizer")
    elif(optimizer == "adamw"):
        opt = torch.optim.AdamW
        print("use AdamW optmizer")
    elif(optimizer == "pesg"):
        opt = PESG_AUC
        print("Pesg optimizer")
    else:
        raise NotImplementedError("Unknown Optimizer.")
    if 'resnet' in base_model:
        wd = 1e-1
        if head_only:
            lr = (8e-3*(batch_size/256))
            optimizer = opt(
                model.l1.parameters(), lr=lr, weight_decay=wd)
        else:
            lr = 0.01
            if not discriminative_lr:
                optimizer = opt(
                    model.parameters(), lr=lr, weight_decay=wd)
            else:
                param_dict = dict(model.named_parameters())
                keys = param_dict.keys()
                weight_layer_nrs = set()
                for key in keys:
                    if "features" in key:
                        # parameter names have the form features.x
                        weight_layer_nrs.add(key[9])
                weight_layer_nrs = sorted(weight_layer_nrs, reverse=True)
                features_groups = []
                while len(weight_layer_nrs) > 0:
                    if len(weight_layer_nrs) > 1:
                        features_groups.append(list(filter(
                            lambda x: "features." + weight_layer_nrs[0] in x or "features." + weight_layer_nrs[1] in x,  keys)))
                        del weight_layer_nrs[:2]
                    else:
                        features_groups.append(
                            list(filter(lambda x: "features." + weight_layer_nrs[0] in x,  keys)))
                        del weight_layer_nrs[0]
                # filter linear layers
                linears = list(filter(lambda x: "l" in x, keys))
                groups = [linears] + features_groups
                optimizer_param_list = []
                tmp_lr = lr

                for layers in groups:
                    layer_params = [param_dict[param_name]
                                    for param_name in layers]
                    optimizer_param_list.append(
                        {"params": layer_params, "lr": tmp_lr})
                    tmp_lr /= 4
                optimizer = torch.optim.AdamW(
                    optimizer_param_list, lr=lr, weight_decay=wd)

        print("lr", lr)
        print("wd", wd)
        print("batch size", batch_size)

    elif base_model == "cpc" or base_model =='s4':
        lr = 1e-4
        wd = 1e-3
        if(head_only):
            lr = 1e-3
            print("Head only training, model head:", model.head)
            if use_meta_information_in_head:
                print("Meta Head:", model.meta_head)
                optimizer = opt([
                    {'params': model.head.parameters(),"lr": lr},
                    {'params': model.meta_head.parameters(), "lr": lr}]
                                , lr, weight_decay=wd)
            else:
                optimizer = opt(model.head.parameters(), lr, weight_decay=wd)
        elif(discriminative_lr_factor != 1.):  # discrimative lrs
            paramgroup = [{
                            "params": model.encoder.parameters(), 
                            "lr": lr*discriminative_lr_factor*discriminative_lr_factor
                            }, 
                            {
                            "params": model.rnn.parameters(), 
                            "lr": lr*discriminative_lr_factor
                            }, 
                            {"params": model.head.parameters(), 
                            "lr": lr
                            }]
            if use_meta_information_in_head:
               paramgroup.append(
                           {"params": model.meta_head.parameters(), 
                           "lr": lr*discriminative_lr_factor
                           }
                           )
            optimizer = opt(paramgroup, 
                            lr, 
                            weight_decay=wd)
            print("Finetuning: model head", model.head)
            if use_meta_information_in_head:
                print("Finetuning: model meta head", model.meta_head)
            print("discriminative lr: ", discriminative_lr_factor)
            print('lr',lr)
        else:
            lr = 1e-3
            print("normal supervised training")
            optimizer = opt(model.parameters(), lr, weight_decay=wd)
    elif base_model == "lenet":
        lr = 1e-4
        wd = 1e-3
        if(head_only):
            lr = 1e-2
            print("Head only: model", model)
            optimizer = opt(model[-1].parameters(), lr, weight_decay=wd)
        else:
            print("normal supervised training")
            optimizer = opt(model.parameters(), lr, weight_decay=wd)
    else:
        raise("model unknown")
    return loss_fn, optimizer


def load_model(linear_evaluation, num_classes, use_pretrained, discriminative_lr=False, hidden=False, conv_encoder=False, bn_head=False, ps_head=0.5,
               location="./checkpoints/moco_baselinewonder200.ckpt", method="simclr",
               base_model="xresnet1d50", out_dim=71, widen=1, n_hidden=512,
               use_meta_information_in_head=False):
    discriminative_lr_factor = 1
    
    def tlenet():
        bn = True
        ps = False
        c1 = 12  # b/c single time-series
        c2 = 32  # 4
        c3 = 64  # 16
        c4 = 128  # 32
        k = 5  # kernel size #7
        s = 2  # stride #3
        E = 128  # 128
        modules = []
        modules.append(nn.Conv1d(c1, c2, k, s))
        if bn:
            modules.append(nn.BatchNorm1d(c2))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool1d(2))
        if ps:
            modules.append(nn.Dropout(0.1))
        modules.append(nn.Conv1d(c2, c3, k, s))
        if bn:
            modules.append(nn.BatchNorm1d(c3))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool1d(2))
        if ps:
            modules.append(nn.Dropout(0.1))
        modules.append(nn.Conv1d(c3, c4, k, s))
        if bn:
            modules.append(nn.BatchNorm1d(c4))
        modules.append(nn.ReLU())
        # modules.append(nn.MaxPool1d(2))
        modules.append(nn.AdaptiveAvgPool1d(2))
        if ps:
            modules.append(nn.Dropout(0.1))
        modules.append(nn.Flatten())
        modules.append(nn.Linear(2*c4, E))  # 96
        modules.append(nn.ReLU())
        if args.hidden:
            modules.append(nn.Dropout(0.5))
            modules.append(nn.Linear(E, E))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(0.5))
        modules.append(nn.Linear(E, out_dim))
        mod = nn.Sequential(*modules)
        return mod

    if use_pretrained:
        print("load model from " + location)
        discriminative_lr_factor = 0.1
        if base_model == "cpc":
            lightning_state_dict = torch.load(location, map_location=device)

            # num_head = np.sum([1 if 'proj' in f else 0 for f in lightning_state_dict.keys()])
            if linear_evaluation:
                lin_ftrs_head = []
                bn_head = False
                ps_head = 0.0
            else:
                if hidden:
                    lin_ftrs_head = [512]
                else:
                    lin_ftrs_head = []

            if conv_encoder:
                strides = [2, 2, 2, 2]
                kss = [10, 4, 4, 4]
            else:
                strides = [1]*4
                kss = [1]*4

            # model = CPCModel(input_channels=12, strides=strides, kss=kss, features=[512]*4, n_hidden=n_hidden, n_layers=2, mlp=False, lstm=True, bias_proj=False,
            #                 num_classes=num_classes, skip_encoder=False, bn_encoder=True, lin_ftrs_head=lin_ftrs_head, ps_head=ps_head, bn_head=bn_head).to(device)
            # model = CPCModel(input_channels=12, strides = [1]*4, kss = [1]*4, features=[512]*4, n_hidden=512, n_layers=2, num_classes=num_classes, lin_ftrs_head=[512], bn_encoder=True).to(device)

            # small cpc
            model = CPCModel(input_channels=12, strides=[1]*4, kss=[1]*4, features=[
                             512]*4, n_hidden=512, n_layers=2, num_classes=num_classes, lin_ftrs_head=[512], bn_encoder=True, 
                             concat_pooling=True, bn_head=True, ps_head=0.5).to(device)

            if "state_dict" in lightning_state_dict.keys():
                print("load pretrained model")
                model_state_dict = get_new_state_dict(
                    model.state_dict(), lightning_state_dict["state_dict"], method="cpc")
            else:
                print("load already finetuned model")
                model_state_dict = lightning_state_dict

            model.load_state_dict(model_state_dict)
        elif base_model == "lenet":
            model = tlenet().to(device)
            model_state_dict = torch.load(location, map_location=device)
            if "state_dict" in model_state_dict.keys():
                model_state_dict = model_state_dict["state_dict"]
            if "l1.weight" in model_state_dict.keys():  # load already fine-tuned model
                model_classes = model_state_dict["l1.weight"].shape[0]
                if model_classes != num_classes:
                    raise Exception("Loaded model has different output dim ({}) than needed ({})".format(
                        model_classes, num_classes))
                adjust(model, num_classes, hidden=hidden)
                if not hidden and "l2.weight" in model_state_dict:
                    del model_state_dict["l2.weight"]
                    del model_state_dict["l2.bias"]
                model.load_state_dict(model_state_dict)
            else:  # load pretrained model
                base_dict = model.state_dict()
                model_state_dict = get_new_state_dict(
                    base_dict, model_state_dict, method=method, model="lenet")
                model.load_state_dict(model_state_dict)
                #  adjust(model, num_classes, hidden=hidden)
            # init_state_dict = model.state_dict()
            # new_state_dict = {}
            # for key in init_state_dict:
            #     l_key = 'model.' + key
            #     if l_key in lightning_state_dict.keys():
            #         new_state_dict[key] = lightning_state_dict[l_key]
            # pdb.set_trace()
            # model.load_state_dict(new_state_dict)
        elif base_model == "s4":
            lightning_state_dict = torch.load(location, map_location=device)
            strides=[1]*4 
            kss = [1]*4 
            features = [512]*4
            if linear_evaluation:
                ps_head = 0.0
                lin_ftrs_head = None 
                bn_head = False
            else:
                ps_head = ps_head
                lin_ftrs_head = [512] if hidden else None 
                bn_head = bn_head
            model = CPCModel(input_channels=12, num_classes=num_classes, strides=strides,
                             kss=kss,features=features, mlp=True, bn_encoder=True,
                             lin_ftrs_head=lin_ftrs_head, s4=True,bn_head=bn_head, 
                             ps_head=ps_head, s4_d_model=512, s4_d_state=8, s4_l_max= 1024,
                             concat_pooling=False, use_meta_information_in_head=use_meta_information_in_head).to(device)
            if "state_dict" in lightning_state_dict.keys():
                print("load pretrained model")
                model_state_dict = get_new_state_dict(
                    model.state_dict(), lightning_state_dict["state_dict"], method="cpc")
            else:
                print("load already finetuned model")
                model_state_dict = lightning_state_dict
            model.load_state_dict(model_state_dict)
        else:
            model = ResNetSimCLR(base_model, num_classes,
                                 hidden=hidden, widen=widen).to(device)
            model_state_dict = torch.load(location, map_location=device)
            if "state_dict" in model_state_dict.keys():
                model_state_dict = model_state_dict["state_dict"]
            if "l1.weight" in model_state_dict.keys():  # load already fine-tuned model
                model_classes = model_state_dict["l1.weight"].shape[0]
                if model_classes != num_classes:
                    raise Exception("Loaded model has different output dim ({}) than needed ({})".format(
                        model_classes, num_classes))
                adjust(model, num_classes, hidden=hidden)
                if not hidden and "l2.weight" in model_state_dict:
                    del model_state_dict["l2.weight"]
                    del model_state_dict["l2.bias"]

                model.load_state_dict(model_state_dict)
            else:  # load pretrained model
                base_dict = model.state_dict()
                model_state_dict = get_new_state_dict(
                    base_dict, model_state_dict, method=method)

                # adjust(model, num_classes, hidden=hidden)
                if method != "stab":
                    model.load_state_dict(model_state_dict)
                    adjust(model, num_classes, hidden=hidden)
                else:
                    def def_forward(self):
                        def new_forward(x):
                            h = self.features(x)
                            h = h.squeeze()

                            x = self.l1(h)
                            if hidden:
                                x = F.relu(x)
                                x = self.l2(x)
                            return x
                        return new_forward

                    model.forward = def_forward(model)
                    model.load_state_dict(model_state_dict)

    else:
        if "xresnet1d" in base_model:
            model = ResNetSimCLR(base_model, out_dim,
                                 hidden=hidden, widen=widen).to(device)
            adjust(model, num_classes, hidden=hidden)
        elif base_model == "cpc":
            if linear_evaluation:
                lin_ftrs_head = []
                bn_head = False
                ps_head = 0.0
            else:
                if hidden:
                    lin_ftrs_head = [512]
                else:
                    lin_ftrs_head = []

            if conv_encoder:
                strides = [2, 2, 2, 2]
                kss = [10, 4, 4, 4]
            else:
                strides = [1]*4
                kss = [1]*4

            # model = CPCModel(input_channels=12, strides=strides, kss=kss, features=[512]*4, n_hidden=n_hidden, n_layers=2, mlp=False, lstm=True, bias_proj=False,
            #                 num_classes=num_classes, skip_encoder=False, bn_encoder=True, lin_ftrs_head=lin_ftrs_head, ps_head=ps_head, bn_head=bn_head).to(device)
            # model = CPCModel(input_channels=12, strides = [1]*4, kss = [1]*4, features=[512]*4, n_hidden=512, n_layers=2, num_classes=num_classes, lin_ftrs_head=[512], bn_encoder=True).to(device)
            model = CPCModel(input_channels=12, strides=[1]*4, kss=[1]*4, features=[
                             512]*4, n_hidden=512, n_layers=2, num_classes=num_classes, lin_ftrs_head=[512], bn_encoder=True).to(device)
        elif base_model == 's4':
            strides=[1]*4 
            kss = [1]*4 
            features = [512]*4
            if linear_evaluation:
                ps_head = 0.0
                lin_ftrs_head = None 
                bn_head = False
            else:
                ps_head = ps_head
                lin_ftrs_head = [512] if hidden else None 
                bn_head = bn_head
            model = CPCModel(input_channels=12, num_classes=num_classes, strides=strides, 
                             kss=kss, features=features, mlp=True,
             bn_encoder=True, ps_head=ps_head, bn_head=bn_head, lin_ftrs_head=lin_ftrs_head,
             s4=True, s4_d_model=512, s4_d_state=8, s4_l_max= 1024, 
             use_meta_information_in_head=use_meta_information_in_head).to(device)
        else:
            raise Exception("model unknown")

    return model


def evaluate(model, dataloader, idmap, lbl_itos, base_model='xresnet1d',
             ptb_to_zheng=False, use_meta_information_in_head=False):
    if use_meta_information_in_head:
        preds, targs = eval_model_with_meta(model, dataloader, base_model=base_model)
    else:
        preds, targs = eval_model(model, dataloader, base_model=base_model)
    
    if ptb_to_zheng:
        mapping = get_ptb_to_zheng()
        preds = preds[:, torch.tensor(mapping, dtype=torch.int)]

    preds = torch.sigmoid(torch.tensor(preds)).numpy() # improves performance for aggregation
    scores = eval_scores(targs, preds, classes=lbl_itos, parallel=True)
    preds_agg, targs_agg = aggregate_predictions(preds, targs, idmap)
    # scores_agg = scores
    scores_agg = eval_scores(targs_agg, preds_agg,
                             classes=lbl_itos, parallel=True)
    # print(scores_agg)
    macro = scores["label_AUC"]["macro"]
    macro_agg = scores_agg["label_AUC"]["macro"]

    # preds1 = np.argmax(preds_agg, axis=1)
    # targs1 = np.argmax(targs_agg, axis=1)
    # cm = confusion_matrix(y_target=targs1,
    #                       y_predicted=preds1,
    #                       binary=False)
    # fig, ax = plot_confusion_matrix(conf_mat=cm, class_names=lbl_itos)
    # plt.title("Multi Label Confusion matrix for the Large Dataset")
    # plt.show()
    return preds, macro, macro_agg, scores_agg


def set_train_eval(model, cpc, linear_evaluation):
    if linear_evaluation:
        if cpc:
            model.encoder.eval()
        else:
            model.features.eval()
    else:
        model.train()


def train_model(model, train_loader, valid_loader, test_loader, epochs, loss_fn, optimizer, 
                head_only=True, linear_evaluation=False, percentage=1, lr_schedule=None,
                save_model_at=None, val_idmap=None, test_idmap=None, lbl_itos=None,
                cpc=False, it_per_print=100, base_model='xresnet1d', use_meta_information_in_head=False):
    if head_only:
        if linear_evaluation:
            print("linear evaluation for {} epochs".format(epochs))
        else:
            print("head-only for {} epochs".format(epochs))
    else:
        print("fine tuning for {} epochs".format(epochs))

    if head_only:
        for key, param in model.named_parameters():
            if "l1." not in key and "head." not in key and "meta_head." not in key:
                param.requires_grad = False
        print("copying state dict before training for sanity check after training")

    else:
        for param in model.parameters():
            param.requires_grad = True
    if cpc:
        data_type = model.encoder[0][0].weight.type()
    else:
        if base_model == "lenet":
            data_type = model[0].weight.type()
            # data_type = model.features[0].weight.type()
        else:
            data_type = model.features[0][0].weight.type()

    # turned off because lenet has no .features
    set_train_eval(model, cpc, linear_evaluation)
    state_dict_pre = deepcopy(model.state_dict())
    print("epoch", "batch", "loss\n========================")
    loss_per_epoch = []
    macro_agg_per_epoch = []
    max_batches = len(train_loader)
    break_point = int(percentage*max_batches)
    best_macro = 0
    best_macro_agg = 0
    best_epoch = 0
    best_preds = None
    test_macro = 0
    test_macro_agg = 0
   
    _, _, macro_agg, _ = evaluate(
        model, valid_loader, val_idmap, lbl_itos, base_model=base_model,
        use_meta_information_in_head=use_meta_information_in_head)
    
    best_macro_agg = macro_agg
    set_train_eval(model, cpc, linear_evaluation)
    print('agg macro before training:', macro_agg)
    for epoch in tqdm(range(epochs)):
        if type(lr_schedule) == dict:
            if epoch in lr_schedule.keys():
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= lr_schedule[epoch]
        total_loss_one_epoch = 0
        for batch_idx, samples in enumerate(train_loader):
            if batch_idx == break_point:
                print("break at batch nr.", batch_idx)
                break
            data = samples[0].to(device).type(data_type)
            labels = samples[1].to(device).type(data_type)
            optimizer.zero_grad()
            if use_meta_information_in_head:
                static = samples[2].to(device).type(data_type)
                preds = model.forward_with_meta(data, static)
            else:
                preds = model(data)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss_one_epoch += loss.item()
            if(batch_idx % it_per_print == 0):
                print(epoch, batch_idx, loss.item())
        loss_per_epoch.append(total_loss_one_epoch)

        preds, macro, macro_agg, _ = evaluate(
            model, valid_loader, val_idmap, lbl_itos, base_model=base_model, 
            use_meta_information_in_head=use_meta_information_in_head)
        macro_agg_per_epoch.append(macro_agg)

        print("loss:", total_loss_one_epoch)
        print("aggregated macro:", macro_agg)
        if macro_agg > best_macro_agg:
            print('save model at', save_model_at)
            torch.save(model.state_dict(), save_model_at)
            best_macro_agg = macro_agg
            best_macro = macro
            best_epoch = epoch
            best_preds = preds
            _, test_macro, test_macro_agg, _ = evaluate(
                model, test_loader, test_idmap, lbl_itos,
                base_model=base_model, use_meta_information_in_head=use_meta_information_in_head)

        set_train_eval(model, cpc, linear_evaluation)

    if epochs > 0:
        sanity_check(model, state_dict_pre, linear_evaluation, head_only)
    return loss_per_epoch, macro_agg_per_epoch, best_macro, best_macro_agg, test_macro, test_macro_agg, best_epoch, best_preds


def sanity_check(model, state_dict_pre, linear_evaluation, head_only):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading state dict for sanity check")
    state_dict = model.state_dict()
    if linear_evaluation:
        for k in list(state_dict.keys()):
            # only ignore fc layer
            if 'fc.' in k or 'head.' in k or 'l1.' in k:
                continue

            equals = (state_dict[k].cpu() == state_dict_pre[k].cpu()).all()
            if (linear_evaluation != equals):
                raise Exception(
                    '=> failed sanity check in {}'.format("linear_evaluation"))
    elif head_only:
        for k in list(state_dict.keys()):
            # only ignore fc layer
            if 'fc.' in k or 'head.' in k:
                continue

            equals = (state_dict[k].cpu() == state_dict_pre[k].cpu()).all()
            if (equals and "running_mean" in k):
                raise Exception(
                    '=> failed sanity check in {}'.format("head-only"))
    # else:
    #     for k in list(state_dict.keys()):
    #         equals=(state_dict[k].cpu() == state_dict_pre[k].cpu()).all()
    #         if equals:
    #             pdb.set_trace()
    #             raise Exception('=> failed sanity check in {}'.format("fine_tuning"))

    print("=> sanity check passed.")


def eval_model(model, valid_loader, base_model='xresnet1d'):
    if 'cpc' in base_model or 's4' in base_model:
        data_type = model.encoder[0][0].weight.type()
    elif 'lenet' in base_model:
        data_type = model[0].weight.type()  # model.features[0]
    else:
        data_type = model.features[0][0].weight.type()
    model.eval()
    preds = []
    targs = []
    with torch.no_grad():
        for batch_idx, samples in tqdm(enumerate(valid_loader)):
            data = samples[0].to(device).type(data_type)
            preds_tmp = model(data)
            targs.append(samples[1])
            preds.append(preds_tmp.cpu())
        preds = torch.cat(preds).numpy()
        targs = torch.cat(targs).numpy()
        

    return preds, targs


def eval_model_with_meta(model, valid_loader, base_model='xresnet1d'):
    if 'cpc' in base_model or 's4' in base_model:
        data_type = model.encoder[0][0].weight.type()
    elif 'lenet' in base_model:
        data_type = model[0].weight.type()  # model.features[0]
    else:
        data_type = model.features[0][0].weight.type()
    model.eval()
    preds = []
    targs = []
    with torch.no_grad():
        for batch_idx, samples in tqdm(enumerate(valid_loader)):
            data = samples[0].to(device).type(data_type)
            static = samples[2].to(device).type(data_type)
            preds_tmp = model.forward_with_meta(data, static)
            targs.append(samples[1])
            preds.append(preds_tmp.cpu())
        preds = torch.cat(preds).numpy()
        targs = torch.cat(targs).numpy()
    return preds, targs


def get_dataset(batch_size, num_workers, target_folders, apply_noise=False, percentage=1.0, 
                folds=8, noise_level=0, test=False, normalize=False, label_class="label_all",
                stability_training=False, domain_adaptation=False, label_noise=False,
                p=0.01, filter_labels=[], input_size=250, plus_norm_as_background=False, 
                use_meta_information_in_head=False):
    t_params_by_level = {
        0: {},
        1: {"bw_cmax": 0.05, "em_cmax": 0.25, "pl_cmax": 0.1, "bs_cmax": 0.5},
        2: {"bw_cmax": 0.1, "em_cmax": 0.5, "pl_cmax": 0.2, "bs_cmax": 1},
        3: {"bw_cmax": 0.1, "em_cmax": 1, "pl_cmax": 0.2, "bs_cmax": 2},
        4: {"bw_cmax": 0.2, "em_cmax": 1, "pl_cmax": 0.4, "bs_cmax": 2},
        5: {"bw_cmax": 0.2, "em_cmax": 1.5, "pl_cmax": 0.4, "bs_cmax": 2.5},
        6: {"bw_cmax": 0.3, "em_cmax": 2, "pl_cmax": 0.5, "bs_cmax": 3},
    }
    transformations = []
    if noise_level > 0:
        transformations += ["BaselineWander",
                            "PowerlineNoise", "EMNoise", "BaselineShift"]
    if normalize:
        transformations.append("Normalize")
    
    lbl_itos_file = [x for x in os.listdir(target_folders[0]) if 'lbl_itos' in x][0]
    lbl_itos = np.load(join(target_folders[0], lbl_itos_file),
                       allow_pickle=True)[label_class]
    
    dataset = ECGDataSetWrapper(batch_size, num_workers, target_folders[0], normalize=normalize, 
        test=test, input_size=input_size, label=label_class, use_meta_information_in_head=use_meta_information_in_head)

    train_loader, valid_loader = dataset.get_data_loaders()
    return dataset, train_loader, valid_loader


def get_ptb_to_zheng():
    # ptb -> zheng
    mapping = {'SR': 'SR',
               'AFIB': 'AFIB',
               'STACH': 'ST',
               'SARRH': 'SI',
               'SBRAD': 'SB',
               'AFLT': 'AF',
               'SVTAC': 'SVT'}
    zheng_rhythm = np.array(['AF', 'AFIB', 'AT', 'AVNRT', 'AVRT', 'SA', 'SAAWR', 'SB', 'SR',
                             'ST', 'SVT'], dtype='<U5')
    ptb_rhythm = np.array(['STACH', 'AFLT', 'AFIB', 'SVTAC', 'SVARR', 'SARRH', 'SR', 'SBRAD',
                           'PACE', 'TRIGU', 'BIGU', 'PSVT'], dtype='<U5')
    dummy = "TRIGU"
    ptb_to_zheng = np.ones(len(zheng_rhythm)) * \
        int(np.where(ptb_rhythm == dummy)[0])

    for p_diag, z_diag in mapping.items():
        p_ind = np.where(ptb_rhythm == p_diag)[0]
        z_ind = np.where(zheng_rhythm == z_diag)[0]
        ptb_to_zheng[z_ind] = p_ind
    return ptb_to_zheng


def tlenet(hidden=True):
    bn = True
    ps = False
    c1 = 12  # b/c single time-series
    c2 = 32  # 4
    c3 = 64  # 16
    c4 = 128  # 32
    k = 5  # kernel size #7
    s = 2  # stride #3
    E = 128  # 128
    modules = []
    modules.append(nn.Conv1d(c1, c2, k, s))
    if bn:
        modules.append(nn.BatchNorm1d(c2))
    modules.append(nn.ReLU())
    modules.append(nn.MaxPool1d(2))
    if ps:
        modules.append(nn.Dropout(0.1))
    modules.append(nn.Conv1d(c2, c3, k, s))
    if bn:
        modules.append(nn.BatchNorm1d(c3))
    modules.append(nn.ReLU())
    modules.append(nn.MaxPool1d(2))
    if ps:
        modules.append(nn.Dropout(0.1))
    modules.append(nn.Conv1d(c3, c4, k, s))
    if bn:
        modules.append(nn.BatchNorm1d(c4))
    modules.append(nn.ReLU())
    # modules.append(nn.MaxPool1d(2))
    modules.append(nn.AdaptiveAvgPool1d(2))
    if ps:
        modules.append(nn.Dropout(0.1))
    modules.append(nn.Flatten())
    modules.append(nn.Linear(2*c4, E))  # 96
    modules.append(nn.ReLU())
    if hidden:
        modules.append(nn.Dropout(0.5))
        modules.append(nn.Linear(E, E))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(0.5))
    modules.append(nn.Linear(E, len(dataset.lbl_itos)))
    mod = nn.Sequential(*modules)
    return mod


def tlenet():
    class LeNetBackbone(nn.Module):
        def __init__(self):
            super(LeNetBackbone, self).__init__()
            bn = True
            ps = False
            hidden = False
            c1 = 12  # b/c single time-series
            c2 = 32  # 4
            c3 = 64  # 16
            c4 = 128  # 32
            k = 5  # kernel size #7
            s = 2  # stride #3
            E = 128  # 128
            modules = []
            modules.append(nn.Conv1d(c1, c2, k, s))
            if bn:
                modules.append(nn.BatchNorm1d(c2))
            modules.append(nn.ReLU())
            modules.append(nn.MaxPool1d(2))
            if ps:
                modules.append(nn.Dropout(0.1))
            modules.append(nn.Conv1d(c2, c3, k, s))
            if bn:
                modules.append(nn.BatchNorm1d(c3))
            modules.append(nn.ReLU())
            modules.append(nn.MaxPool1d(2))
            if ps:
                modules.append(nn.Dropout(0.1))
            modules.append(nn.Conv1d(c3, c4, k, s))
            if bn:
                modules.append(nn.BatchNorm1d(c4))
            # modules.append(nn.ReLU())
            # modules.append(nn.MaxPool1d(2))
            modules.append(nn.AdaptiveAvgPool1d(2))
            # if ps:
            #    modules.append(nn.Dropout(0.1))
            modules.append(nn.Flatten())
            # modules.append(nn.Linear(2*c4, E))  # 96
            # modules.append(nn.ReLU())
            if hidden:
                modules.append(nn.Dropout(0.5))
                modules.append(nn.Linear(E, E))
                modules.append(nn.ReLU())
                modules.append(nn.Dropout(0.5))
            # modules.append(nn.Linear(2*c4, ptb_num_classes)) # changed E to 2*c4
            self.features = nn.Sequential(*modules)
            self.l1 = nn.Linear(2*c4, 128)

        def forward(self, x):
            h = self.features(x)
            h = h.squeeze()
            x = self.l1(h)
            return h, x

    return LeNetBackbone()

def run():
    args = parse_args()
    # if "" in args.target_folder:
    #     num_classes = zheng_num_classes[args.label_class] if args.num_classes is None else args.num_classes
    # else:
    #     num_classes = ptb_num_classes[args.label_class] if args.num_classes is None else args.num_classes
    _, train_loader, _ = get_dataset(
        args.batch_size, args.num_workers, [args.target_folder], folds=args.folds, 
        test=args.test, normalize=args.normalize, label_class=args.label_class, 
        filter_labels=args.filter_labels, input_size=args.input_size, 
        use_meta_information_in_head=args.use_meta_information_in_head)
    dataset, _, valid_loader = get_dataset(
        args.batch_size, args.num_workers, [args.target_folder], folds=args.folds, 
        test=False, normalize=args.normalize, label_class=args.label_class, 
        filter_labels=args.filter_labels, input_size=args.input_size,
        use_meta_information_in_head=args.use_meta_information_in_head)
    val_idmap = dataset.val_ds_idmap
    dataset, _, test_loader = get_dataset(
        args.batch_size, args.num_workers, [args.target_folder], test=True, 
        normalize=args.normalize, label_class=args.label_class, 
        filter_labels=args.filter_labels, input_size=args.input_size, 
        use_meta_information_in_head=args.use_meta_information_in_head)
    
    test_idmap = dataset.val_ds_idmap
    lbl_itos = dataset.lbl_itos
    num_classes = len(lbl_itos)
    tag = "f=" + str(args.folds) + "_" + args.tag
    tag = tag if args.use_pretrained else "ran_" + tag
    tag = "eval_" + tag if args.eval_only else tag
    tag = "clean_" + tag if "clean" in args.target_folder else tag
    model_tag = "finetuned" if args.load_finetuned else "ckpt"
    if args.test_noised:
        dataset, _, noise_valid_loader = get_dataset(
            args.batch_size, args.num_workers, [args.target_folder], apply_noise=True, noise_level=args.noise_level, test=args.test, label_class=args.label_class, filter_labels=args.filter_labels)

    else:
        noise_valid_loader = None
    losses, macros, predss, result_macros, result_macros_agg, test_macros, test_macros_agg, noised_macros, noised_macros_agg, scores_agg, noise_scores_agg = [
    ], [], [], [], [], [], [], [], [], [], []
    ckpt_epoch_lin = 0
    ckpt_epoch_fin = 0
    train_mode = 'lin' if args.f_epochs == 0 else "fin"
    
    save_model_at = os.path.join(os.path.dirname(
            args.model_file), "n=" + str(args.noise_level) + "_"+tag + train_mode +"_finetuned") if args.save_model_at is None else args.save_model_at
    
    filename = os.path.join(os.path.dirname(
        save_model_at), "t=" + args.trial_nr+"_n=" + str(args.noise_level) + "_"+tag + "res_" + train_mode +".pkl")

    model = load_model(
        args.linear_evaluation, num_classes, args.use_pretrained or args.load_finetuned, hidden=args.hidden,
        location=args.model_file, discriminative_lr=args.discriminative_lr, method=args.method, base_model=args.base_model,
        bn_head=args.bn_head, ps_head=args.ps_head, out_dim=args.num_classes, n_hidden=args.n_hidden, 
        conv_encoder=args.conv_encoder, use_meta_information_in_head=args.use_meta_information_in_head)
    used_optimizer = "pesg" if args.auc_maximization else args.optimizer
    loss_fn, optimizer = configure_optimizer(
        model, args.batch_size, head_only=True, optimizer=used_optimizer, auc_maximization=args.auc_maximization, num_classes=num_classes,
        discriminative_lr=args.discriminative_lr, discriminative_lr_factor=0.1 if args.use_pretrained and args.discriminative_lr else 1, base_model=args.base_model,
        use_meta_information_in_head=args.use_meta_information_in_head)
    if not args.eval_only:
        print("train model...")
        if not isdir(save_model_at):
            os.mkdir(save_model_at)

        l1, m1, bm, bm_agg, tm, tm_agg, ckpt_epoch_lin, preds = train_model(
            model, train_loader, valid_loader, test_loader, args.l_epochs, loss_fn, optimizer,
            head_only=True, linear_evaluation=args.linear_evaluation, lr_schedule=args.lr_schedule,
            save_model_at=join(save_model_at, "finetuned_"+args.trial_nr+".pt"),
            val_idmap=val_idmap, test_idmap=test_idmap, lbl_itos=lbl_itos, 
            cpc=(args.method == "cpc" or args.method == "s4"), base_model=args.base_model,
            use_meta_information_in_head=args.use_meta_information_in_head
            )
        
        if bm != 0:
            print("best macro after head-only training:", bm_agg)
        l2 = []
        m2 = []
        if args.f_epochs != 0:
            if args.l_epochs != 0:
                model = load_model(
                    False, num_classes, True, hidden=args.hidden,
                    location=join(save_model_at, "finetuned_" + args.trial_nr + ".pt"), discriminative_lr=args.discriminative_lr, method=args.method,
                    base_model=args.base_model, bn_head=args.bn_head, ps_head=args.ps_head, out_dim=args.num_classes, n_hidden=args.n_hidden, conv_encoder=args.conv_encoder,
                    use_meta_information_in_head=args.use_meta_information_in_head)
            loss_fn, optimizer = configure_optimizer(
                model, args.batch_size, head_only=False, num_classes=num_classes, optimizer=used_optimizer, 
                auc_maximization=args.auc_maximization, discriminative_lr=args.discriminative_lr,
                discriminative_lr_factor=0.1 if args.use_pretrained and args.discriminative_lr else 1, 
                base_model=args.base_model, use_meta_information_in_head=args.use_meta_information_in_head)
            l2, m2, bm, bm_agg, tm, tm_agg, ckpt_epoch_fin, preds = train_model(model, train_loader, valid_loader, test_loader, args.f_epochs, loss_fn,
                                                                                optimizer, head_only=False, linear_evaluation=False, lr_schedule=args.lr_schedule, save_model_at=join(save_model_at, "finetuned_"+args.trial_nr+".pt"),
                                                                                val_idmap=val_idmap, test_idmap=test_idmap, lbl_itos=lbl_itos, cpc=(args.method == "cpc"), base_model=args.base_model, use_meta_information_in_head=args.use_meta_information_in_head)
        losses.append(l1+l2)
        macros.append(m1+m2)
        test_macros.append(tm)
        test_macros_agg.append(tm_agg)
        result_macros.append(bm)
        result_macros_agg.append(bm_agg)

    else:
        preds, eval_macro, eval_macro_agg, scores_agg = evaluate(
            model, valid_loader, val_idmap, lbl_itos, ptb_to_zheng=args.ptb_to_zheng, base_model=args.base_model)
        result_macros.append(eval_macro)
        result_macros_agg.append(eval_macro_agg)

        preds, test_macro, test_macro_agg, scores_agg = evaluate(
            model, test_loader, test_idmap, lbl_itos, ptb_to_zheng=args.ptb_to_zheng, base_model=args.base_model)
        test_macros.append(test_macro)
        test_macros_agg.append(test_macro_agg)
        if args.verbose:
            print("macro:", eval_macro)
    predss.append(preds)

    if noise_valid_loader is not None:
        if args.test:
            val_idmap = test_idmap

        _, noise_macro, noise_macro_agg, noise_scores_agg = evaluate(
            model, noise_valid_loader, val_idmap, lbl_itos, base_model=args.base_model)
        noised_macros.append(noise_macro)
        noised_macros_agg.append(noise_macro_agg)
    res = {"filename": filename, "epochs": args.l_epochs+args.f_epochs, "model_location": args.model_location,
           "losses": losses, "macros": macros, "predss": predss, "result_macros": result_macros, "result_macros_agg": result_macros_agg,
           "test_macros": test_macros, "test_macros_agg": test_macros_agg, "noised_macros": noised_macros, "noised_macros_agg": noised_macros_agg, "scores_agg": scores_agg, "noise_scores_agg": noise_scores_agg, "ckpt_epoch_lin": ckpt_epoch_lin, "ckpt_epoch_fin": ckpt_epoch_fin,
           "discriminative_lr": args.discriminative_lr, "hidden": args.hidden, "lr_schedule": args.lr_schedule,
           "use_pretrained": args.use_pretrained, "linear_evaluation": args.linear_evaluation, "loaded_finetuned": args.load_finetuned,
           "eval_only": args.eval_only, "noise_level": args.noise_level, "test_noised": args.test_noised, "normalized": args.normalize, 
           'use_meta_information_in_head':args.use_meta_information_in_head, "batch_size":args.batch_size
           }
    pickle.dump(res, open(filename, "wb"))
    print("dumped results to", filename)
    print(res)
    print("Done!")
    return res


if __name__ == "__main__":
    run()
