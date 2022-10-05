from platform import architecture
from pyexpat import model
import time
import logging
import os
import pdb
from os.path import exists, join, dirname
from argparse import ArgumentParser
import pickle

import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim import AdamW
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from clinical_ts.create_logger import create_logger
from clinical_ts.eval_utils_cafa import eval_scores, eval_scores_bootstrap
from clinical_ts.timeseries_utils import aggregate_predictions
from ecg_datamodule import ECGDataModule
from dl_models.ecg_resnet import ECGResNet
from dl_models.cpc import CPCModel
from dl_models.s4_model import S4Model
from mlxtend.evaluate import confusion_matrix


logger = create_logger(__name__)


class ECGLightningModel(pl.LightningModule):
    def __init__(
        self,
        model,
        batch_size,
        num_samples,
        lr=0.001,
        wd=0.001,
        loss_fn=F.binary_cross_entropy_with_logits,  # nn.CrossEntropyLoss()
        opt=AdamW,
        cal_acc=False,
        rate=1.0, 
        sigmoid_eval=True,
        save_preds=False,
        **kwargs
    ):
        """
        Args:
            create_model: function that returns torch model that is used for training
            batch_size: batch size
            lr: learning rate
            loss_fn : loss function
            opt: optimizer

        """

        super(ECGLightningModel, self).__init__()
        self.save_hyperparameters(logger=True)
        self.model = model()
        self.epoch = 0
        self.ce = isinstance(loss_fn, nn.CrossEntropyLoss)
        self.cal_acc = self.ce or cal_acc
        self.val_scores = None
        self.test_scores = None
        self.sigmoid_eval=sigmoid_eval
        self.save_preds = save_preds

    def configure_optimizers(self):
        optimizer = self.hparams.opt(self.model.parameters(
        ), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        # pdb.set_trace()
        return optimizer

    def forward(self, x, eval=False):
        if isinstance(self.model, S4Model):
            return self.model(x.float(), rate=self.hparams.rate)
        return self.model(x.float())

    def training_step(self, batch, batch_idx):
        x, targets = batch
        preds = self(x)
        if self.ce:
            preds = nn.Softmax(dim=1)(preds)
            loss = self.hparams.loss_fn(preds, targets)
        else:
            loss = self.hparams.loss_fn(preds, targets)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        epoch_loss = torch.tensor(
            [get_loss_from_output(output, key="loss") for output in outputs]
        ).mean()
        self.log("train/total_loss", epoch_loss, on_step=False, on_epoch=True)
        self.log("lr", self.hparams.lr, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):

        x, targets = batch
        preds = self(x)
        if self.ce:
            preds = nn.Softmax(dim=1)(preds)
            loss = self.hparams.loss_fn(preds, targets)
        else:
            loss = self.hparams.loss_fn(preds, targets)
        results = {
            "val_loss": loss,
            "preds": preds.cpu(),
            "targets": targets.cpu(),
        }
        return results

    def test_step(self, batch, batch_idx):

        x, targets = batch
        preds = self(x)
        if self.ce:
            preds == nn.Softmax(dim=1)(preds)
            loss = self.hparams.loss_fn(preds, targets)
        else:
            loss = self.hparams.loss_fn(preds, targets)
        results = {
            "test_loss": loss,
            "preds": preds.cpu(),
            "targets": targets.cpu(),
        }
        return results

    def validation_epoch_end(self, outputs):
        # outputs[0] when using multiple datasets
        preds = cat(outputs, "preds")
        targets = cat(outputs, "targets")
        macro, macro_agg, scores_agg = evaluate_macro(
            preds, targets, self.trainer, self.trainer.datamodule.idmap)

        if self.sigmoid_eval:
            preds = torch.sigmoid(Tensor(preds)).numpy()
        sigmacro, sigmacro_agg, sigscores_agg = evaluate_macro(
            preds, targets, self.trainer, self.trainer.datamodule.idmap)
        val_loss = mean(outputs, "val_loss")

        log = {
            "val/total_loss": val_loss,
            "val/val_macro": macro,
            "val/val_macro_agg": macro_agg,
            "val/val_macro_agg_sig": sigmacro_agg,
        }
        if self.cal_acc:
            hard_preds = np.zeros(preds.shape)
            hard_preds[torch.arange(len(hard_preds)), preds.argmax(axis=1)] = 1
            acc = (hard_preds.argmax(axis=1) ==
                   targets.argmax(axis=1)).sum()/len(targets)
            log["val/val_acc"] = acc

            preds_agg, targs_agg = aggregate_predictions(
                preds, targets, self.trainer.datamodule.idmap[: preds.shape[0]])
            preds1 = np.argmax(preds_agg, axis=1)
            targs1 = np.argmax(targs_agg, axis=1)
            cm = confusion_matrix(y_target=targs1,
                                  y_predicted=preds1,
                                  binary=False)
            self.cm = cm
        self.val_scores = scores_agg
        self.log_dict(log)
        return {"val_loss": val_loss, "log": log, "progress_bar": log}

    def test_epoch_end(self, outputs):
        # outputs[0] when using multiple datasets
        preds = cat(outputs, "preds")
        targets = cat(outputs, "targets")
        macro, macro_agg, scores_agg = evaluate_macro(
            preds, targets, self.trainer, self.trainer.datamodule.test_idmap)
        if self.sigmoid_eval:
            preds = torch.sigmoid(Tensor(preds)).numpy()
        if self.save_preds:
            self.preds = preds
            self.targets = targets
        sigmacro, sigmacro_agg, sigscores_agg = evaluate_macro(
            preds, targets, self.trainer, self.trainer.datamodule.test_idmap)
        test_loss = mean(outputs, "test_loss")
        log = {
            "test/total_loss": test_loss,
            "test/test_macro": macro,
            "test/test_macro_agg": macro_agg,
            "test/test_macro_agg_sig": sigmacro_agg,
        }
        if self.cal_acc:
            hard_preds = np.zeros(preds.shape)
            hard_preds[torch.arange(len(hard_preds)), preds.argmax(axis=1)] = 1
            acc = (hard_preds.argmax(axis=1) ==
                   targets.argmax(axis=1)).sum()/len(targets)
            log["test/test_acc"] = acc
            # preds1 = np.argmax(preds_agg, axis=1)
            # targs1 = np.argmax(targs_agg, axis=1)
            # cm = confusion_matrix(y_target=targs1,
            #                       y_predicted=preds1,
            #                       binary=False)
        self.log_dict(log)
        self.test_scores = sigscores_agg

        return {"test_loss": test_loss, "log": log, "progress_bar": log}

    def on_train_start(self):
        self.epoch = 0

    def on_train_epoch_end(self):
        self.epoch += 1


def get_loss_from_output(out, key="minimize"):
    return out[key] if isinstance(out, dict) else get_loss_from_output(out[0], key)


def mean(res, key1, key2=None):
    if key2 is not None:
        return torch.stack([x[key1][key2] for x in res]).mean()
    return torch.stack(
        [x[key1] for x in res if type(x) == dict and key1 in x.keys()]
    ).mean()


def cat(res, key):
    return torch.cat(
        [x[key] for x in res if type(x) == dict and key in x.keys()]
    ).numpy()


def evaluate_macro(preds, targets, trainer, idmap):
    # for val sanity check TODO find cleaner solution
    idmap = idmap[: preds.shape[0]]
    lbl_itos = trainer.datamodule.lbl_itos
    scores = eval_scores(targets, preds, classes=lbl_itos, parallel=True)
    preds_agg, targs_agg = aggregate_predictions(preds, targets, idmap)
    scores_agg = eval_scores(targs_agg, preds_agg,
                             classes=lbl_itos, parallel=True)
    macro = scores["label_AUC"]["macro"]
    macro_agg = scores_agg["label_AUC"]["macro"]
    return macro, macro_agg, scores_agg


def parse_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--dataset",
        dest="target_folder",
        help="used dataset for training",
    )
    parser.add_argument("--logdir", default="./logs")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--label_class", default="label_all")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--input_size", type=int, default=250)
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--nomemmap", action="store_true", default=False)
    parser.add_argument("--test_folds", nargs="+", default=[9, 10], type=int)
    parser.add_argument("--filter_label")
    parser.add_argument("--combination",  default="both")
    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument("--model", default='xresnet1d50')
    parser.add_argument("--rate", default=1.0, type=float)
    parser.add_argument("--d_state", default=8, type=int)
    parser.add_argument("--d_model", default=512, type=int)
    parser.add_argument("--n_layers", default=4, type=int)
    parser.add_argument("--s4_dropout", default=0.2, type=float)
    parser.add_argument("--bn", action='store_true', default=False)
    parser.add_argument("--binary_classification",
                        action='store_true', default=False)
    parser.add_argument("--concat_pooling",
                        action='store_true', default=False)
    parser.add_argument("--normalize", action='store_true', default=False)
    return parser


def init_logger(debug=False, log_dir="./experiment_logs"):
    level = logging.INFO

    if debug:
        level = logging.DEBUG

    # remove all handlers to change basic configuration
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        filename=os.path.join(log_dir, "info.log"),
        level=level,
        format="%(asctime)s %(name)s:%(lineno)s %(levelname)s:  %(message)s  ",
    )
    return logging.getLogger(__name__)


def get_experiment_name(args):
    experiment_name = str(time.asctime()) + "_" + \
        str(time.time_ns())[-3:]

    return experiment_name


def cli_main():
    parser = ArgumentParser()
    parser = parse_args(parser)

    args = parser.parse_args()
    print(args)
    experiment_name = get_experiment_name(args)

    init_logger(log_dir=join(args.logdir, experiment_name))

    # data
    datamodule = ECGDataModule(
        args.batch_size,
        args.target_folder,
        label_class=args.label_class,
        num_workers=args.num_workers,
        test_folds=args.test_folds,
        nomemmap=args.nomemmap,
        combination=args.combination,
        filter_label=args.filter_label,
        data_input_size=args.input_size,
        normalize=args.normalize
    )
    # pdb.set_trace()

    def create_linear():
        mod = nn.Sequential(
            nn.Conv1d(12, 64, 15, 10),
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
            # nn.ReLU(),
            nn.Linear(64, datamodule.num_classes)
        )
        return mod

    def create_model():
        return ECGResNet("xresnet1d50", datamodule.num_classes, big_input=False)

    def lenet():
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
        modules.append(nn.ReLU())
            # modules.append(nn.MaxPool1d(2))
        modules.append(nn.AdaptiveAvgPool1d(2))
        if ps:
            modules.append(nn.Dropout(0.1))
        modules.append(nn.Flatten())
        modules.append(nn.Linear(2*c4, E))  # 96
        modules.append(nn.ReLU())
        if hidden:
            if ps:
                modules.append(nn.Dropout(0.5))
            modules.append(nn.Linear(E, E))
            modules.append(nn.ReLU())
        if ps:
            modules.append(nn.Dropout(0.5))
        modules.append(nn.Linear(E, datamodule.num_classes))
        mod = nn.Sequential(*modules)
        return mod

    def create_s4():
        return S4Model(d_input=12, d_output=datamodule.num_classes, l_max=datamodule.data_input_size, d_state=args.d_state,
                       d_model=args.d_model, n_layers=args.n_layers, dropout=args.s4_dropout, bn=args.bn, bidirectional=True)

    def create_s4_causal():
        return S4Model(d_input=12, d_output=datamodule.num_classes, l_max=datamodule.data_input_size, 
        d_state=args.d_state, d_model=args.d_model, n_layers=args.n_layers, dropout=args.s4_dropout,
         bn=args.bn, bidirectional=False)

    # def create_cpc():
    #     model = CPCModel(input_channels=12, strides=[1]*3, kss=[1]*3, features=[
    #                      512]*3, n_hidden=128, n_layers=2, num_classes=datamodule.num_classes, lin_ftrs_head=[256], bn_encoder=True)
    #     return model

    def create_cpc():
        model = CPCModel(input_channels=12, strides=[1]*4, kss=[1]*4, features=[
        512]*4, n_hidden=512, n_layers=2, num_classes=datamodule.num_classes, lin_ftrs_head=[512], bn_encoder=True,
         ps_head=0.5, bn_head=True, concat_pooling=True)
        return model

    def create_cpc_s4():
        num_encoder_layers=4
        strides=[1]*num_encoder_layers 
        kss=[1]*num_encoder_layers
        features = [512]*num_encoder_layers
        bn_encoder=False
        model = CPCModel(input_channels=12, num_classes=datamodule.num_classes, strides=strides, kss=kss, features=features, mlp=True,
        bn_encoder=bn_encoder, ps_head=0.0, bn_head=False, lin_ftrs_head=None, s4=True, s4_d_model=512, s4_d_state=8,
        s4_l_max=1024, concat_pooling=args.concat_pooling, skip_encoder=False, s4_n_layers=args.n_layers)
        return model

    if args.model == 's4':
        fun = create_s4
    elif args.model == 's4_causal':
        fun = create_s4_causal
    elif args.model == 'xresnet1d50':
        fun = create_model
    elif args.model == 'lenet':
        fun = lenet
    elif args.model == 'linear':
        fun = create_linear
    elif args.model == 'cpc':
        fun = create_cpc
    elif args.model == 'cpc_s4':
        fun = create_cpc_s4
    else:
        raise Exception("model {} not found".format(args.model))

    # pytorch lightning module
    pl_model = ECGLightningModel(
        fun,
        args.batch_size,
        datamodule.num_samples,
        lr=args.lr,
        rate=args.rate,
        loss_fn=nn.CrossEntropyLoss(
        ) if args.binary_classification else F.binary_cross_entropy_with_logits
    )
    # configure trainer
    tb_logger = TensorBoardLogger(
        args.logdir, name=experiment_name, version="",) if not args.test_only else None
    # pdb.set_trace()
    trainer = Trainer(
        logger=tb_logger,
        max_epochs=args.epochs,
        gpus=1 if args.gpu else 0,
        callbacks=[ModelCheckpoint(monitor='val/val_macro_agg', mode='max')],
        # resume_from_checkpoint=None if args.checkpoint_path == "" else args.checkpoint_path
    )

    def load_from_checkpoint(pl_model, checkpoint_path):
        lightning_state_dict = torch.load(checkpoint_path)
        state_dict = lightning_state_dict["state_dict"]

        for name, param in pl_model.named_parameters():
            param.data = state_dict[name].data
        for name, param in pl_model.named_buffers():
            param.data = state_dict[name].data

    # load checkpoint
    if args.checkpoint_path != "":
        if exists(args.checkpoint_path):
            logger.info("Retrieve checkpoint from " + args.checkpoint_path)
            # pl_model.load_from_checkpoint(args.checkpoint_path)
            load_from_checkpoint(pl_model, args.checkpoint_path)
        else:
            raise ("checkpoint does not exist")

    # start training
    if not args.test_only:
        trainer.fit(pl_model, datamodule)
        trainer.save_checkpoint(os.path.join(
            args.logdir, experiment_name, "checkpoints", "model.ckpt"))

    _ = trainer.validate(pl_model, datamodule=datamodule)
    _ = trainer.test(pl_model, datamodule=datamodule)
    # pdb.set_trace()
    val_scores = pl_model.val_scores
    test_scores = pl_model.test_scores
    filename = "score.pkl"
    scores_file = join(dirname(args.checkpoint_path), filename) if args.checkpoint_path != "" else join(
        args.logdir, experiment_name, "checkpoints", filename)
    prefix = 'ptb' if 'ptb' in args.target_folder else "chap"
    scores = pickle.load(open(scores_file, "rb")
                         ) if exists(scores_file) else {}
    if not args.test_only:
        scores['description'] = "training=" + prefix
    pickle.dump(scores, open(scores_file, "wb"))
    scores[prefix] = {"val_scores": val_scores, "test_scores": test_scores}
    pickle.dump(scores, open(scores_file, "wb"))


if __name__ == "__main__":
    cli_main()
