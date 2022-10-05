
import sys
import numpy as np 
from os.path import join, dirname, basename, isdir
import pandas as pd
import pickle
import glob
import os
from tqdm import tqdm 
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pdb
from sklearn.utils import resample
import click
from os.path import dirname, basename
from sklearn.utils import resample
from clinical_ts.eval_utils_cafa import multiclass_roc_curve


def eval_auc(y_true, y_pred, classes=None, num_thresholds=100, full_output=False, parallel=True):
    '''returns label aucs
    '''
    results = {}

    fpr, tpr, roc_auc = multiclass_roc_curve(
        y_true, y_pred, classes=classes, precision_recall=False)

    results["label_AUC"] = roc_auc

    return results

def eval_auc_bootstrap(y_true, y_pred1, y_pred2, classes=None, n_iterations=1000, alpha=0.95):
    # https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf empirical bootstrap rather than bootstrap percentiles

    label_AUC_diff = []
    # point estimate
    res_point1 = eval_auc(y_true, y_pred1, classes=classes)
    res_point2 = eval_auc(y_true, y_pred2, classes=classes)
    res_point = {'label_AUC': {key:res_point1['label_AUC'][key]-res_point2['label_AUC'][key] \
                               for key in res_point1['label_AUC'].keys()
                              }
                }
    label_AUC_point = np.array(list(res_point["label_AUC"].values()))

    # bootstrap
    i = 0
    # for i in tqdm(range(n_iterations)):
    while i < n_iterations:
        print(i)
        ids = resample(range(len(y_true)), n_samples=len(y_true))
        if 0 in y_true[ids].sum(axis=0):
            continue
        res1 = eval_auc(y_true[ids], y_pred1[ids], classes=classes)
        res2 = eval_auc(y_true[ids], y_pred2[ids], classes=classes)
        res = {'label_AUC': {key:res1['label_AUC'][key]-res2['label_AUC'][key] \
                               for key in res1['label_AUC'].keys()
                              }
                }
        label_AUC_keys = list(res1["label_AUC"].keys())
        label_AUC_diff.append(
            np.array(list(res["label_AUC"].values()))-label_AUC_point)
        i += 1

    p = ((1.0-alpha)/2.0) * 100
    label_AUC_low = label_AUC_point + np.percentile(label_AUC_diff, p, axis=0)
    p = (alpha+((1.0-alpha)/2.0)) * 100
    label_AUC_high = label_AUC_point + np.percentile(label_AUC_diff, p, axis=0)

    return {"point1":res_point1['label_AUC'], "point2":res_point2['label_AUC'], "label_AUC": {k: [v1, v2, v3] for k, v1, v2, v3 in zip(label_AUC_keys, label_AUC_low, label_AUC_point, label_AUC_high)}}

def evaluate(model_path, train_fs, eval_fs, test=True, ret_scores=False, ret_preds=False, datamodule=None, model='s4', input_size_in_seconds_train=1, input_size_in_seconds_eval=1, d_state=8, batch_size=128):
    if datamodule is None:
        datamodule = ECGDataModule(
            batch_size,
            "./datasets/ptb_xl_fs"+str(eval_fs),
            label_class=args.label_class,
            num_workers=args.num_workers,
            data_input_size=int(100*input_size_in_seconds_eval*eval_fs/100)
        )

    def create_s4():
        return S4Model(d_input=12, d_output=datamodule.num_classes, l_max=int(100*input_size_in_seconds_train*train_fs/100), d_state=d_state)

    def create_res():
        return ECGResNet("xresnet1d50", datamodule.num_classes, big_input=(datamodule.data_input_size == 1250))

    if model == 's4':
        fun = create_s4
    else:
        fun = create_res
    pl_model = ECGLightningModel(
        fun,
        args.batch_size,
        datamodule.num_samples,
        lr=args.lr,
        rate=train_fs/eval_fs
    )
    load_from_checkpoint(pl_model, model_path)

    pl_model.save_preds = True

    if test:
        _ = trainer.test(pl_model, datamodule=datamodule)
        scores = pl_model.test_scores
        idmap = datamodule.test_idmap
    else:
        _ = trainer.validate(pl_model, datamodule=datamodule)
        scores = pl_model.val_scores
        idmap = datamodule.val_idmap
    preds = pl_model.preds
    targets = pl_model.targets
    if ret_preds:
        return aggregate_predictions(preds, targets, idmap), scores
    if ret_scores:
        return scores
    return scores['label_AUC']['macro']


@click.command()
@click.option('--preds_dir1', help='path to dataset')
@click.option('--preds_dir2', help='path to dataset')
@click.option('--n_iter', type=int, help='nr of iterations')
def bootstrap_comparison(preds_dir1, preds_dir2, n_iter):
    targets = np.load(join(dirname(preds_dir1), 'targets.npy'))
    lbl_itos = np.load(join(dirname(preds_dir1), 'lbl_itos.npy'))
    results = {}
    prediction_files1 = sorted([x for x in os.listdir(preds_dir1) if x.endswith(".npy")]) 
    prediction_files2 = sorted([x for x in os.listdir(preds_dir2) if x.endswith(".npy")])
    for i, pred_file1 in enumerate(prediction_files1):
        for j, pred_file2 in enumerate(prediction_files2):
            # pdb.set_trace()
            print(join(preds_dir1, pred_file1))
            print(join(preds_dir2, pred_file2))
            preds1 = np.load(join(preds_dir1, pred_file1))
            preds2 = np.load(join(preds_dir2, pred_file2))
            comparison = eval_auc_bootstrap(targets, preds1, preds2, classes=lbl_itos, n_iterations=n_iter)
            results[str(i)+"_"+str(j)] = comparison

            pickle.dump(results, open(
                join(preds_dir1, basename(preds_dir1)+"_vs_"+basename(preds_dir2)+'_results.pkl'), "wb"))







if __name__ == '__main__':
    bootstrap_comparison()
