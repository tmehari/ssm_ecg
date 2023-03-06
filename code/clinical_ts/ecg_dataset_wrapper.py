from numpy.lib import index_tricks
from sklearn.utils.validation import _num_samples
from .create_logger import create_logger
import numpy as np
import torch
from torch.utils.data import DataLoader
# from .customDataLoader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset
# from torchvision import datasets
from functools import partial
from pathlib import Path
import pandas as pd
from pandas.core.series import Series
import pdb
# try:
#     import pickle5 as pickle
# except ImportError as e:
#     import pickle

import pickle
from .timeseries_utils import TimeseriesDatasetCrops, reformat_as_memmap, load_dataset, ToTensor
from .ecg_utils import *

logger = create_logger(__name__)

class Transformation:
    def __init__(self, *args, **kwargs):
        self.params = kwargs

    def get_params(self):
        return self.params

class TNormalize(Transformation):
    """Normalize using given stats.
    """
    def __init__(self, stats_mean=None, stats_std=None, input=True, channels=[]):
        super(TNormalize, self).__init__(
            stats_mean=stats_mean, stats_std=stats_std, input=input, channels=channels)
        self.stats_mean = torch.tensor([-0.00184586, -0.00130277,  0.00017031, -0.00091313, -0.00148835,  -0.00174687, -0.00077071, -0.00207407,  0.00054329,  0.00155546,  -0.00114379, -0.00035649])
        self.stats_std = torch.tensor([0.16401004, 0.1647168 , 0.23374124, 0.33767231, 0.33362807,  0.30583013, 0.2731171 , 0.27554379, 0.17128962, 0.14030828,   0.14606956, 0.14656108])
        self.stats_mean = self.stats_mean if stats_mean is None else stats_mean
        self.stats_std = self.stats_std if stats_std is None else stats_std
        self.input = input
        if(len(channels)>0):
            for i in range(len(stats_mean)):
                if(not(i in channels)):
                    self.stats_mean[:,i]=0
                    self.stats_std[:,i]=1

    def __call__(self, sample):
        if len(sample) == 3:
            return self._call3(sample)
        return self._call2(sample)
    
    def _call2(self, sample):
        datax, labelx = sample
        data = datax if self.input else labelx
        #assuming channel last
        data=data.T
        if(self.stats_mean is not None):
            data = data - self.stats_mean
        if(self.stats_std is not None):
            data = data/self.stats_std

        if(self.input):
            return (data.T, labelx)
        else:
            return (datax.T, data)
        
    def _call3(self, sample):
        datax, labelx, static = sample
        data = datax if self.input else labelx
        #assuming channel last
        data=data.T
        if(self.stats_mean is not None):
            data = data - self.stats_mean
        if(self.stats_std is not None):
            data = data/self.stats_std

        if(self.input):
            return (data.T, labelx, static)
        else:
            return (datax.T, data, static)

class ECGDataSetWrapper(object):

    def __init__(self, batch_size, num_workers, target_folder, input_size=250, label="label_diag_superclass", test=False,
                 shuffle_train=True, drop_last=True, nomemmap=False, test_folds=[8, 9], filter_label=None, combination='both', val_stride=None,
                  normalize=False, use_meta_information_in_head=False):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_folder = Path(target_folder)
        self.input_size = input_size
        self.val_ds_idmap = None
        self.lbl_itos = None
        self.train_ds_size = 0
        self.val_ds_size = 0
        self.label = label
        self.test = test
        self.shuffle_train = shuffle_train
        self.drop_last = drop_last
        # only for training without memmap file (e.g. syn vs real)
        self.no_memmap = nomemmap
        self.test_folds = np.array(test_folds)
        self.filter_label = filter_label
        self.combination = combination
        self.val_stride = val_stride
        self.normalize=normalize
        self.use_meta_information_in_head=use_meta_information_in_head

    def get_data_loaders(self):
        if self.normalize:
            print("use normalizaiton")
            data_augment = transforms.Compose([ToTensor(), TNormalize()])
        else:
            data_augment = ToTensor()

        if self.no_memmap:
            train_ds, val_ds = self._get_datasets_no_memmap(
                self.target_folder, transforms=data_augment)
        else:
            train_ds, val_ds = self._get_datasets(
                self.target_folder, transforms=data_augment)
        self.val_ds = val_ds
        self.val_ds_idmap = val_ds.get_id_mapping()

        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  num_workers=self.num_workers, pin_memory=True, shuffle=self.shuffle_train, drop_last=self.drop_last)
        valid_loader = DataLoader(val_ds, batch_size=self.batch_size,
                                  shuffle=False, num_workers=self.num_workers, pin_memory=True)

        self.train_ds_size = len(train_ds)
        self.val_ds_size = len(val_ds)
        return train_loader, valid_loader

    def get_training_params(self):
        chunkify_train = False
        chunkify_valid = True
        chunk_length_train = self.input_size  # target_fs*6
        chunk_length_valid = self.input_size
        min_chunk_length = self.input_size  # chunk_length
        stride_length_train = chunk_length_train//4  # chunk_length_train//8
        stride_length_valid = self.val_stride if self.val_stride is not None else self.input_size #//2  # chunk_length_valid

        copies_valid = 0  # >0 should only be used with chunkify_valid=False
        return chunkify_train, chunkify_valid, chunk_length_train, chunk_length_valid, min_chunk_length, stride_length_train, stride_length_valid, copies_valid

    def get_folds(self, target_folder):
        if self.test:
            valid_fold = 10
            test_fold = 9
        else:
            valid_fold = 9
            test_fold = 10
        if "thew" in str(target_folder) or "chapman" in str(target_folder) or "sph" in str(target_folder) or 'icbeb' in str(target_folder):
            valid_fold -= 1
            test_fold -= 1

        train_folds = []
        train_folds = list(range(1, 11))
        train_folds.remove(test_fold)
        train_folds.remove(valid_fold)
        train_folds = np.array(train_folds)
        train_folds = train_folds - \
            1 if "thew" in str(target_folder) or "zheng" in str(
                target_folder) else train_folds
        return train_folds, valid_fold, test_fold

    def get_dfs(self, df_mapped, target_folder):
        train_folds, valid_fold, test_fold = self.get_folds(target_folder)
        df_train = df_mapped[(df_mapped.strat_fold.apply(
            lambda x: x in train_folds))]
        df_valid = df_mapped[(df_mapped.strat_fold == valid_fold)]
        df_test = df_mapped[(df_mapped.strat_fold == test_fold)]
        return df_train, df_valid, df_test

    def _get_datasets(self, target_folder, transforms=None):
        logger.info("get dataset from " + str(target_folder))
        chunkify_train, chunkify_valid, chunk_length_train, chunk_length_valid, min_chunk_length, stride_length_train, stride_length_valid, copies_valid = self.get_training_params()

        ############### Load dataframe with memmap indices ##################

        df_mapped, lbl_itos, mean, std = load_dataset(target_folder)
        self.lbl_itos = lbl_itos

        ############### get right labels & map to multihot encoding ###################
        if "ptb" in str(target_folder):
            label = self.label  # just possible for ptb xl
            self.lbl_itos = np.array(lbl_itos[label])
            label = label + "_filtered_numeric"
            df_mapped["label"] = df_mapped[label].apply(
                lambda x: multihot_encode(x, len(self.lbl_itos)))
        elif 'sph' in str(target_folder):
            label = self.label  # just possible for ptb xl
            self.lbl_itos = np.array(lbl_itos[label])
            label = label + "_numeric"
            df_mapped["label"] = df_mapped[label].apply(
                lambda x: multihot_encode(x, len(self.lbl_itos)))
            
        elif "chapman" in str(target_folder):
            label = self.label
            # self.lbl_itos = np.array(lbl_itos[label.split("_")[-1]])
            self.lbl_itos = np.array(lbl_itos[label])
            df_mapped["label"] = df_mapped[label + "_numeric"].apply(
                lambda x: multihot_encode(x, len(self.lbl_itos)))
        elif "cinc" in str(target_folder):
            label = 'label'
            self.lbl_itos = lbl_itos
            df_mapped["label"] = df_mapped["label"].apply(
                lambda x: multihot_encode(x, len(lbl_itos)))
        elif 'icbeb' in str(target_folder):
            label = self.label
            self.lbl_itos = np.array(lbl_itos[label])
            df_mapped['label'] = df_mapped[label].apply(
                    lambda x: multihot_encode(x, len(self.lbl_itos)))
        else:
            label = "label"
            self.lbl_itos = lbl_itos
            df_mapped["label"] = df_mapped[label].apply(
                lambda x: np.array([1, 0, 0, 0, 0]))

        self.num_classes = len(self.lbl_itos)
        # import pdb; pdb.set_trace()
        df_mapped["diag_label"] = df_mapped[label].copy()

        df_train, df_valid, df_test = self.get_dfs(df_mapped, target_folder)

        cols_static = ['sex', 'age_nonan', 'height_nonan', 'weight_nonan', 'age_isnan', 'height_isnan', 
                     'weight_isnan'] if self.use_meta_information_in_head else None
                        
        ################## create datasets ########################
        train_ds = TimeseriesDatasetCrops(df_train, self.input_size, num_classes=self.num_classes, data_folder=target_folder, chunk_length=chunk_length_train if chunkify_train else 0, cols_static=cols_static,
                                          min_chunk_length=min_chunk_length, stride=stride_length_train, transforms=transforms, annotation=False, col_lbl="label", memmap_filename=target_folder/("memmap.npy"))
        val_ds = TimeseriesDatasetCrops(df_valid, self.input_size, num_classes=self.num_classes, data_folder=target_folder, chunk_length=chunk_length_valid if chunkify_valid else 0, cols_static=cols_static,
                                        min_chunk_length=min_chunk_length, stride=stride_length_valid, transforms=transforms, annotation=False, col_lbl="label", memmap_filename=target_folder/("memmap.npy"))

        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test

        return train_ds, val_ds

    def _get_datasets_no_memmap(self, target_folder, transforms=None):
        logger.info("get dataset from " + str(target_folder))
        chunkify_train, chunkify_valid, chunk_length_train, chunk_length_valid, min_chunk_length, stride_length_train, stride_length_valid, copies_valid = self.get_training_params()

        ############### Load dataframe with memmap indices ##################
        
        df = pickle.load(open(target_folder/("df.pkl"), 'rb'))
        # df = pd.read_pickle(target_folder/'df.pkl')
        lbl_itos = pickle.load(
            open(target_folder/("lbl_itos.pkl"), 'rb'))[self.label]

        self.lbl_itos = lbl_itos

        self.num_classes = len(self.lbl_itos)

        train_folds = []
        train_folds = list(range(10))

        for fold in self.test_folds:
            train_folds.remove(fold)
        train_folds = np.array(train_folds)

        df_train = df[(df.strat_fold.apply(lambda x: x in train_folds))]
        df_test = df[(df.strat_fold.apply(lambda x: x in self.test_folds))]
        patho_label_to_numeric = {"AVBlock":0, "LBBB": 1, "Normal":2, "RBBB":3}

        def filter_dataset(data_df, label=None, combination=''):
            if combination == 'real':
                data_df = data_df[data_df['label_real'].apply(
                    lambda x: x.argmax() == 0)]
            elif combination == 'syn':
                data_df = data_df[data_df['label_real'].apply(
                    lambda x: x.argmax() == 1)]

            if label is not None:
                data_df = data_df[data_df['label_patho'].apply(
                    lambda x: x.argmax() == patho_label_to_numeric[label])]

            return data_df

        
        df_train = filter_dataset(
            df_train, self.filter_label, combination=self.combination.split("_")[0])
        
        df_test = filter_dataset(
            df_test, self.filter_label, combination=self.combination.split("_")[1] if "_" in self.combination else '')
        
        self.df_train = df_train
        self.df_test = df_test
        
        ################## create datasets ########################
        train_ds = TimeseriesDatasetCrops(df_train, self.input_size, num_classes=self.num_classes, data_folder=target_folder, chunk_length=chunk_length_train if chunkify_train else 0,
                                          min_chunk_length=min_chunk_length, stride=stride_length_train, transforms=transforms, annotation=False, col_lbl=self.label)
        test_ds = TimeseriesDatasetCrops(df_test, self.input_size, num_classes=self.num_classes, data_folder=target_folder, chunk_length=chunk_length_valid if chunkify_valid else 0,
                                         min_chunk_length=min_chunk_length, stride=stride_length_valid, transforms=transforms, annotation=False, col_lbl=self.label)

        return train_ds, test_ds


def multihot_encode(x, num_classes):
    res = np.zeros(num_classes, dtype=np.float32)
    res[x] = 1
    return res


def filter_multihot_encode(x, filter_labels, numeric_mapping):
    res = res = np.zeros(len(filter_labels), dtype=np.float32)
    for label in x:
        if label in filter_labels:
            res[numeric_mapping[label]] = 1
    return res
