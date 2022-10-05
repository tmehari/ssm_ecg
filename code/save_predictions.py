from train_ecg_model import *
import sys
import glob
import os
from tqdm import tqdm
from collections import defaultdict
import pdb
import click
from tqdm import tqdm

def load_from_checkpoint(pl_model, checkpoint_path):
    print("load model..")
    lightning_state_dict = torch.load(checkpoint_path)

    state_dict = lightning_state_dict["state_dict"] if 'state_dict' in lightning_state_dict.keys() else lightning_state_dict
    fails = []
    
    assert(len(state_dict.keys()) == len(pl_model.state_dict().keys()))
    for name, param in pl_model.named_parameters():
        if name in state_dict.keys():
            param.data = state_dict[name].data
        else:
            param.data = state_dict[name[6:]].data # remove 'model.' from key
    for name, param in pl_model.named_buffers():
        if name in state_dict.keys():
            param.data = state_dict[name].data
        else:
            param.data = state_dict[name[6:]].data # remove 'model.' from key


def evaluate(trainer, model_path, train_fs, eval_fs, label_class='label_all', data_dir='./data/ptb_xl_fs',batch_size=128, test=True, ret_scores=False,
             ret_preds=False, datamodule=None, model='s4', input_size_in_seconds_train=1,
             input_size_in_seconds_eval=1, d_state=8, normalize=False):
    if datamodule is None:
        datamodule = ECGDataModule(
            batch_size,
            data_dir+str(eval_fs),
            label_class=label_class,
            num_workers=8,
            data_input_size=int(100*input_size_in_seconds_eval*eval_fs/100),
            normalize=normalize
        )

    def create_cpc():
        model = CPCModel(input_channels=12, strides=[1]*4, kss=[1]*4, features=[
        512]*4, n_hidden=512, n_layers=2, num_classes=datamodule.num_classes, lin_ftrs_head=[512], bn_encoder=True, concat_pooling=True, bn_head=True, ps_head=0.5)
        return model

    def create_s4():
        return S4Model(d_input=12, d_output=datamodule.num_classes, l_max=int(100*input_size_in_seconds_train*train_fs/100), d_state=d_state, bidirectional=True)

    def create_s4_causal():
        return S4Model(d_input=12, d_output=datamodule.num_classes, l_max=int(100*input_size_in_seconds_train*train_fs/100), d_state=d_state, bidirectional=False)

    def create_res():
        return ECGResNet("xresnet1d50", datamodule.num_classes, big_input=(datamodule.data_input_size == 1250))

    def create_cpc_s4():
        num_encoder_layers = 4
        strides = [1]*num_encoder_layers
        kss = [1]*num_encoder_layers
        features = [512]*num_encoder_layers
        bn_encoder = True
        model = CPCModel(input_channels=12, num_classes=datamodule.num_classes, strides=strides, kss=kss, features=features, mlp=True,
                         bn_encoder=bn_encoder, ps_head=0.0, bn_head=False, lin_ftrs_head=None, s4=True, s4_d_model=512, s4_d_state=8,
                         s4_l_max=1024, concat_pooling=False, skip_encoder=False, s4_n_layers=4)
        return model

    if model == 's4':
        fun = create_s4
    elif model == 's4_causal':
        fun = create_s4_causal
    elif model == 'cpc_s4':
        fun = create_cpc_s4
    elif model == 'cpc':
        fun = create_cpc
    else:
        fun = create_res

    pl_model = ECGLightningModel(
        fun,
        batch_size,
        datamodule.num_samples,
        lr=0.001,
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
@click.option('--directory', help='path to models')
@click.option('--model', default='s4', help='model; res or s4')
@click.option('--save_dir', default='./preds', help='model')
@click.option('--fs', default=100, type=int)
@click.option('--batch_size', default=128, type=int)
@click.option('--data_dir', default='./data/ptb_xl_fs', help='dataset')
@click.option('--label_class', default='label_all', help='dataset')
@click.option('--load_finetuned', default=False, is_flag=True)
@click.option('--extra_tag')
def save_predictions(directory, model, save_dir, fs, batch_size, data_dir, label_class, load_finetuned, extra_tag):
    extra_tag = '' if extra_tag is None else "_"+extra_tag
    tag = '_finetuned' if load_finetuned else ''
    res_dir = join(save_dir, model + str(fs) + tag + extra_tag)
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)
    if load_finetuned:
        mod_files = list({file for file in glob.iglob(directory + '**/**',
                                                    recursive=True) if '.pt' in file})
    else:
        mod_files = list({file for file in glob.iglob(directory + '**/**',
                                                    recursive=True) if 'epoch' in file})

    trainer = Trainer(
        max_epochs=100,
        gpus=1,
        callbacks=[ModelCheckpoint(monitor='val/val_macro_agg', mode='max')],
        # resume_from_checkpoint=None if args.checkpoint_path == "" else args.checkpoint_path
    )
    for i, mod_file in tqdm(enumerate(mod_files)):
        (preds, targets), res100_aucs = evaluate(trainer, mod_file, fs, fs, label_class=label_class,data_dir=data_dir, test=True, ret_preds=True, ret_scores=False,
                                                    datamodule=None, model=model, input_size_in_seconds_train=2.5,
                                                    input_size_in_seconds_eval=2.5, d_state=8, batch_size=batch_size, normalize=load_finetuned)
        np.save(join(res_dir, str(i) + "_.npy"), preds)
        if not os.path.isfile(join(save_dir, 'targets.npy')):
            np.save(join(save_dir, 'targets.npy'), targets)
    if not os.path.isfile(join(save_dir, 'lbl_itos.npy')):
        np.save(join(save_dir, 'lbl_itos.npy'), trainer.datamodule.lbl_itos)

if __name__ == '__main__':
    save_predictions()
