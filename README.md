# Advancing the State-of-the-Art for ECG Analysis through Structured State Space Models
This is a preliminary version of the offical code repository accompanying the ML4H submission on **Advancing the State-of-the-Art for ECG Analysis through Structured State Space Models**. The full functionality is included in the current repository but the interface/code structure is likely to improve until the end of the review period.

## Usage information
### Preparation
1. Install dependencies from `environment.yml` by running `conda env create -f environment.yml`, activate the environment via `conda activate ssm`

2. Follow the instructions in `data_preprocessing.ipynb` on how to download and preprocess the ECG datasets; in the following, we assume for definiteness that the preprocessed dataset folders can be found at `./data/sph_fs100`, `./data/cinc_fs100`,`./data/chapman_fs100`,`./data/ribeiro_fs100` and `./data/ptb_xl_fs100`.

### Code Structure 
The most functionality is provided in the files `code/pretraining.py`, `code/train_ecg_model.py` and `code/finetuning.py`. 

### A1. Pretraining on All
Pretrain a model (`4FC+4LSTM+FC` or the causal `S4` model, indicated by the `--s4` flag) using Contrastive Predictive Coding (CPC) on All2020 (which includes the cinc2020, chapman and ribeiro datasets) (will take about 6 days on a Tesla V100 GPU):
`python code/pretraining.py --data ./data/cinc_fs100 --data ./data/chapman_fs100 --data ./data/ribeiro_fs100 --normalize --epochs 200 --output-path=./runs/cpc/all --lr 0.0001 --batch-size 32 --input-size 1000 --fc-encoder --negatives-from-same-seq-only --mlp --s4`

The resulting model will be stored under the location indicated by the `--output-path` argument.

### A2. Finetuning on PTB-XL
Finetune the model by running:
`python eval.py  --dataset ./data/ptb_xl_fs100 --model_file "path/to/pretrained/model.ckpt" --batch_size 32 --use_pretrained --l_epochs 50 --f_epochs 50 --model s4`

The finetuned model will be stored in a directory next to the location of the pretrained model. Also, a pkl file will be stored, which contains the performance evaluations.

### B1. Supervised Training 
`code/train_ecg_model.py` was used to perform the supervised training experiments, run 

`python code/train_ecg_model.py --dataset ./data/ptb_xl_fs100 --label_class label_all --gpu --model s4
--logdir logs/s4_bidirectional`

to train a bidirectional S4 model on the PTB-XL dataset.

The results of the runs will be written to the log_dir in the following form:

experiment\_logs </br>
     	→ Wed Dec  2 17:06:54 2020\_simclr\_696\_RRC TO </br>
            → checkpoints </br>
                  → epoch=1654.ckpt </br>
                  → model.ckpt </br>
            → events.out.tfevents.1606925231 </br>
            → hparams.yaml </br>
      

2 directories, 5 files

While hparams.yaml and the event file store the hyperparameters and tensorboard logs of the run, respectively, we store the trained models in the checkpoints directory. We store two models, the best model according to the validation loss and the last model after training.

### B2. Evaluation of Supervised Training 
Revaluate a trained model by running

`python code/train_ecg_model.py --dataset ./data/ptb_xl_fs100 --label_class --label_all --model s4
--logdir logs/s4_bidirectional --test_only --checkpoint_path ./path/to/model.ckpt`

### C Comparison of two models via bootstrapping
For the bootstrap comparison the files `code/save_predictions.py` and `code/bootstrap_comparison.py` are used. 
First save the predictions on the test set of the models you like to compare 
For example first run 

`python code/save_predictions --directory ./path/to/s4_models --model s4 --data_dir ./data/ptb_xl_fs --fs 100 --label_class label_all --save_dir ./preds/ptb` 

and 

`python code/save_predictions --directory ./path/to/xresnet1d50_models --model xresnet1d50 --data_dir ./data/ptb_xl_fs --fs 100 --label_class label_all --save_dir ./preds/ptb` 

to save the predictions of all s4 models located in `./path/to/s4_models` in the `save_dir` `./preds/ptb/s4`, and the predictions of all xresnet1d50 models located in `./path/to/xresnet1d50_models` in `./preds/ptb/xresnet1d50`

Then, by running 

`python code/bootstrap_comparison.py --preds_dir1 preds/ptb/s4 --preds_dir2 preds/ptb/xresnet1d50`

you compare the predictions of the all the s4 models and xresnet1d50 models and save them in the `--preds_dir1` directory, which in this case, is `./preds/ptb/s4`, in form of a dictionary. 

## Pretrained models
Will be provided in the final version of the repository.

