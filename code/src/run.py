import argparse
import os, sys
from os.path import join

import numpy as np
import re
curr_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, join(curr_path,"../data/"))
sys.path.insert(0, join(curr_path,"../utils/"))
from util_functions import bcolors, find_best_ckpt

from datetime import datetime
import time

import torch
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar, LearningRateMonitor

from datasets import MnistDataModule, SynthDataModule, SynthPredictDataset
from network import FFNetwork
from ultrametric_callback import UltraMetricCallback, LitProgressBar
from custom_callbacks import Custom_EarlyStopping


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=1, help="Nb GPUs")
    parser.add_argument('--datafolder', type=str, default='../../data', help="Folder to save the data")
    parser.add_argument('--logfolder', type=str, default='../../logs', help="Folder to save the data")
    parser.add_argument('--verbose', type=int, default=0, help="define level of verbosity")
    parser.add_argument('--dataset', type=str, default="synth", help="define datset to use")

    parser.add_argument('--input_size', type=int, default=200, help="define level of verbosity")
    parser.add_argument('--batch_size_train', type=int, default=32, help="define level of verbosity")
    parser.add_argument('--batch_size_test', type=int, default=1000, help="define level of verbosity")
    parser.add_argument('--max_tree_depth', type=int, default=3, help="define level of verbosity")
    parser.add_argument('--noise_level', type=int, default=1, help="define level of verbosity")
    parser.add_argument('--p_flip', type=float, default=0.1, help="define level of verbosity")
    parser.add_argument('--p_noise', type=float, default=0.05, help="define level of verbosity")

    parser.add_argument('--hidden_size', type=int, default=200, help="define level of verbosity")
    parser.add_argument('--lr', type=float, default=0.01, help="define level of verbosity")
    parser.add_argument('--max_epochs', type=int, default=100, help="define level of verbosity")
    parser.add_argument('--generate_chain', action='store_true', help="define level of verbosity")
    parser.add_argument('--nb_folds', type=int, default=1, help="define level of verbosity")
    parser.add_argument('-T', '--temperature', type=float, default=0.4, help="define level of verbosity")
    parser.add_argument('--mode', type=str, default='rand', choices=['rand', 'um', 'split'], help="Folder to save the data")
    parser.add_argument('--num_workers', type=int, default=4, help="define level of verbosity")
    parser.add_argument('--b_len', type=int, default=0, help="if b_len > 0 -> do shuffles")
    parser.add_argument('--normalize_data', action='store_true', help="define level of verbosity")
    parser.add_argument('--test_split', type=float, default=0.2, help="define level of verbosity")
    parser.add_argument('--optimizer', type=str, default="sgd", choices=['sgd', 'adam'], help="define datset to use")
    parser.add_argument('--metric', type=str, default="val_loss", choices=['val_acc', 'val_loss', 'train_acc'], help="define datset to use")
    parser.add_argument('--auto_lr_find', action='store_true', help="define level of verbosity")
    parser.add_argument('--lr_scheduler', type=str, default=None, choices=['reduce_lr'], help="define datset to use")
    parser.add_argument('--eval_freq', type=int, default=250, help="define level of verbosity")
    parser.add_argument('--eval_freq_factor', type=float, default=2.0, help="define level of verbosity")
    parser.add_argument('--job_id', type=str, default="", help="define level of verbosity")
    parser.add_argument('--early_stop', action='store_true', help="define level of verbosity")
    parser.add_argument('--start_seed', type=int, default=0, help="define level of verbosity")
    parser.add_argument('--show_progbar', action='store_true', help="define level of verbosity")
    parser.add_argument('--no_reshuffle', action='store_true', help="true means eval only at one point")
    parser.add_argument('--save_ckpt', action='store_true', default=False, help="define level of verbosity")
    parser.add_argument('--last_val_step', type=int, default=0, help="define level of verbosity")
    parser.add_argument('--s_len', type=int, default=400, help="block lenght for split protocol")
    parser.add_argument('--val_step', type=int, default=100, help="Determines steps btw validations")
    parser.add_argument('--keep_correlations', action='store_true', help="true means eval only at one point")
    parser.add_argument('--stoch_s_len', action='store_true', help="true means eval only at one point")
    parser.add_argument('--ckpt_path', type=str, default='', help="Folder where ckpt to load is saved")

    args = parser.parse_args()
    print("All args: ", args)
    nb_classes = 2**args.max_tree_depth
    logs_path = def_logs_path(args)
    eval_freq = args.eval_freq
    if eval_freq <= args.b_len: 
        eval_freq = args.b_len
    elif args.b_len > 0: 
        eval_freq = (args.eval_freq // args.b_len) * args.b_len # ensure eval_freq is a multiple of b_len
    else: 
        eval_freq = args.eval_freq 
    assert(eval_freq > 0 and eval_freq >= args.b_len)
    print(f'Start eval_freq: {eval_freq}')

    if args.dataset == 'mnist':
        input_size = 28*28
    else:
        input_size = args.input_size

    val_step = args.val_step
    if eval_freq < args.val_step:
        val_step = eval_freq

    # training
    for seed in np.arange(args.start_seed, args.start_seed + args.nb_folds, step=1):
        start_time = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        
        print(f'{bcolors.OKCYAN}Running mode: {args.mode} seed: {seed} {bcolors.ENDC}')
        callbacks, checkpoint_callback = def_callbacks(args, logs_path, seed)

        model = FFNetwork(input_size=input_size, hidden_size=args.hidden_size, nb_classes=nb_classes, 
                        mode=args.mode, optimizer=args.optimizer, lr=args.lr, lr_scheduler=args.lr_scheduler,
                        b_len=args.b_len, eval_freq=eval_freq, eval_freq_factor=args.eval_freq_factor,
                        no_reshuffle=args.no_reshuffle, batch_size=args.batch_size_train, s_len=args.s_len, 
                        depth=args.max_tree_depth, keep_correlations=args.keep_correlations, 
                        stoch_s_len=args.stoch_s_len, last_val_step=args.last_val_step, val_step=val_step)
        
        data_module = create_data_modules(args)
        if args.mode == 'um':
            data_module.set_markov_chain(args, seed)
            val_check_interval=1
        elif args.mode == 'split':
            data_module.set_split_chain(args.stoch_s_len)
            val_check_interval=1
        elif args.mode == 'rand':
            val_check_interval=15

        
        logger = TensorBoardLogger(logs_path, name=f"metrics", version=f"fold_{seed}")
        trainer = pl.Trainer(default_root_dir=logs_path, gpus=args.gpu, 
                        num_nodes=1, precision=32, logger=logger, max_epochs=args.max_epochs,
                        callbacks=callbacks,
                        check_val_every_n_epoch=1, 
                        val_check_interval=val_check_interval,
                        enable_checkpointing=args.save_ckpt)
        
        if args.auto_lr_find:
            print("Fitting lr...")
            suggested_lr = find_lr(trainer=trainer, model=model, 
                                checkpoint_path_fold=logs_path, data_module=data_module)
            if suggested_lr <= 5.0 and suggested_lr > 1e3:
                model.hparams.lr = suggested_lr

        if args.ckpt_path:
            ckpt_path = args.ckpt_path
            if not ckpt_path.endswith('.ckpt'):
                ckpt_path = args.ckpt_path + '.ckpt'

            if not os.path.isfile(ckpt_path):
                ckpt_dir = join(logs_path, f'fold_{seed}')
                assert(os.path.isdir(ckpt_dir) and len(os.listdir(ckpt_dir)) > 0)
                ckpt_path = join(ckpt_dir, find_best_ckpt(ckpt_dir))

            print(f'@trainer.fit {ckpt_path}, {os.path.isfile(ckpt_path)}')
            trainer.fit(model, data_module, ckpt_path=ckpt_path)
        else:
            trainer.fit(model, data_module)

        if not checkpoint_callback == None:
            best_model_path = checkpoint_callback.best_model_path
            print(f"{bcolors.OKCYAN}best_model_path: {best_model_path} {bcolors.ENDC}")
            try:
                model = FFNetwork.load_from_checkpoint(checkpoint_callback.best_model_path)
            except IsADirectoryError:
                print('No best model saved: using last checkpoint as best one.')

        model.eval()
        trainer.test(model, dataloaders=data_module.val_dataloader()) #datamodule=data_module)

        end_time = time.time()
        print(f"{bcolors.OKGREEN}Fold {seed} computed in {(end_time-start_time)/60:.3}min {bcolors.ENDC}")

def create_data_modules(args):
        if args.dataset == 'mnist':
            data_module = MnistDataModule(data_dir=args.datafolder, mode=args.mode, 
                                    batch_size_train=args.batch_size_train, batch_size_test=args.batch_size_test,
                                    num_workers=args.num_workers, 
                                    normalization_transform=transforms.Normalize((0.1307,), (0.3081,)),  
                                    b_len=args.b_len, no_reshuffle=args.no_reshuffle, s_len=args.s_len, keep_correlations=args.keep_correlations)
        elif args.dataset == 'synth':
            data_module = SynthDataModule(data_dir=args.datafolder, mode=args.mode, 
                                        batch_size_train=args.batch_size_train, batch_size_test=args.batch_size_test, 
                                        num_workers=args.num_workers, max_depth=args.max_tree_depth, 
                                        noise_level=args.noise_level, p_flip=args.p_flip, p_noise=args.p_noise, 
                                        leaf_length=args.input_size, normalize_data=args.normalize_data, 
                                        test_split=args.test_split, b_len=args.b_len,
                                        no_reshuffle=args.no_reshuffle, s_len=args.s_len, keep_correlations=args.keep_correlations)
        return data_module

def find_lr(trainer, model, checkpoint_path_fold, data_module):
    lr_finder = trainer.tuner.lr_find(model, datamodule=data_module)
    print(f"Suggested_lr {lr_finder.suggestion()}")
    fig = lr_finder.plot(suggest=True)
    fig.savefig(join(checkpoint_path_fold, "lr_plot.jpg"))
    return lr_finder.suggestion()

def def_callbacks(args, checkpoint_path, seed):
    callbacks=[]
    checkpoint_callback=None
    checkpoint_folder_name = f"fold_{seed}/"
    checkpoint_path_fold = os.path.join(checkpoint_path, checkpoint_folder_name)

    
    if args.save_ckpt and (not args.no_reshuffle):
        optim_mode = 'max' if 'acc' in args.metric else 'min'
        print(f'Saving ckpt with {optim_mode} {args.metric}')
        checkpoint_callback = ModelCheckpoint(monitor=args.metric, dirpath=checkpoint_path_fold,
                                        filename="{step}_{val_acc:.4f}",
                                        save_top_k=1, mode=optim_mode)
        callbacks.append(checkpoint_callback)

    if args.early_stop:
        if args.mode in ['um', 'rand']:
            patience = 70 if args.b_len > 0 else 70
        elif args.mode == 'split':
            patience = 300
        stopping_threshold = 0.985 if args.b_len > 0 else 0.997
        callbacks.append(
            Custom_EarlyStopping(monitor="val_acc", min_delta=0.00, verbose=True, 
                        mode="max", stopping_threshold=stopping_threshold, patience=patience, strict=True))

    if not args.show_progbar:
        progressbar_callback = TQDMProgressBar(refresh_rate=0, process_position=0)
        callbacks.append(progressbar_callback)
    if args.lr_scheduler: # Activate lr monitor
        lr_callback = LearningRateMonitor(logging_interval='epoch', log_momentum=False)
        callbacks.append(lr_callback)
    
    return callbacks, checkpoint_callback

def def_logs_path(args, new_logs=False):
    if new_logs or len(args.ckpt_path) == 0:
        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")
        if args.dataset == 'mnist':
            tree_depth_str = ''
        elif args.dataset == 'synth':
            tree_depth_str = f'_d{args.max_tree_depth}'
        if args.mode == 'um':
            s_len_str = ''
        elif args.mode == 'split':
            s_len_str = f'_s{args.s_len}'
            if args.stoch_s_len:
                s_len_str = '_stoch' + s_len_str[1:]
        corr_string = ''
        if args.keep_correlations:
            corr_string = 'corr'
        ckpt_str = ''    

        ckpt_name = f"{args.job_id}_{dt_string}_{args.dataset}{corr_string}{ckpt_str}_{args.mode}_b{args.b_len}{tree_depth_str}{s_len_str}_h{args.hidden_size}_lr{args.lr}_rep{args.start_seed}/"
        logs_path = os.path.join(args.logfolder, ckpt_name)

    else:
        matched = re.search('(2[0-9]{6}_([0-9]{4}_){2}(?:synth|mnist)_(?:um|split|rand)_[a-z0-9._]+_rep[0-9]+)', args.ckpt_path)
        if matched:
            logs_name = matched.group(1)
            logs_path = os.path.join(args.logfolder, logs_name)
        else:
            print('Old logs not found, creating new log folder')
            def_logs_path(args, new_logs=True)
        print(logs_path)
        assert(os.path.isdir(logs_path))

    print(f"Saving logs at: {logs_path}")
    return logs_path

if __name__ == '__main__':
    run()
