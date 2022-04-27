import argparse
import os, sys
from os.path import join

import numpy as np
curr_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, join(curr_path,"../data/"))
sys.path.insert(0, join(curr_path,"../utils/"))
from util_functions import bcolors

from datetime import datetime
import time

import torch
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar, LearningRateMonitor

from Datasets import MnistDataModule, SynthDataModule, SynthPredictDataset
from network import FFNetwork
from ultrametric_callback import UltraMetricCallback, LitProgressBar


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
    parser.add_argument('--max_tree_depth', type=int, default=3, required=True, help="define level of verbosity")
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
    parser.add_argument('--repeat_data', type=int, default=1, help="define level of verbosity")
    parser.add_argument('--test_split', type=float, default=0.2, help="define level of verbosity")
    parser.add_argument('--optimizer', type=str, default="sgd", choices=['sgd', 'adam'], help="define datset to use")
    parser.add_argument('--metric', type=str, default="val_loss", choices=['val_acc', 'val_loss', 'train_acc'], help="define datset to use")
    parser.add_argument('--auto_lr_find', action='store_true', help="define level of verbosity")
    parser.add_argument('--lr_scheduler', type=str, default=None, choices=['reduce_lr'], help="define datset to use")
    parser.add_argument('--eval_freq', type=int, default=1, help="define level of verbosity")
    parser.add_argument('--job_id', type=str, default="", help="define level of verbosity")
    parser.add_argument('--early_stop', action='store_true', help="define level of verbosity")
    parser.add_argument('--start_seed', type=int, default=0, help="define level of verbosity")
    parser.add_argument('--show_progbar', action='store_true', help="define level of verbosity")
    parser.add_argument('--single_eval', type=int, default=0, help="define level of verbosity")

    args = parser.parse_args()
    print("All args: ", args)
    nb_classes = 2**args.max_tree_depth
    logs_path = def_logs_path(args)
    max_batches_per_epoch = int(int((1-args.test_split) * nb_classes * args.noise_level) / args.batch_size_train)

    # training
    for seed in np.arange(args.start_seed, args.start_seed + args.nb_folds, step=1):
        start_time = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)

        eval_steps=None
        if args.b_len > 0:
            eval_steps = def_eval_steps(args)
            if args.single_eval > 0:
                eval_steps = [args.single_eval]
                print(f'Only a single eval will be done at: {eval_steps}')

        
        data_module = create_data_modules(args, args.dataset)
        print(f'{bcolors.OKCYAN}Running mode: {args.mode} seed: {seed} {bcolors.ENDC}')
        callbacks, checkpoint_callback = def_callbacks(args, logs_path, seed)

        model = FFNetwork(input_size=args.input_size, hidden_size=args.hidden_size, nb_classes=nb_classes, 
                        mode=args.mode, optimizer=args.optimizer, lr=args.lr, lr_scheduler=args.lr_scheduler,
                        eval_steps=eval_steps, max_batches_per_epoch=max_batches_per_epoch, b_len=args.b_len,
                        eval_freq=args.eval_freq)
        
        logger = TensorBoardLogger(logs_path, name=f"metrics", version=f"fold_{seed}")
        if args.mode == 'um':
            data_module.set_markov_chain(args, seed)
            val_check_interval=1
        else:
            val_check_interval=10
            
        trainer = pl.Trainer(default_root_dir=logs_path, gpus=args.gpu, 
                            num_nodes=1, precision=32, logger=logger, max_epochs=args.max_epochs,
                            callbacks=callbacks,
                            log_every_n_steps=1, 
                            check_val_every_n_epoch=args.eval_freq, 
                            val_check_interval=val_check_interval) #checks val after each train batch -> expensive
                            #, fast_dev_run=4)
        
        if args.auto_lr_find:
            print("Fitting lr...")
            suggested_lr = find_lr(trainer=trainer, model=model, 
                                checkpoint_path_fold=logs_path, data_module=data_module)
            if suggested_lr <= 5.0 and suggested_lr > 1e3:
                model.hparams.lr = suggested_lr

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

def create_data_modules(args, dataset_name: str):
        if dataset_name == 'mnist':
            data_module = MnistDataModule(data_dir=args.datafolder, mode=args.mode, 
                                    batch_size_train=args.batch_size_train, batch_size_test=args.batch_size_test,
                                    num_workers=args.num_workers, 
                                    normalization_transform=transforms.Normalize((0.1307,), (0.3081,)))
        elif dataset_name == 'synth':
            data_module = SynthDataModule(data_dir=args.datafolder, mode=args.mode, 
                                        batch_size_train=args.batch_size_train, batch_size_test=args.batch_size_test, 
                                        num_workers=args.num_workers, max_depth=args.max_tree_depth, 
                                        noise_level=args.noise_level, p_flip=args.p_flip, p_noise=args.p_noise, 
                                        leaf_length=args.input_size, normalize_data=args.normalize_data, 
                                        repeat_data=args.repeat_data, test_split=args.test_split, b_len=args.b_len)
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

    optim_mode = 'max' if 'acc' in args.metric else 'min'
    print(f'Optimizing on {args.metric} mode {optim_mode}')
    if args.single_eval == 0:
        checkpoint_callback = ModelCheckpoint(monitor=args.metric, dirpath=checkpoint_path_fold,
                                        filename="{epoch:02d}_{val_loss:.2f}",
                                        save_top_k=1, mode=optim_mode)
        callbacks.append(checkpoint_callback)

    if args.early_stop:
        callbacks.append(
            EarlyStopping(monitor="val_acc", min_delta=0.00, verbose=True, 
                        mode="max", stopping_threshold=0.95, patience=1500, strict=True))

    if not args.show_progbar:
        progressbar_callback = TQDMProgressBar(refresh_rate=0, process_position=0)
        callbacks.append(progressbar_callback)
    if args.lr_scheduler: # Activate lr monitor
        lr_callback = LearningRateMonitor(logging_interval='epoch', log_momentum=False)
        callbacks.append(lr_callback)
    
    return callbacks, checkpoint_callback

def def_logs_path(args):
    now = datetime.now()
    dt_string = now.strftime("%d%m_%H%M")
    ckpt_name = f"{args.job_id}_{dt_string}_{args.dataset}_{args.mode}_b{args.b_len}_d{args.max_tree_depth}_h{args.hidden_size}_lr{args.lr}/"
    logs_path = os.path.join(args.logfolder, ckpt_name)
    print(f"Saving logs at: {logs_path}")
    return logs_path

def def_eval_steps(args):
    eval_steps = np.arange(0, int(args.max_epochs / args.eval_freq), 1) * args.eval_freq
    for i in range(1, len(eval_steps)):
        eval_steps[i] = eval_steps[i-1] + eval_steps[i]
    eval_steps=eval_steps-1

    eval_steps = [el for el in eval_steps if el < args.max_epochs]
    eval_steps = eval_steps[1:]
    print(f"Evaluation at steps: {eval_steps[0:15]}..., net will be evaluated {len(eval_steps)} times")
    return eval_steps

if __name__ == '__main__':
    run()
