import argparse
import os, sys
from os.path import join
from matplotlib import pyplot as plt

import numpy as np
curr_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, join(curr_path,"../data/"))
sys.path.insert(0, join(curr_path,"../utils/"))
from functions_markov import generate_markov_chain
from util_functions import bcolors

from datetime import datetime
import time

import torch
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from Datasets import MnistDataModule, SynthDataModule, SynthPredictDataset
from network import FFNetwork


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

    parser.add_argument('--hidden_size', type=int, default=1000, help="define level of verbosity")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="define level of verbosity")
    parser.add_argument('--max_epochs', type=int, default=100, help="define level of verbosity")
    parser.add_argument('--generate_chain', action='store_true', help="define level of verbosity")
    parser.add_argument('--nb_folds', type=int, default=1, help="define level of verbosity")
    parser.add_argument('-T', '--temperature', type=float, default=0.4, help="define level of verbosity")
    parser.add_argument('--mode', type=str, nargs='+', default='rand', help="Folder to save the data") #choices=['rand', 'um', 'split'],
    parser.add_argument('--num_workers', type=int, default=4, help="define level of verbosity")
    #parser.add_argument('-s', '--shuffles', nargs='+', type=int, help="define level of verbosity") # if want to do multiple shuffles
    parser.add_argument('--b_len', type=int, default=0, help="define level of verbosity")
    parser.add_argument('--normalize_data', action='store_true', help="define level of verbosity")
    parser.add_argument('--repeat_data', type=int, default=1, help="define level of verbosity")
    parser.add_argument('--test_split', type=float, default=0.2, help="define level of verbosity")
    parser.add_argument('--optimizer', type=str, default="sgd", choices=['sgd', 'adam'], help="define datset to use")

    
    args = parser.parse_args()
    print("All args: ", args)
    dataset_name = args.dataset
    nb_classes = 2**args.max_tree_depth
    patience = max(10, args.max_epochs / 50)
    nb_batches_per_epoch = int(nb_classes * args.noise_level / args.batch_size_train)
    early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=patience, verbose=False, mode="max")

    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    checkpoint_path = os.path.join(args.logfolder, f"ckpt_{dt_string}_{dataset_name}/")
    print(f"Saving checkpoints at: {checkpoint_path}")

    # training 
    for seed in range(args.nb_folds):
        data_modules = create_data_modules(args, dataset_name)
        print("Creating Network")
        model = FFNetwork(input_size=args.input_size, hidden_size=args.hidden_size, nb_classes=nb_classes, 
                        mode=args.mode, optimizer=args.optimizer, learning_rate=args.learning_rate)
        torch.manual_seed(seed)
        np.random.seed(seed)
        for mode, data_module in zip(args.mode, data_modules):
            print(f'Running mode: {mode} seed: {seed}')
            checkpoint_folder_name = f"{mode}_fold_{seed}/"
            checkpoint_path_fold = os.path.join(checkpoint_path, checkpoint_folder_name)
            checkpoint_callback = ModelCheckpoint(monitor="val_acc", dirpath=checkpoint_path_fold,
                                            filename="{epoch:02d}_{val_loss:.2f}",
                                            save_top_k=1, mode="max")
            logger = TensorBoardLogger(checkpoint_path, name=f"metrics_{dataset_name}", version=f"fold_{seed}")

            def set_markov_chain(args, mode, seed, data_module):
                if mode == 'um':
                    if args.generate_chain:
                        markov_chain = generate_markov_chain(chain_length=args.total_sample_nb, T=args.T, 
                                                            tree_levels=args.max_tree_depth, dia=0).tolist()
                    else:
                        saved_chain_name = f'saved_chains/tree_levels{args.max_tree_depth:02d}_clen1.0e+06_seed{seed:}.npy'
                        path_to_data = os.path.join(os.getcwd(), args.datafolder, saved_chain_name)
                        markov_chain = np.load(path_to_data).tolist()
            data_module.set_chain(args)

            trainer = pl.Trainer(default_root_dir=checkpoint_path, gpus=args.gpu, 
                                num_nodes=1, precision=32, logger=logger, max_epochs=args.max_epochs,
                                callbacks=[checkpoint_callback, early_stop_callback],
                                log_every_n_steps=nb_batches_per_epoch, check_val_every_n_epoch=max(args.max_epochs//20, 1)) #, fast_dev_run=4)
            
            model.hparams.lr = find_lr(trainer=trainer, model=model, checkpoint_path_fold=checkpoint_path_fold)
            print("hparams", model.hparams)

            trainer.fit(model, data_module)

            best_model_path = checkpoint_callback.best_model_path
            print(f"{bcolors.OKCYAN} best_model_path: {best_model_path} {bcolors.ENDC}")
            model = FFNetwork.load_from_checkpoint(checkpoint_callback.best_model_path)
            model.eval()
            trainer.test(model, dataloaders=data_module.val_dataloader())
            #print("Best Val Acc", checkpoint_callback.best_model_score.detach())

if __name__ == '__main__':
    run()

def create_data_modules(args, dataset_name: str):
        print("Creating datamodules")
        data_modules = []
        if dataset_name == 'mnist':
            if 'rand' in args.mode:
                data_modules.append(MnistDataModule(data_dir=args.datafolder, mode=args.mode, 
                                        batch_size_train=args.batch_size_train, batch_size_test=args.batch_size_test,
                                        num_workers=args.num_workers, 
                                        normalization_transform=transforms.Normalize((0.1307,), (0.3081,))))
            if 'um' in args.mode:
                data_modules.append(MnistDataModule(data_dir=args.datafolder, mode=args.mode, 
                                        batch_size_train=args.batch_size_train, batch_size_test=args.batch_size_test,
                                        num_workers=args.num_workers, 
                                        normalization_transform=transforms.Normalize((0.1307,), (0.3081,))))
        elif dataset_name == 'synth':
            if 'rand' in args.mode:
                data_modules.append(SynthDataModule(data_dir=args.datafolder, mode='rand', 
                                            batch_size_train=args.batch_size_train, batch_size_test=args.batch_size_test, 
                                            num_workers=args.num_workers, max_depth=args.max_tree_depth, 
                                            noise_level=args.noise_level, p_flip=args.p_flip, p_noise=args.p_noise, 
                                            leaf_length=args.input_size, normalize_data=args.normalize_data, 
                                            repeat_data=args.repeat_data, test_split=args.test_split, b_len=args.b_len))
            if 'um' in args.mode:
                data_modules.append(SynthDataModule(data_dir=args.datafolder, mode='um', 
                                            batch_size_train=args.batch_size_train, batch_size_test=args.batch_size_test, 
                                            num_workers=args.num_workers, max_depth=args.max_tree_depth, 
                                            noise_level=args.noise_level, p_flip=args.p_flip, p_noise=args.p_noise, 
                                            leaf_length=args.input_size, normalize_data=args.normalize_data, 
                                            repeat_data=args.repeat_data, test_split=args.test_split, b_len=args.b_len))
        return data_modules

def find_lr(trainer, model, checkpoint_path_fold):
                lr_finder = trainer.tuner.lr_find(model)
                print(f"Suggested_lr {lr_finder.results}")
                fig = lr_finder.plot(suggest=True)
                fig.savefig(join(checkpoint_path_fold, "lr_plot.jpg"))
                return lr_finder.suggestion()