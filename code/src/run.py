import argparse
import os, sys
from matplotlib import pyplot as plt

import numpy as np
curr_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(curr_path,"../data/"))
sys.path.insert(0, os.path.join(curr_path,"../utils/"))
from functions_markov import generate_markov_chain
from util_functions import bcolors

from datetime import datetime
import time

import torch
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from Datasets import MnistDataModule, SynthDataModule, SynthPredictDataset
from network import FFNetwork


def run():
    print("Parsing args")
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

    parser.add_argument('--hidden_size', type=int, default=1000, help="define level of verbosity")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="define level of verbosity")
    parser.add_argument('--max_epochs', type=int, default=100, help="define level of verbosity")
    parser.add_argument('--generate_chain', action='store_true', help="define level of verbosity")
    parser.add_argument('--nb_folds', type=int, default=1, help="define level of verbosity")
    parser.add_argument('-T', '--temperature', type=float, default=0.4, help="define level of verbosity")
    parser.add_argument('--mode', type=str, default='rand', choices=['rand', 'um', 'split'], help="Folder to save the data")
    parser.add_argument('--num_workers', type=int, default=4, help="define level of verbosity")
    parser.add_argument('--log_every_n_steps', type=int, default=10, help="define level of verbosity")
    parser.add_argument('-s', '--shuffles', nargs='+', type=int, help="define level of verbosity")
    
    
    args = parser.parse_args()
    dataset_name = args.dataset
    nb_classes = 2**args.max_tree_depth
    print("Creating DMs")
    if dataset_name == 'mnist':
        data_module = MnistDataModule(data_dir=args.datafolder, mode=args.mode, 
                                batch_size_train=args.batch_size_train, batch_size_test=args.batch_size_test,
                                num_workers=args.num_workers, normalization_transform=transforms.Normalize((0.1307,), (0.3081,)))
    elif dataset_name == 'synth':
        data_module = SynthDataModule(data_dir=args.datafolder, mode=args.mode, 
                                    batch_size_train=args.batch_size_train, batch_size_test=args.batch_size_test, 
                                    num_workers=args.num_workers, max_depth=args.max_tree_depth, noise_level=args.noise_level, 
                                    p_flip=args.p_flip, leaf_length=args.input_size)
    
    print("Creating Network")
    # model
    model = FFNetwork(args.input_size, args.hidden_size, args.learning_rate, nb_classes, mode=args.mode)

    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    checkpoint_path = os.path.join(args.logfolder, f"checkpoints_{dt_string}/")

    # training 
    for seed in range(args.nb_folds):
        torch.manual_seed(seed)
        checkpoint_folder_name = f"{dataset_name}_fold_{seed}/"
        checkpoint_path_fold = os.path.join(checkpoint_path, checkpoint_folder_name)
        checkpoint_callback = ModelCheckpoint(monitor="val_acc", dirpath=checkpoint_path_fold,
                                            filename="{epoch:02d}_{val_loss:.2f}",
                                            save_top_k=1, mode="max")
        logger = TensorBoardLogger(checkpoint_path, name=f"metrics_{dataset_name}", version=f"fold_{seed}")

        if args.mode == 'um':
            if args.generate_chain:
                markov_chain = generate_markov_chain(chain_length=args.total_sample_nb, T=args.T, 
                                                    tree_levels=args.max_tree_depth, dia=0).tolist()
            else:
                #put shuffle
                saved_chain_name = f'saved_chains/tree_levels{args.max_tree_depth:02d}_clen1.0e+06_seed{seed:}.npy'
                path_to_data = os.path.join(os.getcwd(), args.datafolder, saved_chain_name)
                markov_chain = np.load(path_to_data).tolist()
            data_module.set_chain(markov_chain)

        trainer = pl.Trainer(default_root_dir=checkpoint_path, gpus=args.gpu, 
                            num_nodes=1, precision=32, logger=logger, max_epochs=args.max_epochs,
                            reload_dataloaders_every_n_epochs=0, callbacks=[checkpoint_callback],
                            log_every_n_steps=args.log_every_n_steps) #, fast_dev_run=4)
        trainer.fit(model, data_module)

        best_model_path = checkpoint_callback.best_model_path
        print(f"{bcolors.OKCYAN} best_model_path: {best_model_path} {bcolors.ENDC}")
        model = FFNetwork.load_from_checkpoint(checkpoint_callback.best_model_path)
        model.eval()
        trainer.test(model, dataloaders=data_module.val_dataloader())
        #print("Best Val Acc", checkpoint_callback.best_model_score.detach())

if __name__ == '__main__':
    run()