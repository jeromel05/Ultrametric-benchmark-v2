import argparse
import os, sys
from matplotlib import pyplot as plt

import numpy as np

from repo.code.utils.functions_markov import generate_markov_chain
from repo.code.utils.util_functions import print_metrics
curr_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(curr_path,"../data/"))
sys.path.insert(0, os.path.join(curr_path,"../utils/"))

from datetime import datetime
import time

import torch
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from repo.code.utils.Datasets import MnistDataModule, SynthDataModule, SynthPredictDataset
from repo.code.utils.network import FFNetwork

def run(args):

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=1, help="Nb GPUs")
    parser.add_argument('--savefolder', type=str, default='./', help="Folder to save the data")
    parser.add_argument('--verbose', type=int, default=0, help="define level of verbosity")
    parser.add_argument('--dataset', type=str, default="synth", help="define datset to use")

    parser.add_argument('--input_size', type=int, default=200, help="define level of verbosity")
    parser.add_argument('--batch_size_train', type=int, default=64, help="define level of verbosity")
    parser.add_argument('--batch_size_test', type=int, default=1000, help="define level of verbosity")
    parser.add_argument('--max_tree_depth', type=int, default=3, required=True, help="define level of verbosity")
    parser.add_argument('--noise_level', type=int, default=1, help="define level of verbosity")
    parser.add_argument('--p_flip', type=float, default=0.1, help="define level of verbosity")

    parser.add_argument('--hidden_size', type=int, default=1000, help="define level of verbosity")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="define level of verbosity")
    parser.add_argument('--max_epochs', type=int, default=100, help="define level of verbosity")
    parser.add_argument('--generate_chain', action='store_true', help="define level of verbosity")
    parser.add_argument('--nb_folds', type=int, default=1, help="define level of verbosity")
    parser.add_argument('--total_sample_nb', type=int, default=1000000, help="define level of verbosity")
    parser.add_argument('-T', '--temperature', type=float, default=0.4, help="define level of verbosity")
    parser.add_argument('--mode', type=str, default='rand', choices=['rand', 'um', 'split'], help="Folder to save the data")
    parser.add_argument('--num_workers', type=int, default=4, help="define level of verbosity")
    
    args = parser.parse_args()

    nb_classes = 2**args.max_tree_depth
    
    if args.dataset == 'mnist':
        data_module = MnistDataModule(data_dir="../data/", mode=args.mode, 
                                batch_size_train=args.batch_size_train, batch_size_test=args.batch_size_test,
                                num_workers=args.num_workers, normalization_transform=transforms.Normalize((0.1307,), (0.3081,)))
    elif args.dataset == 'synth':
        data_module = SynthDataModule(data_dir="../data/", mode=args.mode, 
                                    batch_size_train=args.batch_size_train, batch_size_test=args.batch_size_test, 
                                    num_workers=args.num_workers, max_depth=args.max_tree_depth, noise_level=args.noise_level, 
                                    p_flip=args.p_flip, leaf_length=args.input_size)
        
    # model
    model = FFNetwork(args.input_size, args.hidden_size, args.learning_rate, nb_classes, mode=args.mode)
    logger = TensorBoardLogger("tb_logs", name="my_model")
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", dirpath="my/path/",
                                        filename="sample-mnist-{epoch:02d}-{val_loss:.2f}",
                                        save_top_k=3, mode="max")

    # training 
    for seed in range(args.nb_folds):
        torch.manual_seed(seed)
        if args.generate_chain and args.mode == 'um':
            markov_chain = generate_markov_chain(chain_length=args.total_sample_nb, T=args.T, 
                                                tree_levels=args.max_tree_depth, dia=0).tolist()
        else:
            markov_chain = np.load('../saved_chains/tree_levels{tree_levels:02d}_clen{chain_length:.1e}_seed{seed:}.npy').tolist()

        trainer = pl.Trainer(gpus=args.gpu, num_nodes=1, precision=32, logger=logger, max_epochs=args.max_epochs,
                            reload_dataloaders_every_n_epochs=0, callbacks=[checkpoint_callback]) #, fast_dev_run=4)
        trainer.fit(model, data_module)

        model = FFNetwork.load_from_checkpoint(checkpoint_callback.best_model_path)
        model.eval()

        X_test, target = data_module.get_test_data()
        predict_loader = DataLoader(SynthPredictDataset(X_test), batch_size=args.batch_size_test, 
                                        shuffle=False, num_workers=args.num_workers)
        preds = trainer.predict(model, predict_loader)
        acc, ap, auroc_, cf_mat, roc_curve = FFNetwork.evaluate_metrics(preds, target, nb_classes, cm_figure=True, roc_figure=True)
        print_metrics(acc, ap, auroc_, cf_mat, roc_curve, seed)
            

if __name__ == '__main__':
    run()