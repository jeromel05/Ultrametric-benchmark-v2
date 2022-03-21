import argparse
import os, sys
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

from Datasets import MnistDataModule, SynthDataModule
from network import FFNetwork

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=int, default=1, help="Nb GPUs")
parser.add_argument('--savefolder', type=str, default='./', help="Folder to save the data")
parser.add_argument('--verbose', type=int, default=0, help="define level of verbosity")
parser.add_argument('--dataset', type=str, default="synth", help="define datset to use")

parser.add_argument('--input_size', type=int, default=200, help="define level of verbosity")
parser.add_argument('--batch_size_train', type=int, default=64, help="define level of verbosity")
parser.add_argument('--batch_size_test', type=int, default=1000, help="define level of verbosity")
parser.add_argument('--max_tree_depth', type=int, default=4, help="define level of verbosity")
parser.add_argument('--noise_level', type=int, default=1, help="define level of verbosity")
parser.add_argument('--p_flip', type=float, default=0.1, help="define level of verbosity")

parser.add_argument('--hidden_size', type=int, default=1000, help="define level of verbosity")
parser.add_argument('--learning_rate', type=float, default=0.001, help="define level of verbosity")
parser.add_argument('--max_epochs', type=int, default=100, help="define level of verbosity")


def run(args):
    nb_classes = 2**args.max_tree_depth
    total_sample_nb = 2000000 # 5000000
    random_seed = 1
    torch.manual_seed(random_seed)

    #for mode in ['um', 'rand', 'split']:
    mode='rand'
    if args.dataset == 'mnist':
        data_module = MnistDataModule("../data/", mode=mode, 
                                normalization_transform=transforms.Normalize((0.1307,), (0.3081,)))
    elif args.dataset == 'synth':
        data_module = SynthDataModule(nb_classes, "../data/", mode=mode, leaf_length=args.input_size, 
                                    batch_size_train=args.batch_size_train, batch_size_test=args.batch_size_test, 
                                    num_workers=4, max_depth=args.max_tree_depth, noise_level=args.noise_level, 
                                    p_flip=args.p_flip)
        
    # model
    model = FFNetwork(args.input_size, args.hidden_size, args.learning_rate, nb_classes, mode=mode)
    logger = TensorBoardLogger("tb_logs", name="my_model")
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", dirpath="my/path/",
                                        filename="sample-mnist-{epoch:02d}-{val_loss:.2f}",
                                        save_top_k=3, mode="max")

    # training 
    trainer = pl.Trainer(gpus=args.gpu, num_nodes=1, precision=32, logger=logger, max_epochs=args.max_epochs,
                        reload_dataloaders_every_n_epochs=0, callbacks=[checkpoint_callback]) #, fast_dev_run=4)
    trainer.fit(model, data_module)

    model = FFNetwork.load_from_checkpoint(checkpoint_callback.best_model_path)
    model.eval()
    y_hat = model(data_module.test_loader())

if __name__ == '__main__':
    args = parser.parse_args()
    run(args)