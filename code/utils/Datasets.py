import os
from os.path import join
from sklearn import model_selection
import torch
import torchvision
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchmetrics.utilities.data import to_onehot
from sklearn.preprocessing import OneHotEncoder

from functions_markov import generate_markov_chain, shuffle_blocks_v2
from UltrametricTree import SynthUltrametricTree


class MnistPredictDataset(Dataset):
    """MnistLinearDataset dataset"""
    
    def __init__(self, data_df, transform=None):
        """
        Args:
            data_df (DataFrame): 
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = data_df["img"].values
        self.transform = transform
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_img = self.images[idx].reshape(28*28)
        
        if self.transform:
            sample_img = self.transform(sample_img)

        return sample_img
    

class SynthPredictDataset(Dataset):
    """MnistLinearDataset dataset"""
    
    def __init__(self, X, transform=None):
        """
        Args:
            data_df (DataFrame): 
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = X
        self.transform = transform
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample 

class UltraMetricSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, chain, class_index, nb_classes, batch_size_train, b_len=0, 
                with_replacement=False, epoch_size=1000):
        self.data_source = data_source
        self.chain = chain
        self.temp_shuff_chain = chain.copy()
        self.class_index = class_index
        self.nb_classes = nb_classes
        self.total_length = 0
        self.b_len = b_len
        self.batch_size_train = batch_size_train
        self.um_indexes = []
        self.with_replacement = with_replacement
        self.epoch_size = epoch_size

    def __iter__(self):
        idx = self.total_length
        
        if self.with_replacement:
            um_indexes = self.sample_with_replacement(idx)
        else:
            um_indexes = self.sample_without_replacement(idx)

        self.um_indexes = self.remove_excess_samples(um_indexes)
        self.total_length += len(self.um_indexes)
        return iter(self.um_indexes)

    def __len__(self):
        return len(self.data_source)

    def sample_with_replacement(self, idx):
        um_indexes = []
        idx0 = idx
        um_class = self.temp_shuff_chain[idx]
        idx += 1

        while idx < self.epoch_size + idx0:
            um_idx = np.random.choice(self.class_index[um_class], replace=True)
            um_indexes.append(um_idx)
            um_class = self.temp_shuff_chain[idx]
            idx += 1
            #print(f'{um_class}: {um_idx}',  end=' ')

        return um_indexes

    def sample_without_replacement(self, idx):
        um_indexes = []            
        nb_previous_occurences = np.zeros(self.nb_classes, dtype=np.int32)
        um_class = 0

        while (not self.with_replacement and nb_previous_occurences[um_class] < self.class_index[um_class].size) or (self.with_replacement and idx < self.epoch_size):
            um_idx = self.class_index[um_class][nb_previous_occurences[um_class]]
            um_indexes.append(um_idx)
            um_class = self.temp_shuff_chain[idx]
            nb_previous_occurences[um_class] += 1
            idx += 1

        return um_indexes

    def remove_excess_samples(self, um_indexes):
            to_remove = len(um_indexes) % self.batch_size_train
            if to_remove > 0: um_indexes = um_indexes[:-to_remove]
            return um_indexes

    def reset_sampler(self, until_idx=None):
        assert(self.b_len > 0)
        self.temp_shuff_chain = self.chain.copy()
        #print('SHUFFLING UNTIL: ', until_idx)
        if not until_idx == None:
            self.temp_shuff_chain[:until_idx] = shuffle_blocks_v2(self.chain[:until_idx], self.b_len)
        else:
            self.temp_shuff_chain[:self.total_length] = shuffle_blocks_v2(self.chain[:self.total_length], self.b_len)
        self.total_length = 0        


class UMDataModule(pl.LightningDataModule):
    def __init__(self, b_len: int, max_depth: int, data_dir: str = "./", batch_size_train: int=2, batch_size_test: int=1000, 
                 num_workers: int=4, mode: str='rand', chain=None, no_reshuffle=False, s_len=500, keep_correlations=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.num_workers = num_workers
        self.mode = mode
        self.train_sampler = None
        self.test_sampler = None
        self.nb_classes = 2**max_depth
        self.classes = np.arange(0, self.nb_classes)
        self.markov_chain = chain
        self.b_len=b_len
        self.no_reshuffle=no_reshuffle
        self.s_len=s_len
        self.keep_correlations=keep_correlations

    def train_dataloader(self):
        return DataLoader(self.um_train_ds, batch_size=self.batch_size_train, shuffle=False, 
                          num_workers=self.num_workers, sampler=self.train_sampler)

    def val_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size_test, shuffle=False, 
                          num_workers=self.num_workers, sampler=self.test_sampler)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size_test, shuffle=False, 
                          num_workers=self.num_workers, sampler=self.test_sampler)
    
    def set_markov_chain(self, args, seed):
        if args.generate_chain:
            self.markov_chain = generate_markov_chain(chain_length=args.total_sample_nb, T=args.T, 
                                                tree_levels=args.max_tree_depth, dia=0).tolist()
        else:
            saved_chain_name = f'saved_chains/tree_levels{args.max_tree_depth:02d}_clen1.0e+06_seed{seed:}.npy'
            path_to_data = join(os.getcwd(), args.datafolder, saved_chain_name)
            self.markov_chain = np.load(path_to_data).tolist()

        if self.no_reshuffle: # if we shuffle just once
            print('NO RESHUFFLE')
            self.markov_chain[:10000] = shuffle_blocks_v2(self.markov_chain[:10000], self.b_len)

    def set_split_chain(self, stoch_s_len=False):
        split_chain = []
        tot_nb_blocks=10000
        classes = np.random.choice(np.arange(0, self.nb_classes-1, step=2), size=tot_nb_blocks)
        if not stoch_s_len:
            split_chain = [el for class1 in classes for el in np.random.randint(low=class1, high=class1+2, size=self.s_len)]
        else:
            sizes = np.random.randint(low=max(int(self.s_len*0.25), 50), high=min(int(self.s_len*4), 2500), size=tot_nb_blocks)
            split_chain = [el for class1, size1 in zip(classes, sizes) for el in np.random.randint(low=class1, high=class1+2, size=size1)]
        self.markov_chain = split_chain

                             
class MnistDataModule(UMDataModule):
    def __init__(self, b_len: int, data_dir: str = "./", batch_size_train: int=128, batch_size_test: int=1000, 
                 num_workers: int=4, mode: str='rand', chain=None, normalization_transform: torchvision.transforms=None, 
                 no_reshuffle=False, s_len=500, keep_correlations=False):
        super().__init__(max_depth=3, data_dir=data_dir, batch_size_train=batch_size_train, batch_size_test=batch_size_test, 
                        num_workers=num_workers, mode=mode, chain=chain, b_len=b_len, s_len=s_len, keep_correlations=keep_correlations) ## implement shuffling of labels
        self.transform = transforms.Compose([transforms.ToTensor(), normalization_transform])
                             
    def setup(self, stage = None):
        train_ds=torchvision.datasets.MNIST(self.data_dir, train=True, download=False, transform=self.transform) # transform data as it is loaded
        test_ds=torchvision.datasets.MNIST(self.data_dir, train=False, download=False, transform=self.transform)
        train_img_tensor, train_label_list = [[el[column] for el in train_ds if el[1] in self.classes] for column in [0,1]]
        test_img_tensor, test_label_list = [[el[column] for el in test_ds if el[1] in self.classes] for column in [0,1]]

        train_img_tensor = [el.flatten() for el in train_img_tensor]
        test_img_tensor = [el.flatten() for el in test_img_tensor]

        train_img_tensor = torch.stack(train_img_tensor, dim=0)
        test_img_tensor = torch.stack(test_img_tensor, dim=0)
        train_y = torch.tensor(train_label_list, dtype=torch.float)
        test_y = torch.tensor(test_label_list, dtype=torch.float)

        def prepare_target_data(y):
            class_index = [np.where(y==class_label)[0] for class_label in self.classes] 
            y = to_onehot(y, self.nb_classes)
            return y, class_index
        
        train_y, train_class_index = prepare_target_data(train_y)
        test_y, _ = prepare_target_data(test_y)
        self.um_train_ds = TensorDataset(train_img_tensor, train_y)
        self.test_ds = TensorDataset(test_img_tensor, test_y)
        #self.predict_ds = MnistPredictDataset(filtered_test_df, transform=None)

        if self.mode in ['um', 'split']:
            if self.mode == 'um':
                self.train_sampler = UltraMetricSampler(data_source=self.um_train_ds, chain=self.markov_chain, class_index=train_class_index, 
                                                        nb_classes=self.nb_classes, batch_size_train=self.batch_size_train, b_len=self.b_len, with_replacement=False)
            if self.mode == 'split':
                self.train_sampler = UltraMetricSampler(data_source=self.um_train_ds, chain=self.markov_chain, class_index=train_class_index, 
                                                        nb_classes=self.nb_classes, batch_size_train=self.batch_size_train, b_len=self.b_len, with_replacement=True)


                                
class SynthDataModule(UMDataModule):
    def __init__(self, max_depth: int, data_dir: str = "./", batch_size_train: int=8, batch_size_test: int=1000, 
                 num_workers: int=4, mode: str='rand', chain=None, leaf_length=200, noise_level=1, p_flip=0.1,
                 p_noise=0.02, normalize_data=False, test_split=0.1, b_len=0, no_reshuffle=False, s_len=500,
                 keep_correlations=False):
        super().__init__(max_depth=max_depth, data_dir=data_dir, batch_size_train=batch_size_train, 
                        batch_size_test=batch_size_test, num_workers=num_workers, mode=mode, chain=chain,
                        b_len=b_len, no_reshuffle=no_reshuffle, s_len=s_len, keep_correlations=keep_correlations)
        self.leaf_length = leaf_length
        self.tree = SynthUltrametricTree(max_depth=max_depth, p_flip=p_flip, p_noise=p_noise, 
                                         leaf_length=leaf_length, shuffle_labels=(not keep_correlations),
                                         noise_level=noise_level)
        self.normalize_data = normalize_data
        self.test_split = test_split
                             
    def setup(self, stage = None):
        X, y = self.tree.leaves, self.tree.labels
        if self.normalize_data:
            X = np.array([el/np.sum(el) for el in X]) # normalize input

        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=self.test_split)
        print(f"Train data size {list(X_train.size())}, {list(y_train.size())}, classes: {list(torch.unique(y_train).size())[0]}")
        print(f"Val data size {list(X_test.size())}, {list(y_test.size())}, classes: {list(torch.unique(y_test).size())[0]}")

        def prepare_target_data(y):
            class_index = [np.where(y==class_label)[0] for class_label in self.classes] 
            y = to_onehot(y, self.nb_classes)
            return y, class_index

        y_train, train_class_index = prepare_target_data(y_train)
        y_test, _ = prepare_target_data(y_test)
    
        self.um_train_ds=TensorDataset(X_train, y_train)
        self.test_ds=TensorDataset(X_test, y_test)

        if self.mode in ['um', 'split']:
            assert(len(set(self.markov_chain)) == self.tree.nb_classes) #assert all classes are represented in the Markov chain
      
            if self.mode == 'um':
                    self.train_sampler = UltraMetricSampler(data_source=self.um_train_ds, chain=self.markov_chain, class_index=train_class_index, 
                                                            nb_classes=self.nb_classes, batch_size_train=self.batch_size_train, b_len=self.b_len, with_replacement=False)
            if self.mode == 'split':
                self.train_sampler = UltraMetricSampler(data_source=self.um_train_ds, chain=self.markov_chain, class_index=train_class_index, 
                                                        nb_classes=self.nb_classes, batch_size_train=self.batch_size_train, b_len=self.b_len, with_replacement=True)
        self.predict_ds = SynthPredictDataset(X_test)

    