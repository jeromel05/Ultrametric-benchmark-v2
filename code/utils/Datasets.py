from sklearn import model_selection
import torch
import torchvision
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, TensorDataset
from util_functions import one_hot_labels
from functions_markov import generate_markov_chain
from UltrametricTree import SynthUltrametricTree



class MnistLinearDataset(Dataset):
    """MnistLinearDataset dataset"""
    
    def __init__(self, data_df, transform=None):
        """
        Args:
            data_df (DataFrame): 
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = data_df["img"].values
        self.labels = data_df["label"].values
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_img = self.images[idx][0].reshape(28*28)
        sample_label = torch.Tensor(self.labels[idx][0])
        
        if self.transform:
            sample_img = self.transform(sample_img)

        return [sample_img, sample_label]
    
class UltrametricMnistDataset(Dataset):
    """Ultrametric dataset"""
    
    def __init__(self, data_df, transform=None):
        """
        Args:
            data_df (DataFrame): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = data_df["img"].values
        self.labels = data_df["label"].values
        self.transform = transform
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample_img = self.images[idx][0].reshape(28*28)
        sample_label = torch.Tensor(self.labels[idx][0])
        
        if self.transform:
            sample_img = self.transform(sample_img)
        
        return [sample_img, sample_label]

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

        sample_img = self.images[idx][0].reshape(28*28)
        
        if self.transform:
            sample_img = self.transform(sample_img)

        return sample_img
    
class UltraMetricSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, chain, class_index, nb_classes, do_resampling=False, predef_length=0):
        self.data_source = data_source
        self.chain = chain
        self.class_index = class_index
        self.nb_classes = nb_classes
        self.total_length = 0
        self.do_resampling = do_resampling
        self.predef_length = predef_length
        if not self.do_resampling:
            self.__iter__(dummy=True)
        else:
            self.length = self.predef_length

    def __iter__(self, dummy=False):
        um_indexes = []
        if self.do_resampling:
            for el in self.chain[self.total_length:self.total_length+self.predef_length]:
                um_indexes.append(np.random.choice(self.class_index[el]))
            
        else:       
            idx = self.total_length
            nb_previous_occurences = np.zeros(self.nb_classes, dtype=np.int32)
            um_class = 0

            while nb_previous_occurences[um_class] < self.class_index[um_class].size:
                um_idx = self.class_index[um_class][nb_previous_occurences[um_class]]
                um_indexes.append(um_idx)
                um_class = self.chain[idx]
                nb_previous_occurences[um_class] = nb_previous_occurences[um_class] + 1
                idx=idx+1

            self.length = len(um_indexes)
            
        if not dummy:
            self.total_length = self.total_length + len(um_indexes)
        return iter(um_indexes)

    def __len__(self):
        return self.length
    
    def shuffle_um_chain(self):
        np.shuffle(self.chain)
    
class BinarySampler(torch.utils.data.Sampler):
    def __init__(self, data_source, class_index, chain, train=True):
        self.data_source = data_source
        self.class_index = class_index
        self.chain = chain
        self.train = train
        self.curr_epoch_nb = 0

    def __iter__(self):
        classes_to_sample = self.chain[self.curr_epoch_nb:self.curr_epoch_nb+2]
        indexes = np.concatenate((self.class_index[classes_to_sample[0]], self.class_index[classes_to_sample[1]]))
        if self.train:
            np.random.shuffle(indexes)
        print("classes: ", classes_to_sample, self.train, self.curr_epoch_nb)
        
        return iter(indexes)

    def __len__(self):
        classes_to_sample = self.chain[self.curr_epoch_nb:self.curr_epoch_nb+2]
        indexes = np.concatenate((self.class_index[classes_to_sample[0]], self.class_index[classes_to_sample[1]]))
        return len(indexes)
    
    def update_curr_epoch_nb(self):
        self.curr_epoch_nb = self.curr_epoch_nb + 2
        
    def get_curr_epoch_nb(self):
        return self.curr_epoch_nb

class MnistDataModule(pl.LightningDataModule):
    def __init__(self, nb_classes: int, data_dir: str = "./", chain=None, batch_size_train: int=128, 
                batch_size_test=1000, num_workers=4, mode='rand', normalization_transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), normalization_transform])
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.num_workers = num_workers
        self.chain = chain.tolist()
        self.mode = mode
        self.train_sampler = None
        self.test_sampler = None
        self.nb_classes = nb_classes
        self.classes = np.arange(0, self.nb_classes)
                             
    def setup(self, stage = None):
        train_ds=torchvision.datasets.MNIST(self.data_dir, train=True, download=False, transform=self.transform)
        test_ds=torchvision.datasets.MNIST(self.data_dir, train=False, download=False, transform=self.transform)
        
        filtered_train_ds=[el for el in train_ds if el[1] in self.classes]
        filtered_train_df = pd.DataFrame(filtered_train_ds, columns=[["img", "label"]])
        train_class_index = [filtered_train_df.loc[filtered_train_df["label"].values == class_label].index for class_label in self.classes]
        filtered_train_df["label"] = filtered_train_df["label"].apply(one_hot_labels, 
                                                                    args=(self.nb_classes,), axis=1)
        
        filtered_test_ds=[el for el in test_ds if el[1] in self.classes]
        filtered_test_df = pd.DataFrame(filtered_test_ds, columns=[["img", "label"]])
        test_class_index = [filtered_test_df.loc[filtered_test_df["label"].values == class_label].index for class_label in self.classes]
        filtered_test_df["label"] = filtered_test_df["label"].apply(one_hot_labels, 
                                                                    args=(self.nb_classes,), axis=1)
        
        if self.mode == 'rand':
            self.um_train_ds=MnistLinearDataset(filtered_train_df, transform=None)            
            self.test_ds=MnistLinearDataset(filtered_test_df, transform=None)
        
        else: 
            self.um_train_ds=UltrametricMnistDataset(filtered_train_df, transform=None)

            if self.mode == 'um':
                self.train_sampler = UltraMetricSampler(self.um_train_ds, self.chain, train_class_index,self.nb_classes)
                self.test_ds = MnistLinearDataset(filtered_test_df, transform=None)
                
            elif self.mode == 'split':
                split_chain = np.random.randint(0, high=self.nb_classes, size=1000)
                self.train_sampler = BinarySampler(self.um_train_ds, train_class_index, split_chain, train=True)
                self.test_ds = UltrametricMnistDataset(filtered_test_df, transform=None)
                self.test_sampler = BinarySampler(self.test_ds, test_class_index, split_chain, train=False)
                                
    def train_dataloader(self):
        return DataLoader(self.um_train_ds, batch_size=self.batch_size_train, shuffle=False, 
                          num_workers=self.num_workers, sampler=self.train_sampler)

    def val_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size_test, shuffle=False, 
                          num_workers=self.num_workers, sampler=self.test_sampler)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size_test, shuffle=False, 
                          num_workers=self.num_workers, sampler=self.test_sampler)


class SynthDataModule(pl.LightningDataModule):
    def __init__(self, nb_classes: int, data_dir: str = "./", batch_size_train=128, batch_size_test=1000, 
                 num_workers=4, mode='rand', leaf_length=1000, total_sample_nb=1000000, max_depth=12,
                 noise_level=1, p_flip=0.01):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.num_workers = num_workers
        self.mode = mode
        self.train_sampler = None
        self.test_sampler = None
        self.leaf_length = leaf_length
        self.nb_classes = nb_classes
        self.classes = np.arange(0, nb_classes)
        self.tree = SynthUltrametricTree(max_depth=max_depth, p_flip=p_flip, 
                                         leaf_length=self.leaf_length, shuffle_labels=True,
                                         noise_level=noise_level)
                             
    def setup(self, stage = None):
        X, y = self.tree.leaves, self.tree.labels
        X = np.array([el/np.sum(el) for el in X]) # normalize input
        print(X.shape)
        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)

        train_class_index = [np.where(y_train==class_label)[0] for class_label in self.classes]
        test_class_index = [np.where(y_test==class_label)[0] for class_label in self.classes]

        y_train = [one_hot_labels(el, self.nb_classes) for el in y_train]
        y_test = [one_hot_labels(el, self.nb_classes) for el in y_test]

        y_train = torch.stack(y_train)
        y_test = torch.stack(y_test)
        
        self.um_train_ds=TensorDataset(X_train, y_train)            
        self.test_ds=TensorDataset(X_test, y_test)            

        if self.mode == 'um':
            self.markov_chain = generate_markov_chain(chain_length=self.total_sample_nb, T=0.4, 
                                        tree_levels=self.max_depth, dia=0).tolist()
            assert(len(set(self.markov_chain)) == self.tree.nb_classes) #assert all classes are represented in the Markov chain
            self.train_sampler = UltraMetricSampler(self.um_train_ds, self.markov_chain, 
                                                    train_class_index, self.nb_classes, do_resampling=False)

        elif self.mode == 'split':
            split_chain = np.random.randint(0, high=self.nb_classes, size=1000)
            self.train_sampler = BinarySampler(self.um_train_ds, train_class_index, split_chain, train=True)
            self.test_sampler = BinarySampler(self.test_ds, test_class_index, split_chain, train=False)
                                
    def train_dataloader(self):
        return DataLoader(self.um_train_ds, batch_size=self.batch_size_train, shuffle=False, 
                          num_workers=self.num_workers, sampler=self.train_sampler)

    def val_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size_test, shuffle=False, 
                          num_workers=self.num_workers, sampler=self.test_sampler)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size_test, shuffle=False, 
                          num_workers=self.num_workers, sampler=self.test_sampler)
