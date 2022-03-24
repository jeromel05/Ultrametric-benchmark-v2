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
    def __init__(self, data_source, chain, class_index, nb_classes):
        self.data_source = data_source
        self.chain = chain
        self.class_index = class_index
        self.nb_classes = nb_classes
        self.total_length = 0
        self.temp_length = 0
        self.__iter__(dummy=True)

    def __iter__(self, dummy=False):
        um_indexes = []
        idx = self.temp_length
        nb_previous_occurences = np.zeros(self.nb_classes, dtype=np.int32)
        um_class = 0

        while nb_previous_occurences[um_class] < self.class_index[um_class].size:
            um_idx = self.class_index[um_class][nb_previous_occurences[um_class]]
            um_indexes.append(um_idx)
            um_class = self.chain[idx]
            nb_previous_occurences[um_class] = nb_previous_occurences[um_class] + 1
            idx=idx+1

        self.curr_length = len(um_indexes)
            
        if not dummy:
            self.total_length = self.total_length + len(um_indexes)
            self.temp_length = self.temp_length + len(um_indexes) # keep 2 separate incase we want to reset only every x epochs
        return iter(um_indexes)

    def __len__(self):
        return self.curr_length
    
    def reset_sampler(self):
        np.random.shuffle(self.chain[:self.total_length])
        self.temp_length = 0
    
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


class UMDataModule(pl.LightningDataModule):
    def __init__(self, max_depth: int, data_dir: str = "./", batch_size_train: int=128, batch_size_test: int=1000, 
                 num_workers: int=4, mode: str='rand', chain=None):
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

    def train_dataloader(self):
        return DataLoader(self.um_train_ds, batch_size=self.batch_size_train, shuffle=False, 
                          num_workers=self.num_workers, sampler=self.train_sampler)

    def val_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size_test, shuffle=False, 
                          num_workers=self.num_workers, sampler=self.test_sampler)

    def test_dataloader(self):
        return DataLoader(self.predict_ds, batch_size=self.batch_size_test, shuffle=False, 
                          num_workers=self.num_workers, sampler=self.test_sampler)
    
    def set_chain(self, chain: list):
        self.markov_chain = chain

                             
class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size_train: int=128, batch_size_test: int=1000, 
                 num_workers: int=4, mode: str='rand', chain=None, normalization_transform: torchvision.transforms=None):
        super().__init__(max_depth=3, data_dir=data_dir, batch_size_train=batch_size_train, batch_size_test=batch_size_test, 
                        num_workers=num_workers, mode=mode, chain=chain)
        self.transform = transforms.Compose([transforms.ToTensor(), normalization_transform])
                             
    def setup(self, stage = None):
        train_ds=torchvision.datasets.MNIST(self.data_dir, train=True, download=False, transform=self.transform)
        test_ds=torchvision.datasets.MNIST(self.data_dir, train=False, download=False, transform=self.transform)
        
        def prepare_data(ds):
            filtered_ds=[el for el in ds if el[1] in self.classes] # Remove classes 8,9 to have the right nb
            filtered_df = pd.DataFrame(filtered_ds, columns=[["img", "label"]])
            class_index = [filtered_df.loc[filtered_df["label"].values == class_label].index for class_label in self.classes]
            filtered_df["label"] = filtered_df["label"].apply(one_hot_labels, args=(self.nb_classes,), axis=1)
            return filtered_df, class_index

        filtered_train_df, train_class_index = prepare_data(train_ds)
        filtered_test_df, test_class_index = prepare_data(test_ds)
        
        if self.mode == 'rand':
            self.um_train_ds=MnistLinearDataset(filtered_train_df, transform=None)            
            self.test_ds=MnistLinearDataset(filtered_test_df, transform=None)
        
        else: 
            self.um_train_ds=UltrametricMnistDataset(filtered_train_df, transform=None)

            if self.mode == 'um':
                self.train_sampler = UltraMetricSampler(self.um_train_ds, self.markov_chain, train_class_index, self.nb_classes)
                self.test_ds = MnistLinearDataset(filtered_test_df, transform=None)
                
            elif self.mode == 'split':
                split_chain = np.random.randint(0, high=self.nb_classes, size=1000)
                self.train_sampler = BinarySampler(self.um_train_ds, train_class_index, split_chain, train=True)
                self.test_ds = UltrametricMnistDataset(filtered_test_df, transform=None)
                self.test_sampler = BinarySampler(self.test_ds, test_class_index, split_chain, train=False)
        
        self.predict_ds = MnistPredictDataset(filtered_test_df, transform=None)

                                

class SynthDataModule(UMDataModule):
    def __init__(self, max_depth: int, data_dir: str = "./", batch_size_train: int=128, batch_size_test: int=1000, 
                 num_workers: int=4, mode: str='rand', chain=None, leaf_length=1000, noise_level=1, p_flip=0.01,
                 normalize_data=False, repeat_data=1, test_split=0.1):
        super().__init__(max_depth=max_depth, data_dir=data_dir, batch_size_train=batch_size_train, 
                        batch_size_test=batch_size_test, num_workers=num_workers, mode=mode, chain=chain)
        self.leaf_length = leaf_length
        self.tree = SynthUltrametricTree(max_depth=max_depth, p_flip=p_flip, 
                                         leaf_length=leaf_length, shuffle_labels=True,
                                         noise_level=noise_level)
        self.normalize_data = normalize_data
        self.repeat_data = repeat_data
        self.test_split = test_split
                             
    def setup(self, stage = None):
        X, y = self.tree.leaves, self.tree.labels
        if self.normalize_data:
            X = np.array([el/np.sum(el) for el in X]) # normalize input
        
        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=self.test_split)
        X_train = np.tile(X_train, (self.repeat_data, 1))
        y_train = np.tile(y_train, self.repeat_data)
        print(f"Train data size {X_train.shape}, {y_train.shape}, classes: {set(y_train)}")
        print(f"Val data size {X_test.shape}, {y_test.shape}, classes: {set(y_test)}")

        def prepare_target_data(y):
            class_index = [np.where(y==class_label)[0] for class_label in self.classes]
            y = [one_hot_labels(el, self.nb_classes) for el in y]
            y = torch.stack(y)
            return y, class_index      

        y_train, train_class_index = prepare_target_data(y_train)
        y_test, test_class_index = prepare_target_data(y_test)    
    
        self.um_train_ds=TensorDataset(X_train, y_train)            
        self.test_ds=TensorDataset(X_test, y_test)

        if self.mode == 'um':
            assert(len(set(self.markov_chain)) == self.tree.nb_classes) #assert all classes are represented in the Markov chain
            self.train_sampler = UltraMetricSampler(self.um_train_ds, self.markov_chain, train_class_index, 
                                                    self.nb_classes)

        elif self.mode == 'split':
            split_chain = np.random.randint(0, high=self.nb_classes, size=2000)
            self.train_sampler = BinarySampler(self.um_train_ds, train_class_index, split_chain, train=True)
            self.test_sampler = BinarySampler(self.test_ds, test_class_index, split_chain, train=False)

        self.predict_ds = SynthPredictDataset(X_test)

    