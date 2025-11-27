from cProfile import label
from random import shuffle
import torch
from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
import numpy as np


class ImpactEchoDatasetClassifier(Dataset):
    
    def __init__(self, 
            X_path,
            y_path,
            sr = [200000, 104200],
            array_size = 1519,
            shuffle=False
    ):
        self.sr = sr
        self.X = np.array([])
        self.array_size = array_size
        self.shuffle = shuffle
        items = 0
        for path in X_path:
            tmp = np.load(path)
            tmp = tmp[:, 0:self.array_size]
            items += tmp.shape[0]
            self.X = np.append(self.X, tmp)
        self.X = np.reshape(self.X, (items, -1))

        # labels for DS1
        self.labels1 =  np.load(y_path[0])
        if 'overlayed' in y_path[0]:
            self.labels1 = np.flip(self.labels1)
        print(f"Loaded dataset {y_path[0]}, with {len(self.labels1)} data points")
        self.labels1[self.labels1 < 1] = 0
        self.labels1[self.labels1 > 0] = 1
        self.labels1 = [int(label) for label in self.labels1]
        self.dataset1_size = len(self.labels1)
        if len(y_path) == 2:
            self.labels2 =  np.load(y_path[1])
            print(f"Loaded dataset {y_path[1]}, with {len(self.labels2)} data points")
            if 'overlayed' in y_path[1]:
                self.labels2 = np.flip(self.labels2)
            self.y = np.append(self.labels1, self.labels2)

        elif len(y_path) > 2:
            self.labels3 =  np.load(y_path[1])
            self.labels3[self.labels3 < 1] = 0
            self.labels3[self.labels3 > 0] = 1
            self.labels3 = [int(label) for label in self.labels3]
            self.y = np.append(self.labels1, self.labels3)
            # import matplotlib.pyplot as plt
            # plt.imshow(np.reshape(self.labels3[0:1178], (31,38)))
            # plt.savefig('test.png')
            self.dataset1_size = len(self.y)
            print(f"Loaded dataset {y_path[1]}, with {len(self.labels3)} data points")
            
            # labels for SDNET
            # self.labels2 =  np.load(y_path[2])
            # self.labels2[self.labels2 ==1] = 0
            # self.labels2[self.labels2 >= 2] = 1
            # self.y = np.append(self.y, self.labels2)

            self.labels2 =  np.load(y_path[2])
            self.labels2[self.labels2 < 1] = 0
            self.labels2[self.labels2 > 0] = 1
            self.labels2 = [int(label) for label in self.labels2]
            print(f"Loaded dataset {y_path[2]}, with {len(self.labels2)} data points")
            self.y = np.append(self.y, self.labels2)
        else:
            self.y = self.labels1

        if self.shuffle:
            self.X_y = np.column_stack((self.X, self.y))
            np.random.shuffle(self.X_y)
            self.y = [int(xa) for xa in self.X_y[:,-1]]
            self.X = self.X_y[:,:-1]

    def __getitem__(self, index):
        return torch.tensor(self.__normalize_data__(self.X[index][:self.array_size]), dtype=torch.double), torch.tensor(self.y[index], dtype=torch.int8)
        
    def __normalize_data__(self, signal):
        return (signal - np.min(signal))/(np.max(signal)-np.min(signal))
    
    def __len__(self):
        return len(self.X)

    def get_labels(self):
        return self.labels1, self.labels2

    def __gettest__(self):
        return torch.tensor(self.X), self.y


class ImpactEchoDatasetCL(Dataset):
    
    def __init__(self, 
                X_path,
                y_path,
                sr = [200000, 104200],
                array_size = 1519,
                shuffle=False,
                augment_data = True):
        self.sr = sr
        self.X = np.array([])
        self.augment_data = augment_data
        self.array_size = array_size
        self.shuffle = shuffle
        items = 0
        for path in X_path:
            tmp = np.load(path)
            tmp = tmp[:, 0:self.array_size]
            items += tmp.shape[0]
            self.X = np.append(self.X, tmp)
        self.X = np.reshape(self.X, (items, -1))

        # labels for DS1
        self.labels1 =  np.load(y_path[0])
        print(f"Loaded dataset {y_path[0]}, with {len(self.labels1)} data points")
        self.labels1[self.labels1 < 1] = 0
        self.labels1[self.labels1 > 0] = 1
        self.labels1 = [int(label) for label in self.labels1]
        self.dataset1_size = len(self.labels1)
        if len(y_path) == 2:
            # labels for SDNET
            self.labels2 =  np.load(y_path[1])
            print(f"Loaded dataset {y_path[1]}, with {len(self.labels2)} data points")
            self.y = np.append(self.labels1, self.labels2)

        elif len(y_path) > 2:

            self.labels3 =  np.load(y_path[1])
            self.labels3[self.labels3 < 1] = 0
            self.labels3[self.labels3 > 0] = 1
            self.labels3 = [int(label) for label in self.labels3]
            self.y = np.append(self.labels1, self.labels3)

            self.dataset1_size = len(self.y)
            print(f"Loaded dataset {y_path[1]}, with {len(self.labels3)} data points")
            
            self.labels2 =  np.load(y_path[2])
            self.labels2[self.labels2 < 1] = 0
            self.labels2[self.labels2 > 0] = 1
            self.labels2 = [int(label) for label in self.labels2]
            print(f"Loaded dataset {y_path[2]}, with {len(self.labels2)} data points")
            self.y = np.append(self.y, self.labels2)
        else:
            self.y = self.labels1

        if self.shuffle:
            self.X_y = np.column_stack((self.X, self.y))
            np.random.shuffle(self.X_y)
            self.y = [int(xa) for xa in self.X_y[:,-1]]
            self.X = self.X_y[:,:-1]

    def __getitem__(self, index):
        if self.augment_data:
            if index < self.dataset1_size or self.shuffle:
                X_datapoint1 = augment_impact_echo_data(self.X[index], self.sr[0])[:self.array_size]
                X_datapoint2 = augment_impact_echo_data(self.X[index], self.sr[0])[:self.array_size]
            else:
                X_datapoint1 = augment_impact_echo_data(self.X[index], self.sr[1])[:self.array_size]
                X_datapoint2 = augment_impact_echo_data(self.X[index], self.sr[1])[:self.array_size]
            return  [
                        torch.tensor(self.__normalize_data__(X_datapoint1), dtype=torch.float), 
                        torch.tensor(self.__normalize_data__(X_datapoint2), dtype=torch.float), 
                        torch.tensor(self.y[index], dtype=torch.int8),
                        torch.tensor(self.__normalize_data__(self.X[index,:self.array_size]), dtype=torch.float)
                ]
        else:
            return torch.tensor(self.__normalize_data__(self.X[index][:self.array_size]), dtype=torch.double), \
                   torch.tensor(self.y[index], dtype=torch.int8)
        
    def __normalize_data__(self, signal):
        return (signal - np.min(signal))/(np.max(signal)-np.min(signal))
    
    def __len__(self):
        return len(self.X)

    def get_labels(self):
        return self.labels1, self.labels2

    def __gettest__(self):
        return torch.tensor(self.X), self.y


from dataloaders.augmentations import *


class ImpactEchoDatasetClassifierAug(Dataset):
    
    def __init__(self, 
            X_path,
            y_path,
            sr = [200000, 104200],
            array_size = 1519,
            shuffle=False,
            use_augmentation=True
    ):
        self.sr = sr
        self.X = np.array([])
        self.array_size = array_size
        self.shuffle = shuffle
        self.use_augmentation = use_augmentation
        
        items = 0
        for path in X_path:
            tmp = np.load(path)
            tmp = tmp[:, 0:self.array_size]
            items += tmp.shape[0]
            self.X = np.append(self.X, tmp)
        self.X = np.reshape(self.X, (items, -1))
        
        # Pre-compute normalization stats for faster processing
        self.X_min = np.min(self.X, axis=1, keepdims=True)
        self.X_max = np.max(self.X, axis=1, keepdims=True)
        self.X_range = self.X_max - self.X_min
        self.X_range[self.X_range == 0] = 1.0  # Avoid division by zero

        # labels for DS1
        self.labels1 =  np.load(y_path[0])
        if 'overlayed' in y_path[0]:
            self.labels1 = np.flip(self.labels1)
        print(f"Loaded dataset {y_path[0]}, with {len(self.labels1)} data points")
        self.labels1[self.labels1 < 1] = 0
        self.labels1[self.labels1 > 0] = 1
        self.labels1 = [int(label) for label in self.labels1]
        self.dataset1_size = len(self.labels1)
        
        if len(y_path) == 2:
            self.labels2 =  np.load(y_path[1])
            print(f"Loaded dataset {y_path[1]}, with {len(self.labels2)} data points")
            if 'overlayed' in y_path[1]:
                self.labels2 = np.flip(self.labels2)
            self.y = np.append(self.labels1, self.labels2)

        elif len(y_path) > 2:
            self.labels3 =  np.load(y_path[1])
            self.labels3[self.labels3 < 1] = 0
            self.labels3[self.labels3 > 0] = 1
            self.labels3 = [int(label) for label in self.labels3]
            self.y = np.append(self.labels1, self.labels3)
            self.dataset1_size = len(self.y)
            print(f"Loaded dataset {y_path[1]}, with {len(self.labels3)} data points")
            
            self.labels2 =  np.load(y_path[2])
            self.labels2[self.labels2 < 1] = 0
            self.labels2[self.labels2 > 0] = 1
            self.labels2 = [int(label) for label in self.labels2]
            print(f"Loaded dataset {y_path[2]}, with {len(self.labels2)} data points")
            self.y = np.append(self.y, self.labels2)
        else:
            self.y = self.labels1

        # Store original dataset size before shuffling
        self.original_size = len(self.X)
        
        if self.shuffle:
            self.X_y = np.column_stack((self.X, self.y))
            np.random.shuffle(self.X_y)
            self.y = [int(xa) for xa in self.X_y[:,-1]]
            self.X = self.X_y[:,:-1]
        
        print(f"Dataset size: {self.original_size} original samples")
        if self.use_augmentation:
            print(f"With augmentation: {2 * self.original_size} total samples")

    def __getitem__(self, index):
        if self.use_augmentation and index >= self.original_size:
            # Return augmented data for second half of indices
            original_index = index - self.original_size
            # Get augmented data
            X_augmented = augment_impact_echo_data(self.X[original_index], self.sr[0])[:self.array_size]
            return torch.tensor(self.__normalize_data_fast__(X_augmented), dtype=torch.double), torch.tensor(self.y[original_index], dtype=torch.int8)
        else:
            # Return original data with pre-computed normalization
            normalized = (self.X[index][:self.array_size] - self.X_min[index]) / self.X_range[index]
            return torch.tensor(normalized, dtype=torch.double), torch.tensor(self.y[index], dtype=torch.int8)
        
    def __normalize_data_fast__(self, signal):
        """Fast normalization for augmented data"""
        sig_min = signal.min()
        sig_range = signal.max() - sig_min
        if sig_range == 0:
            return signal - sig_min
        return (signal - sig_min) / sig_range
    
    def __len__(self):
        if self.use_augmentation:
            return 2 * self.original_size
        else:
            return self.original_size

    def get_labels(self):
        return self.labels1, self.labels2

    def __gettest__(self):
        return torch.tensor(self.X), self.y
    
    def get_class_distribution(self):
        """Helper method to check class balance in the augmented dataset"""
        if self.use_augmentation:
            # Labels are duplicated for augmented data
            augmented_labels = self.y + self.y
        else:
            augmented_labels = self.y
        
        unique, counts = np.unique(augmented_labels, return_counts=True)
        return dict(zip(unique, counts))
