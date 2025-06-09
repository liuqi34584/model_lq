from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.io

class MIT_BIH_AF_SegLoader(object):
    def __init__(self, mode="train"):

        # name = "CPSC2025_512"
        name = "afdb_5r_768"

        self.mode = mode

        if self.mode == "train":
            train_X1 = scipy.io.loadmat("./dataset/"+ name + "/train_X.mat")
            train_Y = scipy.io.loadmat( "./dataset/"+ name + "/train_Y.mat")

            self.train = train_X1["train_X"]
            self.train_labels = train_Y["train_Y"]

            print("train", self.train.shape)

        if self.mode == 'valid':

            valid_X1 = scipy.io.loadmat("./dataset/"+ name + "/valid_X.mat")
            valid_Y = scipy.io.loadmat( "./dataset/"+ name + "/valid_Y.mat")

            self.valid = valid_X1["valid_X"]
            self.valid_labels = valid_Y["valid_Y"]

            print("valid", self.valid.shape)

        if self.mode == 'test':

            test_X = scipy.io.loadmat("./dataset/"+ name + "/test_X.mat")
            test_Y = scipy.io.loadmat("./dataset/"+ name + "/test_Y.mat")

            self.test = test_X["test_X"]
            self.test_labels = test_Y["test_Y"]

            print("test", self.test.shape)

    def __len__(self):

        if self.mode == "train":
            return len(self.train)
        if self.mode == 'valid':
            return len(self.valid)
        if self.mode == 'test':
            return len(self.test)

    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.train[index]), np.long(self.train_labels[index])
        if (self.mode == 'valid'):
            return np.float32(self.valid[index]), np.long(self.valid_labels[index])
        if (self.mode == 'test'):
            return np.float32(self.test[index]), np.long(self.test_labels[index])

class CPSC2025_SegLoader(object):
    
    def __init__(self, mode="train"):
        
        # name = "cpsc_5r"
        # name = "CPSC2025_512"
        # name = "CPSC2025_768"
        name = "CPSC2025_BME2024_768"

        self.mode = mode
        if self.mode == "train":
            train_X1 = scipy.io.loadmat("./dataset/"+ name + "/train_X.mat")
            train_Y = scipy.io.loadmat( "./dataset/"+ name + "/train_Y.mat")

            self.train = train_X1["train_X"]
            self.train_labels = train_Y["train_Y"]

            print("train", self.train.shape)

        if self.mode == 'valid':

            valid_X1 = scipy.io.loadmat("./dataset/"+ name + "/valid_X.mat")
            valid_Y = scipy.io.loadmat( "./dataset/"+ name + "/valid_Y.mat")

            self.valid = valid_X1["valid_X"]
            self.valid_labels = valid_Y["valid_Y"]

            print("valid", self.valid.shape)

        if self.mode == 'test':

            test_X = scipy.io.loadmat("./dataset/"+ name + "/test_X.mat")
            test_Y = scipy.io.loadmat("./dataset/"+ name + "/test_Y.mat")

            self.test = test_X["test_X"]
            self.test_labels = test_Y["test_Y"]

            print("test", self.test.shape)

    def __len__(self):

        if self.mode == "train":
            return len(self.train)
        if self.mode == 'valid':
            return len(self.valid)
        if self.mode == 'test':
            return len(self.test)

    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.train[index]), np.long(self.train_labels[index])
        if (self.mode == 'valid'):
            return np.float32(self.valid[index]), np.long(self.valid_labels[index])
        if (self.mode == 'test'):
            return np.float32(self.test[index]), np.long(self.test_labels[index])



def get_loader_segment(batch_size, mode='train', dataset='MIT_BIH_AF'):
    if (dataset == 'MIT_BIH_AF'):
        dataset = MIT_BIH_AF_SegLoader(mode)
    if (dataset == 'CPSC2025'):
        dataset = CPSC2025_SegLoader(mode)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True)
    return data_loader
