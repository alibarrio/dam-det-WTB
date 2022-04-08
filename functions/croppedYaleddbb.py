# Interface for Cropped Yale dataset
# It is composed by 38 folders, and inside each one there is 64 images of one person.

import torch
from torch.utils.data import Dataset, DataLoader
import os
from numpy.random import choice as npc
import numpy as np
import time
import random
import torchvision.datasets as dset
from PIL import Image


def splitDataSet(dataPath, percentage):
    np.random.seed(0)
    print("Spliting dataset into training and test subsets")
    all_paths = []
    train_path = []
    test_path = []

    # Create a list of list with all the images and classes
    idx = 0
    for personPath in os.listdir(dataPath):
        all_paths.append([])
        for samplePath in os.listdir(os.path.abspath(os.path.join(dataPath, personPath))):
            all_paths[idx].append(os.path.abspath(os.path.join(dataPath, personPath, samplePath)))
        idx += 1

    num_classes = idx
    for i in range(num_classes):
        num_samples = len(all_paths[i])
        num_train_samples = int(np.round(num_samples * percentage))
        ip = np.random.permutation(num_samples).tolist()
        train_path.append(list(all_paths[i][j] for j in ip[:num_train_samples]))
        test_path.append(list(all_paths[i][j] for j in ip[num_train_samples:]))

    return train_path, test_path


class croppedYaleTrain(Dataset):

    def __init__(self, dataPath, transform=None):
        super(croppedYaleTrain, self).__init__()
        np.random.seed(0)
        self.transform = transform
        self.dataPath = dataPath
        self.num_classes = len(dataPath)

    def __len__(self):
        return 21000000  #Why? Just a large number?

    def __getitem__(self, index):
        label = None
        img1 = None
        img2 = None
        # get image from same class
        if index % 2 == 1:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            image1_path = random.choice(self.dataPath[idx1])
            image1 = Image.open(image1_path)
            image2_path = random.choice(self.dataPath[idx1])
            image2 = Image.open(image2_path)
        # get image from different class
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            image1_path = random.choice(self.dataPath[idx1])
            image1 = Image.open(image1_path)
            image2_path = random.choice(self.dataPath[idx2])
            image2 = Image.open(image2_path)

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))


class croppedYaleTest(Dataset):

    def __init__(self, dataPath, transform=None, times=200, way=20):
        np.random.seed(1)
        super(croppedYaleTest, self).__init__()
        self.transform = transform
        self.times = times
        self.way = way
        self.img1 = None
        self.c1 = None
        self.dataPath = dataPath
        self.num_classes = len(dataPath)

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way  # Defines the proportion of pairs of the same class versus different classes
        label = None
        # generate image pair from same class
        if idx == 0:  # What happens if the first time idx !=0 and self.c1=None
            self.c1 = random.randint(0, self.num_classes - 1)
            image1_path = random.choice(self.dataPath[self.c1])
            self.img1 = Image.open(image1_path)
            image2_path = random.choice(self.dataPath[self.c1])
            img2 = Image.open(image2_path)
        # generate image pair from different class
        else:
            c2 = random.randint(0, self.num_classes - 1)
            while self.c1 == c2:
                c2 = random.randint(0, self.num_classes - 1)
            image2_path = random.choice(self.dataPath[c2])
            img2 = Image.open(image2_path)

        if self.transform:
            img1 = self.transform(self.img1)
            img2 = self.transform(img2)
        return img1, img2
