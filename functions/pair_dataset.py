import os
import glob
import time

import numpy as np
from numpy.random import choice
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from facenet_pytorch import fixed_image_standardization

class siamese_dataset(torch.utils.data.IterableDataset):
    def __init__(self, path, train=True, epoch_size=2000):
        '''
        Create an iterable dataset from a directory containing sub-directories of 
        entities with their images contained inside each sub-directory.
            Parameters:
                    path (str):                 Path to directory containing the dataset.
                    shuffle_pairs (boolean):    Pass True when training, False otherwise. When set to false, the image pair generation will be deterministic
                    augment (boolean):          When True, images will be augmented using a standard set of transformations.
            where b = batch size
            Returns:
                    output (torch.Tensor): shape=[b, 1], Similarity of each pair of images
        '''
        self.path = path
        self.train = train
        self.epoch_size = epoch_size
        
        allpaths = glob.glob(os.path.join(self.path, "*/*.jpg"))
        if self.train:
            self.image_paths = []
            for i in range(self.epoch_size):
                self.image_paths.append(choice(allpaths))
        else:
            self.image_paths = allpaths

        self.feed_shape = [3, 256, 256]
        self.train = train
    
        self.transform = transforms.Compose([
            transforms.Resize(self.feed_shape[1:]),
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization   
        ])

        self.create_pairs()
    

    def create_pairs(self):
        '''
        Creates two lists of indices that will form the pairs, to be fed for training or evaluation.
        '''

        self.image_classes = []
        self.class_indices = {}
        
        for image_path in self.image_paths:
            image_class = image_path.split(os.path.sep)[-2]
            self.image_classes.append(image_class)
            if image_class not in self.class_indices:
                self.class_indices[image_class] = []
            self.class_indices[image_class].append(self.image_paths.index(image_path)) 
  
        if self.train:
            self.indices1 = np.arange(len(self.image_paths))
            np.random.seed(int(time.time()))
            np.random.shuffle(self.indices1)
        else:
            # If validation, set the random seed to 1, to make it deterministic.
            self.indices1 = np.arange(len(self.image_paths))
            np.random.seed(1)

        select_pos_pair = np.random.rand(len(self.image_paths)) < 0.5

        self.indices2 = []

        for i, pos in zip(self.indices1, select_pos_pair):
            class1 = self.image_classes[i]
            if pos:
                class2 = class1
            else:
                class2 = np.random.choice(list(set(self.class_indices.keys()) - {class1}))
            idx2 = np.random.choice(self.class_indices[class2])
            self.indices2.append(idx2)
        self.indices2 = np.array(self.indices2)

    def __iter__(self):
        self.create_pairs()

        for idx, idx2 in zip(self.indices1, self.indices2):
            image_path1 = self.image_paths[idx]
            image_path2 = self.image_paths[idx2]

            class1 = self.image_classes[idx]
            class2 = self.image_classes[idx2]

            image1 = Image.open(image_path1).convert("RGB")
            image2 = Image.open(image_path2).convert("RGB")

            if self.transform:
                image1 = self.transform(image1).float()
                image2 = self.transform(image2).float()

            yield (image1, image2), torch.Tensor([class1!=class2]), (class1, class2)
        
    def __len__(self):
        if self.train:
            n_imgs = self.epoch_size
        else:
            n_imgs = len(self.image_paths)
        return n_imgs

