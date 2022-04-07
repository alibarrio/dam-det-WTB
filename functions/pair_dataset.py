import os
import glob
import time

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from functions.utils import fixed_image_standardization

class siamese_dataset(torch.utils.data.IterableDataset):
    def __init__(self, path, shuffle_pairs=True, resize_image=[256, 256], augment=False):
        '''
        Create an iterable dataset from a directory containing sub-directories of 
        entities with their images contained inside each sub-directory.
            Parameters:
                    path (str):                 Path to directory containing the dataset.
                    shuffle_pairs (boolean):    Pass True when training, False otherwise. When set to false, the image pair generation will be deterministic
                    resize_image: (2-tupla):    [width_resolution, height_resolution]
                    augment (boolean):          When True, images will be augmented using a standard set of transformations.
            Returns:
                    output (torch.Tensor): shape=[batch_size, 1], Similarity of each pair of images. # TODO: something wrong in the output definition
        '''
        self.path = path
        self.feed_shape = resize_image
        self.shuffle_pairs = shuffle_pairs
        self.augment = augment

        if self.augment:
            # If images are to be augmented, add extra operations for it (first two).
            #  TODO: evaluate which combination of transformations is the best
            self.transform = transforms.Compose([
                transforms.Resize(self.feed_shape),
                transforms.RandomHorizontalFlip(p=0.5),
                np.float32,
                transforms.ToTensor(),
                fixed_image_standardization
            ])
        else:
            # If no augmentation is needed then apply only the normalization and resizing operations.
            self.transform = transforms.Compose([
                transforms.Resize(self.feed_shape),
                np.float32,
                transforms.ToTensor(),
                fixed_image_standardization   
            ])

        self.create_pairs()

    def create_pairs(self):
        '''
        Creates two lists of indices that will form the pairs, to be fed for training or evaluation.
        '''

        self.image_paths = glob.glob(os.path.join(self.path, "*/*.jpg"))
        self.image_paths_ext = []
        self.image_classes = []
        self.class_indices = {}
        
        repes = 1

        # Ali
        if self.shuffle_pairs:
            for path in self.image_paths:
                for i in range(repes):
                    self.image_paths_ext.append(path)
        #
        # Ali     
        if self.shuffle_pairs:
            for image_path in self.image_paths_ext:
                image_class = image_path.split(os.path.sep)[-2]
                self.image_classes.append(image_class)
                if image_class not in self.class_indices:
                    self.class_indices[image_class] = []
                self.class_indices[image_class].append(self.image_paths_ext.index(image_path))
        #
        else:
            for image_path in self.image_paths:
                image_class = image_path.split(os.path.sep)[-2]
                self.image_classes.append(image_class)
                if image_class not in self.class_indices:
                    self.class_indices[image_class] = []
                self.class_indices[image_class].append(self.image_paths.index(image_path))

        
        if self.shuffle_pairs:
            # Ali
            self.indices1 = np.repeat(np.arange(len(self.image_paths)), repes)
            #
            np.random.seed(int(time.time()))
            np.random.shuffle(self.indices1)
        else:
            # If shuffling is set to off, set the random seed to 1, to make it deterministic.
            self.indices1 = np.arange(len(self.image_paths))
            np.random.seed(1)
        
        # Ali
        if self.shuffle_pairs:
            select_pos_pair = np.random.rand(len(self.image_paths_ext)) < 0.5
        #
        else:
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
            # Ali
            if self.shuffle_pairs:
                image_path1 = self.image_paths_ext[idx]
                image_path2 = self.image_paths_ext[idx2]
            #
            else:
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
        # Ali
        if self.shuffle_pairs:
            return len(self.image_paths_ext)
        #
        else:
            return len(self.image_paths)
