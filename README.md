# Damage Detection System for Wind Turbine Blades using a Siamese Neural Network

An implementation of the [original paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) in pytorch with training and testing on the blade damage dataset. Adapted from: [repository](https://github.com/fangpin/siamese-pytorch).

## Dataset
### Clases distribution

| Damage type | Code |
| ----------- | ----------- |
| Leading Edge Erosion | D-2 |
| Lightning Strike Damage | D-3 |
| Crack - Transverse and Longitudinal | D-4/5 |
| No Damage | D-0 |

![alt text](https://github.com/alibarrio/dam-det-WTB/blob/main/images/dam_type_distr_simp.png)

The 90\% of the dataset is used for training purposes and the rest for testing the model. Out of that 90\% of the training set, an 80\% is for the training phase and a 20\% for the validation phase.

### Preprocessing
([Notebook](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf))
The images of the damaged regions have different shapes and aspect ratios, so the main preprocessing to be done is to resize them to square images in order to avoid distortions, since the input to the network will be square. To do this, the shortest side of the image is filled with zeros, so that the image is not distorted when resized.

![Example](https://github.com/alibarrio/dam-det-WTB/blob/main/images/d45_res.jpg)

### Data Augmentation
The training dataset is augmented with some flips and rotations in order to increase its volume and improve training performance. Specifically, vertical and horizontal flips, and random 0 to 360º rotations are applied.

## Architecture
It is composed of two main modules: first, a training module, where the network is trained to predict the similarity between the two input images (i.e. a classifier decides whether the two input images belong to the same class or not). And a second module that uses the feature vectors extracted by the previous trained network to classify the query images into the different classes by estimating their highest similarity to the support set (a set of representatives of every class drawn for the training set).

![alt text](https://github.com/alibarrio/dam-det-WTB/blob/main/images/diagrama_comp2.jpg)

## Training
Training from scratch:
`!python train.py --flagfile config_train_scratch.cfg`

Training from pretrained weights:
`!python train.py --flagfile config_train_pretrain.cfg`

For the training part, the Siamese Neural Network consist of two twin networks, in this case two Inception Resnets, with tied (shared) parameters. It accepts distinct inputs, which are independently processed by each network twin. As a result two feature vectors (embeddings) are computed, which are jointly processed by an energy function to compute some similarity measurement. For this purpose, a sigmoid function is used to compute the distance between the twin embeddings, predicting the probability that the two inputs belong to the same class. For the two different inputs to the network, the cropped regions of the database are used, extracting for each iteration two random images from the training set that may or may not belong to the same class with the same probability, overcoming the problem of data imbalance. Additionally, it solves the problem of having few samples at the same time as the imbalance between classes, since the input of the network are pairs of images so each image of the dataset can be combined with the rest of the images of the dataset, which makes the number of samples available to train the network considerably larger. That is, from a dataset of n samples, up to n*(n-1)/2 different pairs can be made. For instance, while training a conventional CNN with a dataset of 500 samples available, training a Siamese network with the same dataset there would be up to 124.750 pairs available for training.

The validation is performed every epoch, for a total of 100 epochs. The validation experiments are performed according the N-way K-shot framework, which consist of drawing a set of random pairs from each of the N different classes where only one of them contains two instances of the same class, and the rest contains instances of different classes. The experiment is successful when the same class pair is the one with the higher similarity, and failed when one of the other pairs obtains the higher one. The number of pairs in each experiment is K that along with the number of experiments in each validation phase are parameters of the model. In this case 400 experiments with K = 20 pairs in each are tested every epoch. Notice that this type of validation is much more exigent than the traditional validation of a normal classification, whose equivalent would be two-pair experiments.

## Inferences
([Notebook]([https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf](https://github.com/alibarrio/dam-det-WTB/blob/main/inference.ipynb))
To make predictions, only one of the trained twin networks is used to extract the embedding from the query image. Then, the distance of this feature vector to the other feature vectors belonging to the support set is calculated, deciding whether it belongs to one class or another depending on to which of them have the smaller distance to the query one. This support set of embeddings have been also extracted by the trained network. In this way, the network trained to predict the similarity between two images is used as an N-class classifier (where N is the number of classes present in the support set).

## Results
### Confusion matrix

![Example](https://github.com/alibarrio/dam-det-WTB/blob/main/images/prop_conf_mat.png)

| Class | Support | Accuracy | Precision | Recall | F1-score |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| D-2 | 6 |  | 1.00 | 0.83 | 0.91 | 
| D-3 | 23 |  | 1.00 | 0.91 | 0.91 | 
| D-4/5 | 26 |  | 0.90 | 1.00 | 0.95 | 
| D-0 | 12 |  | 0.92 | 0.92 | 0.92 | 
| **Total** | **67** | **0.94** | **0.95** | **0.92** | **0.93** | 









