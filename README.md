# Damage Detection System for Wind Turbine Blades using a Siamese Neural Network

A implementation of the [original paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) in pytorch with training and testing on the Omniglot dataset.

- *finetuning_siameseNN.ipynb*: run to train the network with the dataset
- *train.py*: training script
- *config_train_pretrain.cfg* and *config_train_scratch.cfg*: pass training args to train.py

/functions
- *create_dataset.py*: create pair dataset for feed the siamese net
- *siamese_resnet.py*: siamese neural network definition

