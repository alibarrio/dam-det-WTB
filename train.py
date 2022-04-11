# Damage detection and recognition training pipeline

from functions.utils import fixed_image_standardization
from functions import training
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import os
from functions import SiameseInceptionResnetV1, siamese_dataset, pass_epoch, croppedYaleTrain, croppedYaleTest, splitDataSet, Siamese
import pickle
import time
import sys
import gflags


if __name__ == '__main__':

    # Parameters
    Flags = gflags.FLAGS
    gflags.DEFINE_string("data_path", "dataset/all", "training folder")
    gflags.DEFINE_string("model_path", "models", "path to store model")
    gflags.DEFINE_bool("load_model", False, "Whether load a pretrained model or not")
    gflags.DEFINE_string("load_model_path", "models/model.pt", "pathname to load model")
    gflags.DEFINE_integer("imHeight", 105, "Image height")
    gflags.DEFINE_integer("imWidth", 105, "Image wifth")
    gflags.DEFINE_integer("way", 20, "how much way one-shot learning")
    gflags.DEFINE_string("times", 400, "number of samples to test accuracy")
    gflags.DEFINE_integer("workers", 4, "number of dataLoader workers")
    gflags.DEFINE_integer("batch_size", 128, "number of batch size")
    gflags.DEFINE_float("lr", 0.00006, "learning rate")
    gflags.DEFINE_integer("show_every", 10, "show result after each show_every iter.")
    gflags.DEFINE_integer("save_every", 100, "save model after each save_every iter.")
    gflags.DEFINE_integer("test_every", 100, "test model after each test_every iter.")
    gflags.DEFINE_integer("max_iter", 50000, "number of iterations before stopping")
    gflags.DEFINE_bool("colab", False, "If Colab is used, use a reduced number of max_iter")
    gflags.DEFINE_integer("max_iter_colab", 5000, "number of iterations before stopping if using Colab")
    gflags.DEFINE_string("gpu_ids", "0,1,2,3", "gpu ids used to train")

    Flags(sys.argv)

    # Process arguments
    os.makedirs(Flags.model_path, exist_ok=True)      # Creates model folder if not available
    resize_image = [Flags.imHeight, Flags.imWidth]
    if Flags.colab:
        max_iter = Flags.max_iter_colab
    else:
        max_iter = Flags.max_iter


    # Determine if an nvidia GPU is available
    use_gpu = torch.cuda.is_available()
    print('Using GPU: {}'.format(use_gpu))


    # Define dataset, data augmentation, and dataloader
    # Dataset splitting
    percentage = 0.5  # Percentage of samples used for training
    train_path, test_path = splitDataSet(Flags.data_path, percentage)
    # Transformations
    train_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(resize_image),
        transforms.RandomAffine(15),
        transforms.ToTensor()
    ])
    test_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(resize_image),
        transforms.ToTensor()
    ])
    # Dataset
    train_dataset = croppedYaleTrain(train_path, transform=train_transforms)
    val_dataset = croppedYaleTest(test_path, transform=test_transforms, times=Flags.times, way=Flags.way)
    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=Flags.batch_size, shuffle=False, num_workers=Flags.workers)
    val_loader = DataLoader(val_dataset, batch_size=Flags.way, shuffle=False, num_workers=Flags.workers)


    # Model definition
    loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)  # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    
    net = Siamese(
    classify=False,
    pretrained='vggface2',
    )
    if Flags.load_model:
        net.load_state_dict(torch.load(Flags.load_model_path))
    if use_gpu:
        net.cuda()


    # Training
    # Training setup
    net.train()  # https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
    optimizer = torch.optim.Adam(net.parameters(), lr=Flags.lr)
    optimizer.zero_grad()  # Gradient initialization
    train_loss = []
    loss_val = 0
    time_start = time.time()
    accuracy = []

    # Training loop
    for batch_id, (img1, img2, label) in enumerate(train_loader, start=1):
        if batch_id > max_iter:
            break
        if use_gpu:
            img1, img2, label = Variable(img1.cuda()), Variable(img2.cuda()), Variable(label.cuda())
        else:
            img1, img2, label = Variable(img1), Variable(img2), Variable(label)

        optimizer.zero_grad()  # Gradient reset per batch

        # Prediction and error estimation
        output = net.forward(img1, img2)
        loss = loss_fn(output, label)
        loss_val += loss.item()
        train_loss.append(loss_val)

        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch_id % Flags.show_every == 0:
            print('[%d]\tloss:\t%.5f\ttime lapsed:\t%.2f s' % (
            batch_id, loss_val / Flags.show_every, time.time() - time_start))
            loss_val = 0
            time_start = time.time()

        if batch_id % Flags.save_every == 0:
            torch.save(net.state_dict(), Flags.model_path + '/model-inter-' + str(batch_id + 1) + ".pt")

        if batch_id % Flags.test_every == 0:  # Shouldn´t be set the net in eval mode?
            net.eval()
            right, error = 0, 0
            for _, (test1, test2) in enumerate(val_loader, 1):
                if use_gpu:
                    test1, test2 = test1.cuda(), test2.cuda()
                test1, test2 = Variable(test1), Variable(test2)

                # Prediction and error estimation
                output = net.forward(test1, test2).data.cpu().numpy()
                pred = np.argmax(output)
                if pred == 0:
                    right += 1
                else:
                    error += 1

            acc = right * 1.0 / (right + error)
            print('*' * 70)
            print('[%d]\tTest set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f' % (
            batch_id, right, error, acc))
            print('*' * 70)
            accuracy.append(acc)
            net.train()

    #  learning_rate = learning_rate * 0.95


    # Save loss time series
    with open('train_loss', 'wb') as f:
        pickle.dump(train_loss, f)

    # Accuracy metrics
    acc = 0.0
    for d in accuracy:
        acc += d
    print("#" * 70)
    print("final accuracy: ", acc / 20)  # TODO: Averaging of all validation set?? Shouldn´t be the last one?
