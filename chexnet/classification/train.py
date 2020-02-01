import cv2
import numpy as np
import os

import csv
from random import randint
from torch.utils.data.sampler import SubsetRandomSampler

import torch
from torch import nn
from torch import optim
from torchvision import models

"""
################
HYPER PARAMETERS
################
"""
BATCH_SIZE = 64
LEARNING_RATE = 0.001
VALID_SPLIT = 0.125
EPOCHS = 100
IM_SIZE = 224

"""
##############
DATA VARIABLES
#############
"""
DATA_FILE = './data/Data_Entry_2017.csv'
IMG_DIR = './data/all_data_224'
CLASSES = ["Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion","Emphysema",
           "Fibrosis","Hernia","Infiltration","Mass","Nodule","Pleural_Thickening","Pneumonia",
           "Pneumothorax"]

"""
###############
MODEL VARIABLES
###############
"""
MODEL_SAVE_PATH = "./trained_models"


"""
##################
DATASET MANAGEMENT
##################
"""
# Loads labels from a .csv file formatted as:
#   column 0: file name for image
#   column 1: valid classes for the image (separated by pipe "|")
# Returns a dictionary mapping an image file name to a 1D tensor of size |CLASSES|.
# 1 indicates that the class is present in the image, 0 indicates it is not present
def load_labels(fn):
    print("Loading image labels as tensors into global dictionary...")
    data = {}
    with open(fn, "r") as f:
        reader = csv.reader(f)
        next(reader, None) # skip the csv header
        for line in reader:
            fn = line[0].split('.')[0] # get the file name
            labels = set(line[1].split('|')) # split raw string on '|' separator to get classes
            image_label = []
            for item in CLASSES:
                if item in labels:
                    image_label.append(1)
                else:
                    image_label.append(0)
            data[fn] = torch.FloatTensor(image_label)

    return data

# Class for a custom dataset (allows for custom item loading method)
# https://discuss.pytorch.org/t/loading-huge-data-functionality/346/2
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data_files = os.listdir(IMG_DIR)
        self.data_files.sort()

    def __getitem__(self, idx):
        return load_file(self.data_files[idx])

    def __len__(self):
        return len(self.data_files)

# Helper function for loading a file from the dataset (given the file path)
def load_file(fn):
    global fn_to_label # access the global map of fn to classification label tensor

    im = cv2.imread(IMG_DIR + '/' + fn)
    im = np.float32(cv2.resize(im, (IM_SIZE, IM_SIZE)))

    im /= 255 # normalise to [0,1]

    # Random horizontal flip with probability
    if (randint(0,1) == 0):
        im = cv2.flip(im, 1)

    # Normalise on mean/sdev of imagenet
    means = np.array([0.485, 0.456, 0.406])
    sdev = np.array([0.229, 0.224, 0.225])
    im = (im - means)/sdev

    im = np.array(np.transpose(im, (2, 0, 1)))
    im = torch.from_numpy(im)

    base_fn = fn.split('/')[-1].split('.')[0]
    label = fn_to_label[base_fn]

    return [im, label]

"""
##############
LOAD THE DATA
##############
"""
# Load the labels
print("Loading image labels...")
fn_to_label = load_labels(DATA_FILE)

# Create the training/validation datasets
print("Instantiating datasets...")
train_dataset = CustomDataset()
valid_dataset = CustomDataset()

print("Splitting into validation and training sets...")
# Assign an index to each image, and split indices into training/valid
num_images = len(train_dataset)
indices = list(range(num_images))
split = int(np.floor(VALID_SPLIT * num_images))
np.random.shuffle(indices)
train_idx, valid_idx = indices[split:], indices[:split]

# Instantiate random samplers that sample from a list of indices without replacement
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

print("Constructing data loaders...")
trainloader = torch.utils.data.DataLoader(train_dataset,sampler=train_sampler, batch_size=BATCH_SIZE)
validloader = torch.utils.data.DataLoader(valid_dataset, sampler=valid_sampler, batch_size=BATCH_SIZE)


"""
########################
CONSTRUCT THE CLASSIFIER
########################
"""
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# Create CheXNet
model = models.densenet121(pretrained=True)
for param in model.parameters(): # turn on training for all layers
    param.requires_grad = True

model.classifier = nn.Sequential(nn.Linear(1024, 14), # adjust the fully-connected part
                                 nn.Sigmoid())
for param in model.classifier.parameters(): # turn on training for fully-connected part
    param.requires_grad = True


"""
###################
CLASSIFIER TRAINING
###################
"""
# Loss function
def MultiTargetEntropy(predicted, target):
    return torch.sum(-1 * target * torch.log(predicted) - (1 - target) * torch.log(1 - predicted))

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1)

# Send model to the device
model.to(device)

print("Training...")
print("EPOCH\tT_L\tV_L")
train_losses = []
valid_losses = []

for epoch in range(EPOCHS):
    steps = 0
    train_loss = 0
    total_examples = 0
    for inputs, labels in trainloader:
        steps += 1 # count number of batches
        if (steps % 500 == 0):
          print("%d/%d" % (steps, len(trainloader)))
        inputs, labels = inputs.float().to(device), labels.long().to(device) # Converted to float for compatability
        
        optimizer.zero_grad()
        output = model.forward(inputs)
        total_examples += len(inputs)

        loss = MultiTargetEntropy(output, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= total_examples

   
    # ---- EVALUATION ----
    valid_loss = 0
    total_valid_examples = 0

    model.eval()
    print("Evaluating...")
    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.float().to(device), labels.long().to(device) # Converted to float for compatability
            output = model.forward(inputs)
        
            total_valid_examples += len(inputs)
            
            loss = MultiTargetEntropy(output, labels)
            valid_loss += loss.item()

    scheduler.step(valid_loss)
    valid_loss /= total_valid_examples


    # store metrics
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    # Print status
    print("%d\t%.4f\t%.4f" % (epoch, train_loss, valid_loss))

    val_str = "%.2f" % (valid_loss)
    torch.save(model, MODEL_SAVE_PATH + '/epoch_' + str(epoch) + '_' + val_str + '.pth')
    
    model.train()
