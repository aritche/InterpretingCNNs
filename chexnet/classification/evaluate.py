import cv2
import numpy as np
import os

import csv

import torch






BATCH_SIZE = 256
IM_SIZE = 224

DATA_FILE = './data/Data_Entry_2017.csv'
IMG_DIR = './data/test_224'
CLASSES = ["Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion","Emphysema",
           "Fibrosis","Hernia","Infiltration","Mass","Nodule","Pleural_Thickening","Pneumonia",
           "Pneumothorax"]

"""
##############################
DATASET LOADING AND MANAGEMENT
##############################
"""

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

    # Normalise on mean/sdev of imagenet
    means = np.array([0.485, 0.456, 0.406])
    sdev = np.array([0.229, 0.224, 0.225])
    im = (im - means)/sdev

    im = np.array(np.transpose(im, (2, 0, 1)))
    im = torch.from_numpy(im)

    base_fn = fn.split('/')[-1].split('.')[0]
    label = fn_to_label[base_fn]

    return [im, label]

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

print("Loading image labels...")
fn_to_label = load_labels(DATA_FILE)

print("Creating dataset...")
test_dataset = CustomDataset()

print("Constructing data loader...")
testloader = torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_SIZE)

"""
############################
COMPUTING EVALUATION METRICS
############################
"""
"""
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

fn = input(".pth file:")
model = torch.load(fn)

model.eval()

COLUMN = 1 # cardiomegaly
for K in [1, 3, 5]:
    with torch.no_grad(): # speed up computation by disabling backprop.
      correct = 0
      incorrect = 0
      i = 0
      for i, (inputs, classes) in enumerate(testloader):
          inputs = inputs.float().to(device) # convert input tensor to appropriate format
          classes = classes.long().to(device) # convert labels tensor to appropriate format
          outputs = model(inputs) # forward propagate

          x = 0
          for item in classes:
            if (item[COLUMN].item() == 1): # if target has COLUMN = True
              topk, indices = torch.topk(outputs[x], K)
              tops = indices.to('cpu').numpy()
              if 1 in tops: # if COLUMN in top-K
                correct += 1
              else:
                incorrect += 1
            x += 1
          print(i)
    print("Top %d accuracy for class %d: %.2f" % (K, COLUMN, float(correct)/float(correct+incorrect)))
"""


from sklearn import metrics

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

fn = input(".pth file:")
model = torch.load(fn)

model.eval()

truths = []
predictions = []

print("---------- %s ----------" % (fn))
for c in range(14):
    truths.append([])
    predictions.append([])
    print("%d\t" % (c), end="")
print("")

with torch.no_grad(): # speed up computation by disabling backprop.
  for i, (inputs, classes) in enumerate(testloader):
      inputs = inputs.float().to(device) # convert input tensor to appropriate format
      classes = classes.long().to(device) # convert labels tensor to appropriate format
      outputs = model(inputs) # forward propagate

      # Get prediction for each input in batch
      # zip true labels and predictions, and iterate over them
      for c in range(14):
        for i in range(len(classes)):
          truths[c].append(classes[i][c].cpu().numpy())
          predictions[c].append(outputs[i][c].cpu().numpy())

truths = np.array(truths)
predictions = np.array(predictions)

for c in range(14):
    fp, tp, thresholds = metrics.roc_curve(y_true=truths[c], y_score=predictions[c], pos_label=1)
    result = metrics.auc(fp,tp)
    #print("%.3f\t" % (result), end="")
    print("%.3f" % (result))
#print("")
print('^^^^^^^^^^^^^^^^^^^^^^^^^')
