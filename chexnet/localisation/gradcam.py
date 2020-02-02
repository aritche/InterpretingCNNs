import torch
import cv2
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from glob import glob

class DenseBlock4(nn.Module):
    def __init__(self, model_name='model.pth'):
        super(DenseBlock4, self).__init__()

        self.model = torch.load(model_name)

        self.features = self.model.features[:-1]
        self.batchnorm = self.model.features[-1]
        self.classifier = self.model.classifier

        self.num_filters = self.features.denseblock4.denselayer16.conv2.out_channels

        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features(x)

        x = self.batchnorm(x)
        x = F.relu(x, inplace=True)

        self.activations = x.clone()
        h = x.register_hook(self.activations_hook)

        x = F.adaptive_avg_pool2d(x, (1,1))
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

class DenseBlock1(nn.Module):
    def __init__(self, model_name='model.pth'):
        super(DenseBlock1, self).__init__()

        self.model = torch.load(model_name)

        self.part1 = self.model.features[:5]
        self.part2 = self.model.features[5:]
        self.part3 = self.model.classifier

        self.num_filters = self.model.features.denseblock1.denselayer6.conv2.out_channels

        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.part1(x)
        self.activations = x.clone()
        h = x.register_hook(self.activations_hook)
        x = self.part2(x)

        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = torch.flatten(x, 1)

        x = self.part3(x)

        return x

class DenseBlock2(nn.Module):
    def __init__(self, model_name='model.pth'):
        super(DenseBlock2, self).__init__()

        self.model = torch.load(model_name, map_location=lambda storage, loc: storage) # map_locations has to do with converting from CUDA to CPU

        self.part1 = self.model.features[:7]
        self.part2 = self.model.features[7:]
        self.part3 = self.model.classifier

        self.num_filters = self.model.features.denseblock2.denselayer12.conv2.out_channels

        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.part1(x)
        self.activations = x.clone()
        h = x.register_hook(self.activations_hook)
        x = self.part2(x)

        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = torch.flatten(x, 1)

        x = self.part3(x)

        return x

class DenseBlock3(nn.Module):
    def __init__(self, model_name='model.pth'):
        super(DenseBlock3, self).__init__()

        self.model = torch.load(model_name)

        self.part1 = self.model.features[:9]
        self.part2 = self.model.features[9:]
        self.part3 = self.model.classifier

        self.num_filters = self.model.features.denseblock3.denselayer24.conv2.out_channels

        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.part1(x)
        self.activations = x.clone()
        h = x.register_hook(self.activations_hook)
        x = self.part2(x)

        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = torch.flatten(x, 1)

        x = self.part3(x)

        return x

def preprocess(fn, size):
    im = cv2.imread(fn)
    im = np.float32(cv2.resize(im, (size, size)))

    im /= 255 # normalise to [0,1]

    # Normalise on mean/sdev of imagenet
    means = np.array([0.485, 0.456, 0.406])
    sdev = np.array([0.229, 0.224, 0.225])
    im = (im - means)/sdev

    im = np.array(np.transpose(im, (2, 0, 1)))
    im = torch.from_numpy(im)
    im = torch.unsqueeze(im, dim=0)

    return torch.FloatTensor(im.float()).to('cuda')

def grad_cam(im_path, model, block_name, backprop_index):
    # Feed forward
    img = preprocess(im_path, 224)

    # Init model
    model.eval()
    output = model(img)

    # Backprop
    index_num = backprop_index # class to produce a visualisation for
    output[:, index_num].backward()

    # Establish file name based on top-k predictions
    if len(output[0]) < 5:
        topk, indices = torch.topk(output, len(output[0]))
    else:
        topk, indices = torch.topk(output, 5)
    indices = indices.cpu()[0].numpy()
    prediction_label = ""
    for item in indices[:-1]:
        prediction_label += str(item) + "_"
    prediction_label += str(indices[-1])
    #print(output[:, index_num])

    # pull the gradients out of the model
    gradients = model.gradients

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    #print(pooled_gradients[0]*100/100)

    # get the activations of the last convolutional layer
    activations = model.activations.detach()

    # weight the channels by corresponding gradients
    #num_filters = model.num_filters
    num_filters = activations.size(1)
    #print(num_filters)
    for i in range(num_filters):
        #activations[:, i, :, :] *= 1e10*pooled_gradients[i]
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    #heatmap = torch.mean(activations, dim=1).squeeze()/1e10
    heatmap = torch.mean(activations, dim=1).squeeze()
    #print(heatmap)

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap.cpu(), 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.data.numpy()

    # Output the visualisations
    img = cv2.imread(im_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    fn = im_path.split('\\')[-1].split('.')[0]
    cv2.imwrite(fn + '_' + str(prediction_label) +'.png', heatmap)
    #print("Printing...")
    #cv2.imwrite('result.png', heatmap)

    # COLORISE
    #heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    #superimposed_img = heatmap * 0.4 + img * 0.6
    #cv2.imwrite(fn + '_' + block_name + '.png', superimposed_img)

class_name = sys.argv[1]
backprop_index = int(sys.argv[2])
model_path = sys.argv[3] # .pth file

model = DenseBlock4(model_name=model_path)
model.eval()

i = 0
for fn in glob("./data/localisation_classes_256/" + class_name + "/*.png"):
    grad_cam(fn, model, '4', backprop_index)
    i += 1
    print(i)
