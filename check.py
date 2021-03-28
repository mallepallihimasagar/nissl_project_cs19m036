import torch
import matplotlib.pyplot as plt
from prac import Nissl_Dataset
import cv2
import numpy as np
from loss_functions import DiceLoss

'''
dataset = Nissl_Dataset(root_dir='Nissl_Dataset',multiclass=True)

img,mask = next(iter(dataset))
input_img = torch.from_numpy(img/255).permute(2,1,0)
target = torch.from_numpy(mask).to(torch.long)
#print(input_img.shape,target.shape)
#plt.imshow(mask)

onehotmask = torch.nn.functional.one_hot(target)
numpy_onehot_mask = onehotmask.numpy()

retrieved_mask = torch.argmax(onehotmask, dim=2)
numpy_retrieved_mask = retrieved_mask.numpy()

plt.imshow(mask)
plt.imshow(numpy_retrieved_mask)
#cv2.imwrite('mask.tif',mask.astype(np.uint8))

#loss functions 
#BCEwithLogits loss 
BCE_logit_loss = torch.nn.BCEWithLogitsLoss()
#categorical cross-entropy
CrossEntropy_loss = torch.nn.CrossEntropyLoss()
#dice loss
dice_loss = DiceLoss()

'''
#accuracy metrics
def dice_metric(inputs, targets,from_logits=True):
    if from_logits:
        torch.sigmoid(inputs)
    #target should be one-hot encoding
    inputs = inputs.reshape(-1).numpy()
    targets = targets.reshape(-1).numpy()
    inputs = (inputs>0.5).astype(np.uint8)
    intersection = 2.0 * (targets * inputs).sum()
    union = targets.sum() + inputs.sum()
    if targets.sum() == 0 and inputs.sum() == 0:
        return 1.0

    return intersection / union

def pixel_accuracy(inputs,targets,from_logits=True):
    if from_logits:
        softmax = torch.nn.Softmax(dim=1)
        inputs = softmax(inputs)
        inputs = torch.argmax(inputs,dim=1)
    inputs = inputs.view(-1).numpy()
    targets = targets.view(-1).numpy()
    
    return (inputs==targets).sum()/targets.shape[0]

def IoU(inputs,targets,cell=None,from_logits=True):
    if from_logits:
        torch.sigmoid(inputs)
     #target should be one-hot encoding
    if cell==1:
        inputs = inputs[:,1,:,:]
        targets = targets[:,1,:,:]
    elif cell==2:
        inputs = inputs[:,2,:,:]
        targets = targets[:,2,:,:]
    elif cell==3:
        inputs = inputs[:,3,:,:]
        targets = targets[:,3,:,:]
    
    inputs = inputs.reshape(-1).numpy()
    targets = targets.view(-1).numpy()
    inputs = (inputs>0.5).astype(np.uint8)
    intersection = (inputs&targets).sum()
    #union = target.sum() + inputs.sum()
    union = (inputs|targets).sum()
    if targets.sum() == 0 and inputs.sum() == 0:
        return 1.0    
    return intersection/union

        