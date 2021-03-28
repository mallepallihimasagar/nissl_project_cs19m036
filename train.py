#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 17:48:41 2021

@author: himasagar
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.optim.lr_scheduler import StepLR
import copy
import cv2

from Nissl_Dataset import Nissl_Dataset
from models import U_Net
from models import ResAttU_Net
from tqdm import tqdm

from check import pixel_accuracy,IoU,dice_metric

#set random seed
torch.manual_seed(0)
np.random.seed(0)

#configurations
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
VALID_SPLIT = 0.2
NUM_EPOCHS = 50
PATIENCE = 10
ALPHA = 0.5
INPUT_CHANNELS=3
OUTPUT_CHANNELS=4  

#loding dataset

nissl_data = Nissl_Dataset(root_dir='Nissl_Dataset',transforms=None,multiclass=True)

Train , Validation = random_split(nissl_data,[nissl_data.__len__()*(1-VALID_SPLIT),VALID_SPLIT])

train_loader = DataLoader(dataset=Train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset=Validation, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


#LOSS FUNCTIONS

#BCEwithLogits loss 
BCE_logit_loss = torch.nn.BCEWithLogitsLoss()
#categorical cross-entropy
CrossEntropy_loss = torch.nn.CrossEntropyLoss()
#dice loss
dice_loss = DiceLoss()

#DEVICE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Training on {device}')


#MODEL
model = U_Net(UnetLayer=5, img_ch=INPUT_CHANNELS, output_ch=OUTPUT_CHANNELS).to(device)
# model2  = ResAttU_Net(UnetLayer=5,img_ch=3,output_ch=4).to(device)

#OPTIMIZER
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#INSTANTIATE STEP LEARNING SCHEDULER CLASS

# step_size: at how many multiples of epoch you decay
# step_size = 1, after every 1 epoch, new_lr = lr*gamma 
# step_size = 2, after every 2 epoch, new_lr = lr*gamma 

# gamma = decaying factor
scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

def train_model(
        model=model,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        scheduler=None,
        train_loader=train_loader,
        val_loader = val_loader,
        criteria = criteria,
        optimizer=optimizer,
        device=device):
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    
    for epoch in range(num_epochs):
        scheduler.step()
        #training phase
        avg_loss=0
        pixel_acc=0
        dice_coef=0
        IoU_cell1=0
        IoU_cell2=0
        IoU_cell3=0
        model.train()
        with tqdm(train_loader) as tdm:
            for index,inputs,masks in enumerate(tdm):
                tdm.set_description(f'Epoch :{epoch}/{num_epochs}, Lr : {scheduler.get_lr()} Training -')
                tdm.set_postfix(
                    loss=avg_loss,
                    pixel_acc=pixel_acc,
                    dice_coef=dice_coef,
                    IoU_cell_123 = (IoU_cell1,IoU_cell2,IoU_cell3)
                    )
                
                inputs = torch.from_numpy(inputs/255).permute(0,3,1,2).to(device)
                targets = torch.from_numyp(masks).to(torch.long).to(device)
                #onehot encoding with [batchsize, num_classes, Width , Height]
                onehot_masks = torch.nn.functional.one_hot(target).permute(0,3,1,2).to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criteria(outputs,targets)
                loss = loss.backward()
                optimizer.step()
                
                #calculate metrics
                avg_loss = (avg_loss*index+loss.item())/(index+1)
                pixel_acc = (pixel_acc*index+pixel_accuracy(outputs.cpu(),targets.cpu()))/(index+1)
                dice_coef = (dice_coef*index+dice_metric(outputs.cpu(),onehot_masks.cpu()))/(index+1)
                IoU_cell1 = (IoU_cell1*index+IoU(outputs.cpu(),onehot_masks.cpu(),cell=1))/(index+1)
                IoU_cell2 = (IoU_cell2*index+IoU(outputs.cpu(),onehot_masks.cpu(),cell=2))/(index+1)
                IoU_cell3 = (IoU_cell3*index+IoU(outputs.cpu(),onehot_masks.cpu(),cell=3))/(index+1)
                
                
        with torch.no_grad():
            model.eval()
            
            avg_loss=0
            pixel_acc=0
            dice_coef=0
            IoU_cell1=0
            IoU_cell2=0
            IoU_cell3=0
            with tqdm(val_loader) as tdm:
                for index,inputs,masks in enumerate(tdm):
                    tdm.set_description(f'Epoch :{epoch}/{num_epochs}, Lr : {scheduler.get_lr()} Validating -')
                    tdm.set_postfix(
                        loss=avg_loss,
                        pixel_acc=pixel_acc,
                        dice_coef=dice_coef,
                        IoU_cell_123 = (IoU_cell1,IoU_cell2,IoU_cell3)
                        )
                    
                    inputs = torch.from_numpy(inputs/255).permute(0,3,1,2).to(device)
                    targets = torch.from_numyp(masks).to(torch.long).to(device)
                    #onehot encoding with [batchsize, num_classes, Width , Height]
                    onehot_masks = torch.nn.functional.one_hot(target).permute(0,3,1,2).to(device)
                    
                    #optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criteria(outputs,targets)
                    
                    
                    #calculate metrics
                    avg_loss = (avg_loss*index+loss.item())/(index+1)
                    pixel_acc = (pixel_acc*index+pixel_accuracy(outputs.cpu(),targets.cpu()))/(index+1)
                    dice_coef = (dice_coef*index+dice_metric(outputs.cpu(),onehot_masks.cpu()))/(index+1)
                    IoU_cell1 = (IoU_cell1*index+IoU(outputs.cpu(),onehot_masks.cpu(),cell=1))/(index+1)
                    IoU_cell2 = (IoU_cell2*index+IoU(outputs.cpu(),onehot_masks.cpu(),cell=2))/(index+1)
                    IoU_cell3 = (IoU_cell3*index+IoU(outputs.cpu(),onehot_masks.cpu(),cell=3))/(index+1)
                    
        if avg_loss_loss < best_loss:
            print("saving best model")
            best_loss = avg_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_model_wts)
    return model


final_model = train_model(
        model=model,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader = val_loader,
        criteria = CrossEntropy_loss,
        optimizer=optimizer,
        device=device
    
    )
model_path = "/content/nissl_project_cs19m036/trained_models/unet.pt"

print('Training completed , saving model weights to {model_path}')
try:
    torch.save(model.state_dict(), model_path)
    print("model weights saved succesfully")
except:
    print("Incorrect model path , failed to save model weights")


#testing a sample
inputs,masks = next(iter(val_loader))
input_org = inputs
masks_org = masks
inputs = torch.from_numpy(inputs/255).to(device).unsqueeze(dim=0)
targets = torch.from_numyp(masks).to(torch.long).to(device).unsqueeze(dim=0)
#onehot encoding with [batchsize, num_classes, Width , Height]
onehot_masks = torch.nn.functional.one_hot(target).permute(0,3,1,2).to(device)
                
#optimizer.zero_grad()
outputs = model(inputs)
loss = criteria(outputs,targets)
                
                
#calculate metrics
avg_loss = (avg_loss*index+loss.item())/(index+1)
pixel_acc = (pixel_acc*index+pixel_accuracy(outputs.cpu(),targets.cpu()))/(index+1)
dice_coef = (dice_coef*index+dice_metric(outputs.cpu(),onehot_masks.cpu()))/(index+1)
IoU_cell1 = (IoU_cell1*index+IoU(outputs.cpu(),onehot_masks.cpu(),cell=1))/(index+1)
IoU_cell2 = (IoU_cell2*index+IoU(outputs.cpu(),onehot_masks.cpu(),cell=2))/(index+1)
IoU_cell3 = (IoU_cell3*index+IoU(outputs.cpu(),onehot_masks.cpu(),cell=3))/(index+1)
    
output = output.cpu().squeeze(dim=0)
output = torch.nn.Softmax(dim=0)(output)
output_mask = torch.argmax(output, dim=0)
cv2.imwrite('output.tif',output_mask.numpy().astype(np.uint8))
cv2.imwrite('input.png',input_org)
cv2.imwrite('target.tif',masks_org)

print("avg_loss = ",avg_loss)
print("pixel accuracy = ",pixel_acc)
print("dice_coef = ",dice_coef)
print("IOU of cell 1,2,3 = ",IoU_cell1,IoU_cell2,IoU_cell3)


                
                
                
                
















