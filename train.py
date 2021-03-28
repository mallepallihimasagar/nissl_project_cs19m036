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
from loss_functions import DiceLoss
from check import pixel_accuracy,IoU,dice_metric

#set random seed
torch.manual_seed(0)
np.random.seed(0)

#configurations
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
VALID_SPLIT = 0.2
NUM_EPOCHS = 200
PATIENCE = 15
ALPHA = 0.5
INPUT_CHANNELS=3
OUTPUT_CHANNELS=4  

#loding dataset

nissl_data = Nissl_Dataset(root_dir='Nissl_Dataset',transforms=None,multiclass=True)
split_size = int(nissl_data.__len__()*VALID_SPLIT)
Train , Validation = random_split(nissl_data,[nissl_data.__len__()-split_size,split_size])

train_loader = DataLoader(dataset=Train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(dataset=Validation, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


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
scheduler = StepLR(optimizer, step_size=40, gamma=0.1)

def train_model(
        model=model,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        scheduler=None,
        train_loader=train_loader,
        val_loader = val_loader,
        criteria = None,
        optimizer=optimizer,
        device=device):
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    patience_test =0
    for epoch in range(num_epochs):
        
        #training phase
        avg_loss=0
        pixel_acc=0
        dice_coef=0
        IoU_cell1=0
        IoU_cell2=0
        IoU_cell3=0
        model.train()
        with tqdm(train_loader) as tdm:
            for index,data_en in enumerate(tdm):
                tdm.set_description(f'Epoch :{epoch}/{num_epochs}, Lr : {scheduler.get_last_lr()} Training -')
                tdm.set_postfix(
                    loss=round(avg_loss,3),
                    pixel_acc=round(pixel_acc,3),
                    dice_coef=dice_coef,
                    IoU_cell_123 = (round(IoU_cell1,3),round(IoU_cell2,3),round(IoU_cell3,3))
                    )
                inputs,masks = data_en
                inputs = inputs.numpy()
                masks = masks.numpy()
                inputs = torch.from_numpy(inputs/255).permute(0,3,1,2).to(device)
                targets = torch.from_numpy(masks).to(torch.long).to(device)
                inputs = inputs.type(torch.float)
                targets = targets.type(torch.long)
                #onehot encoding with [batchsize, num_classes, Width , Height]
                onehot_masks = torch.nn.functional.one_hot(targets).permute(0,3,1,2).to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criteria(outputs,targets)
                
                
                #calculate metrics
                avg_loss = (avg_loss*index+loss.item())/(index+1)
                pixel_acc = (pixel_acc*index+pixel_accuracy(outputs.detach().cpu(),targets.detach().cpu()))/(index+1)
                dice_coef = (dice_coef*index+dice_metric(outputs.detach().cpu(),onehot_masks.detach().cpu())).item()/(index+1)
                IoU_cell1 = (IoU_cell1*index+IoU(outputs.detach().cpu(),onehot_masks.detach().cpu(),cell=1))/(index+1)
                IoU_cell2 = (IoU_cell2*index+IoU(outputs.detach().cpu(),onehot_masks.detach().cpu(),cell=2))/(index+1)
                IoU_cell3 = (IoU_cell3*index+IoU(outputs.detach().cpu(),onehot_masks.detach().cpu(),cell=3))/(index+1)

                loss = loss.backward()
                optimizer.step()
        scheduler.step()        
                
                
        with torch.no_grad():
            model.eval()
            
            avg_loss=0
            pixel_acc=0
            dice_coef=0
            IoU_cell1=0
            IoU_cell2=0
            IoU_cell3=0
            with tqdm(val_loader) as tdm:
                for index,data_en in enumerate(tdm):
                    tdm.set_description(f'Epoch :{epoch}/{num_epochs}, Lr : {scheduler.get_last_lr()} Training -')
                    tdm.set_postfix(
                        loss=round(avg_loss,3),
                        pixel_acc=round(pixel_acc,3),
                        dice_coef=dice_coef,
                        IoU_cell_123 = (round(IoU_cell1,3),round(IoU_cell2,3),round(IoU_cell3,3))
                        )
                    inputs,masks = data_en
                    inputs = inputs.numpy()
                    masks = masks.numpy()
                    inputs = torch.from_numpy(inputs/255).permute(0,3,1,2).to(device)
                    targets = torch.from_numpy(masks).to(torch.long).to(device)
                    inputs = inputs.type(torch.float)
                    targets = targets.type(torch.long)
                    #onehot encoding with [batchsize, num_classes, Width , Height]
                    onehot_masks = torch.nn.functional.one_hot(targets).permute(0,3,1,2).to(device)
                    
                    
                    outputs = model(inputs)
                    loss = criteria(outputs,targets)
                    
                    
                    #calculate metrics
                    avg_loss = (avg_loss*index+loss.item())/(index+1)
                    pixel_acc = (pixel_acc*index+pixel_accuracy(outputs.detach().cpu(),targets.detach().cpu()))/(index+1)
                    dice_coef = (dice_coef*index+dice_metric(outputs.detach().cpu(),onehot_masks.detach().cpu())).item()/(index+1)
                    IoU_cell1 = (IoU_cell1*index+IoU(outputs.detach().cpu(),onehot_masks.detach().cpu(),cell=1))/(index+1)
                    IoU_cell2 = (IoU_cell2*index+IoU(outputs.detach().cpu(),onehot_masks.detach().cpu(),cell=2))/(index+1)
                    IoU_cell3 = (IoU_cell3*index+IoU(outputs.detach().cpu(),onehot_masks.detach().cpu(),cell=3))/(index+1)
                    
        if avg_loss < best_loss:
            print("saving best model")
            best_loss = avg_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_test=0
        else :
            patience_test +=1
        if patience_test >PATIENCE:
            print("out of patience, breaking the training loop")
            break
            
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
        device=device)
model_path = "/content/nissl_project_cs19m036/trained_models/unet_CEloss_unweighted.pt"

print(f'Training completed , saving model weights to {model_path}')
try:
    torch.save(model.state_dict(), model_path)
    print("model weights saved succesfully")
except:
    print("Incorrect model path , failed to save model weights")


#testing a sample
with torch.no_grad():
  inputs,masks = next(iter(val_loader))
  inputs = inputs.numpy()[0]
  masks = masks.numpy()[0]
  input_org = inputs
  masks_org = masks
  inputs = torch.from_numpy(inputs/255).permute(2,0,1).to(device).unsqueeze(dim=0)
  targets = torch.from_numpy(masks).to(torch.long).to(device).unsqueeze(dim=0)

  inputs = inputs.type(torch.float)
  targets = targets.type(torch.long)
  #onehot encoding with [batchsize, num_classes, Width , Height]
  onehot_masks = torch.nn.functional.one_hot(targets)
  onehot_masks = onehot_masks.permute(0,3,1,2).to(device)
                  
  #optimizer.zero_grad()
  outputs = final_model(inputs)
  loss = CrossEntropy_loss(outputs,targets)
                  
                  
  #calculate metrics
  avg_loss = loss.item()
  pixel_acc = pixel_accuracy(outputs.detach().cpu(),targets.detach().cpu())
  dice_coef = dice_metric(outputs.detach().cpu(),onehot_masks.detach().cpu()).item()
  IoU_cell1 = IoU(outputs.detach().cpu(),onehot_masks.detach().cpu(),cell=1)
  IoU_cell2 = IoU(outputs.detach().cpu(),onehot_masks.detach().cpu(),cell=2)
  IoU_cell3 = IoU(outputs.detach().cpu(),onehot_masks.detach().cpu(),cell=3)
      
  outputs = outputs.detach().squeeze(dim=0)
  outputs = torch.nn.Softmax(dim=0)(outputs)
  output_mask = torch.argmax(outputs, dim=0)
  cv2.imwrite('output.tif',output_mask.cpu().numpy().astype(np.uint8))
  cv2.imwrite('input.png',input_org)
  cv2.imwrite('target.tif',masks_org)

  print("avg_loss = ",avg_loss)
  print("pixel accuracy = ",pixel_acc)
  print("dice_coef = ",dice_coef)
  print("IOU of cell 1,2,3 = ",IoU_cell1,IoU_cell2,IoU_cell3)


                
                
                
                
















