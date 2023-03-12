#!/usr/bin/env python

# Code referenced from the following with slight adjustments for our H&E dataset:
# License: BSD
# Author: Sasank Chilamkurthy
# Source URL: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import re
import argparse

def get_args():
    '''Argparse: Retrieves user-input arguments from command line.'''
    parser = argparse.ArgumentParser(description="A program to classify H&E breast cancer images using a transfer-learning ResNet-18 model. Input: training and validation tiles root directory (-d), test tiles directory (-t), number of epochs (-e), batch size (-b).")

    parser.add_argument("-d","--dirtrain",help="Root directory containing 'train' and 'val' subfolders, with each containing 5 subtype class folders of tile '.png' images",type=str)
    parser.add_argument("-t","--testdir",help="Directory containing test tile '.png' images, organized into 5 subtype folders.",type=str)
    parser.add_argument("-e","--epochnum",help="Number of training/validation epochs to run. Default = 10.",default=10,type=int)
    parser.add_argument("-b","--batchsizenum",help="Number of training images in batch size (images to process as a batch in one training cycle). Default = 50.",default=50,type=int)

    return parser.parse_args()

cudnn.benchmark = True
plt.ion() 

#Retrieve argparse input
args = get_args()
input_traindir = args.dirtrain
input_testdir = args.testdir
input_epochs = args.epochnum
input_batchsize = args.batchsizenum

# Data augmentation and normalization for training
# Only normalization for validation
data_transforms = {
    'train': transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = input_traindir #'../PAM50_set/full_train/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=input_batchsize,
                                             shuffle=True, num_workers=2)
                for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    loss_acc_train_dict = {}
    loss_acc_val_dict = {}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                loss_acc_train_dict[epoch] = (epoch_loss,epoch_acc.item())
            else:
                loss_acc_val_dict[epoch] = (epoch_loss,epoch_acc.item())
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,loss_acc_train_dict,loss_acc_val_dict

def visualize_model(model, num_images=2):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            #print("Preds: ",preds)
            # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)*100
            #print("Probabilities: ",probabilities)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                print(f'Validation prediction: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def plot_Val_Summaries(train_dict,val_dict):
    '''Uses train and val dictionaries of loss or accuracy data from ML run to create plots of # epochs vs. loss/accuracy'''
    x_all = []
    y_acc_train = []
    y_acc_val = []
    y_loss_train = []
    y_loss_val = []
    # print("Train dict: ",train_dict)
    # print(type(train_dict))
    # print("Val dict: ",val_dict)
    # print(type(val_dict))
    for key in train_dict:
        x_all.append(key)
        y_loss_train.append(train_dict[key][0])
        y_acc_train.append(train_dict[key][1])
        y_loss_val.append(val_dict[key][0])
        y_acc_val.append(val_dict[key][1])
    figure,axis = plt.subplots(1,2)
    figure.text(0.5, 0.04, 'Epochs', ha='center')
    figure.text(0.01, 0.5, 'Percentage', va='center', rotation='vertical')
    axis[0].plot(x_all,y_loss_train,label="Training Loss")
    axis[0].plot(x_all,y_loss_val,label="Validation Loss")
    axis[0].set_title("Model Loss")
    axis[0].legend()
    axis[1].plot(x_all,y_acc_train,label="Training Accuracy")
    axis[1].plot(x_all,y_acc_val,label="Validation Accuracy")
    axis[1].set_title("Model Accuracy")
    axis[1].legend()
    plt.savefig("Resnetv5_TrainVal_Summary_PAM50_e" + str(input_epochs) + "_b" + str(input_batchsize) + ".png")
    print("Summary graphs successfully created!")
    return None

def plot_Test_Summaries(res_Basal_dict,res_HER2E_dict,res_LumA_dict,res_LumB_dict,res_nl_dict):
    '''Creates plots to display patterns in subtype data using dictionary input'''
    c_vals = [res_Basal_dict["First Prediction Correct"],res_HER2E_dict["First Prediction Correct"],res_LumA_dict["First Prediction Correct"],res_LumB_dict["First Prediction Correct"],res_nl_dict["First Prediction Correct"]]

    ac_vals = [(res_Basal_dict["First Prediction Correct"]+res_Basal_dict["Top 3 Prediction Correct"]),(res_HER2E_dict["First Prediction Correct"]+res_HER2E_dict["Top 3 Prediction Correct"]),(res_LumA_dict["First Prediction Correct"]+res_LumA_dict["Top 3 Prediction Correct"]),(res_LumB_dict["First Prediction Correct"]+res_LumB_dict["Top 3 Prediction Correct"]),(res_nl_dict["First Prediction Correct"]+res_nl_dict["Top 3 Prediction Correct"])]
    
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))
    br1 = np.arange(len(c_vals))
    br2 = [x + barWidth for x in br1]
    plt.bar(br1,c_vals, color ='#00008b', width = barWidth, edgecolor ='black', label ='First Prediction Correct')
    plt.bar(br2,ac_vals, color ='#8d8de5', width = barWidth, edgecolor ='black', label ='Top 3 Prediction Correct')
    #plt.xlabel("Breast Cancer Subtype",fontweight ='bold', fontsize = 15)
    plt.ylabel("Number of Tile Images",fontweight ='bold', fontsize = 15)
    plt.title("ResNet-18 H&E Model PAM50 Test Set Classification Result",fontweight ='bold', fontsize = 18)
    plt.xticks([r + barWidth*0.5 for r in range(len(c_vals))],
        ['Basal', 'HER2E','LumA','LumB','Normal-like'],fontsize = 12,rotation = 25)
    plt.yticks(fontsize = 15)
    plt.ylim(0,(res_Basal_dict["First Prediction Correct"]+res_Basal_dict["Top 3 Prediction Correct"]+res_Basal_dict["Incorrect"])) #Set upper limit to expected maximum y value for BALANCED test set
    plt.legend(fontsize = 15)
    plt.savefig("Resnetv5_Test_Summary_PAM50_e" + str(input_epochs) + "_b" + str(input_batchsize) + ".png")

def run_Test(resnet_model,labels,img_dir,actual_st):
    '''Run test set (in separate directory) through pre-trained resnet model'''

    from PIL import Image
    from torchvision import transforms

    #Initialize acc/loss dict per image; key=img, value=(actual_st,prediction_label,prediction_percent)
    test_dict = {}
    #Go through each image in directory
    i = 0
    for img in os.listdir(img_dir): 
        i += 1
        img_path = img_dir+img
        img_test = Image.open(img_path).convert('RGB')

        # Create a preprocessing pipeline
        preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
        # Pass the image for preprocessing and the image preprocessed
        img_test_preprocessed = preprocess(img_test)
        # Reshape, crop, and normalize the input tensor for feeding into network for evaluation
        batch_img_test_tensor = torch.unsqueeze(img_test_preprocessed, 0)
        #Set passed in pre-trained model to eval mode
        resnet_model.eval()
        batch_img_test_tensor = batch_img_test_tensor.to(device)
        #Get model predictions of passed in image in eval mode
        out = resnet_model(batch_img_test_tensor)
        ## Find the index (tensor) corresponding to the maximum score in the out tensor.
        # Torch.max function can be used to find the information
        #_, index = torch.max(out, 1)
        # Get top 5 scores along with the image label. Sort function is invoked on the torch to sort the scores.
        _, indices = torch.sort(out, descending=True)
        # Find the score in terms of percentage by using torch.nn.functional.softmax function
        # which normalizes the output to range [0,1] and multiplying by 100
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        # Create dict of first two predictions (per tile) - (key,value) => (rank,(label,acc pred))
        pred_dict = {}
        r = 0 # prediction "rank" variable
        for idx in indices[0][:3]:
            r += 1
            pred_label = labels[idx]
            pred_perc = percentage[idx].item()
            pred_dict[r] = (pred_label,pred_perc)
        # Assign predictions to variables for test_dict
        first_label = pred_dict[1][0]
        first_perc = pred_dict[1][1]
        second_label = pred_dict[2][0]
        second_perc = pred_dict[2][1]
        third_label = pred_dict[3][0]
        third_perc = pred_dict[3][1]
        if i <= 2: #FOR TESTING - avoids all of test set being printed to stdout
            print("Test IMG",i,"- Actual Subtype: ",actual_st)
            print("Test IMG",i,"- Best Prediction: ",first_label,first_perc)
            # Print the top 5 scores along with the image label. Sort function is invoked on the torch to sort the scores.
            #_, indices = torch.sort(out, descending=True)
            print("Test IMG",i,"- Top 3 Predictions: ")
            for guess in pred_dict:
                print(pred_dict[guess])

        #Add current img stats to dict
        test_dict[img] = (actual_st,first_label,first_perc,second_label,second_perc,third_label,third_perc)

    #Determine model statistics
    result_dict = {"First Prediction Correct":0, "Top 3 Prediction Correct":0,"Incorrect":0}
    count = 0
    for key in test_dict:
        count += 1
        if test_dict[key][0] == test_dict[key][1]:
            result_dict["First Prediction Correct"] = result_dict["First Prediction Correct"] + 1
        elif (test_dict[key][0] == test_dict[key][3]) | (test_dict[key][0] == test_dict[key][5]):
            result_dict["Top 3 Prediction Correct"] = result_dict["Top 3 Prediction Correct"] + 1
        else:
            result_dict["Incorrect"] = result_dict["Incorrect"] + 1
    overall_test_acc = ((result_dict["First Prediction Correct"])/len(test_dict)) * 100
    overall_test_acc2 = ((result_dict["First Prediction Correct"]+result_dict["Top 3 Prediction Correct"])/len(test_dict)) * 100
    print("Subtype First Prediction Accuracy: ",result_dict["First Prediction Correct"],"/",len(test_dict)," = ",overall_test_acc)
    print("Subtype Top 3 Prediction Accuracy: ",(result_dict["First Prediction Correct"]+result_dict["Top 3 Prediction Correct"]),"/",len(test_dict)," = ",overall_test_acc2,"\n")
    return result_dict

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#Train model with above and input-specified parameters
model_ft,train_sum_dict,val_sum_dict = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=input_epochs)

#Print out statistics for R graphing (added to make GIA poster figures)
# print("Train Summary: ",train_sum_dict)
# print("Val Summary: ",val_sum_dict)

#Validation Acc/Loss Stats
plot_Val_Summaries(train_sum_dict,val_sum_dict)

visualize_model(model_ft)

#Running test sets:
#IDC Basal
dir_test_Basal = input_testdir + "Basal/" #"/projects/bgmp/shared/groups/2022/z7t/goecks/PAM50_set/test/Basal/"
print("Basal TEST SET:")
dict_Basal_res = run_Test(model_ft,class_names,dir_test_Basal,"Basal")

#IDC HER2E
dir_test_HER2E = input_testdir + "HER2E/" #"/projects/bgmp/shared/groups/2022/z7t/goecks/PAM50_set/test/HER2E/"
print("HER2E TEST SET:")
dict_HER2E_res = run_Test(model_ft,class_names,dir_test_HER2E,"HER2E")

#IDC LumA
dir_test_LumA = input_testdir + "LumA/" #"/projects/bgmp/shared/groups/2022/z7t/goecks/PAM50_set/test/LumA/"
print("LumA TEST SET:")
dict_LumA_res = run_Test(model_ft,class_names,dir_test_LumA,"LumA")

#IDC LumB
dir_test_LumB = input_testdir + "LumB/" #"/projects/bgmp/shared/groups/2022/z7t/goecks/PAM50_set/test/LumB/"
print("LumB TEST SET:")
dict_LumB_res = run_Test(model_ft,class_names,dir_test_LumB,"LumB")

#IDC normal-like
dir_test_nl = input_testdir + "normal-like/" #"/projects/bgmp/shared/groups/2022/z7t/goecks/PAM50_set/test/normal-like/"
print("Normal-like TEST SET:")
dict_nl_res = run_Test(model_ft,class_names,dir_test_nl,"normal-like")

#Plot overall test set results - prediction accuracies by subtype
plot_Test_Summaries(dict_Basal_res,dict_HER2E_res,dict_LumA_res,dict_LumB_res,dict_nl_res)
print("Test Summary Graph Successfully Created!")