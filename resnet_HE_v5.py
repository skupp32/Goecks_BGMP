#!/usr/bin/env python

# Code derived from:
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

cudnn.benchmark = True
plt.ion() 

# Data augmentation and normalization for training
# Just normalization for validation
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

data_dir = '../balanced_data/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=50,
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

            # # Show top categories per image - NOT WORKING
            # top2_prob, top2_catid = torch.topk(probabilities, 2)
            # for p in range(top2_prob.size(0)):
            #     print(class_names[top2_catid[p]], top2_prob[p].item())

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
    plt.savefig("Resnet_Val_Summary_v5_10STB_1500TpST_50B_Btest.png")
    print("Summary graphs successfully created!")
    return None

def plot_Test_Summaries(res_IDC_Basal_dict,res_IDC_HER2E_dict,res_IDC_LumA_dict,res_IDC_LumB_dict,res_IDC_nl_dict,res_ILC_Basal_dict,res_ILC_HER2E_dict,res_ILC_LumA_dict,res_ILC_LumB_dict,res_ILC_nl_dict):
    '''Creates plots to display patterns in subtype data using dictionary input'''
    c_vals = [res_IDC_Basal_dict["First Prediction Correct"],res_IDC_HER2E_dict["First Prediction Correct"],res_IDC_LumA_dict["First Prediction Correct"],res_IDC_LumB_dict["First Prediction Correct"],res_IDC_nl_dict["First Prediction Correct"],res_ILC_Basal_dict["First Prediction Correct"],res_ILC_HER2E_dict["First Prediction Correct"],res_ILC_LumA_dict["First Prediction Correct"],res_ILC_LumB_dict["First Prediction Correct"],res_ILC_nl_dict["First Prediction Correct"]]

    ac_vals = [(res_IDC_Basal_dict["First Prediction Correct"]+res_IDC_Basal_dict["Top 3 Prediction Correct"]),(res_IDC_HER2E_dict["First Prediction Correct"]+res_IDC_HER2E_dict["Top 3 Prediction Correct"]),(res_IDC_LumA_dict["First Prediction Correct"]+res_IDC_LumA_dict["Top 3 Prediction Correct"]),(res_IDC_LumB_dict["First Prediction Correct"]+res_IDC_LumB_dict["Top 3 Prediction Correct"]),(res_IDC_nl_dict["First Prediction Correct"]+res_IDC_nl_dict["Top 3 Prediction Correct"]),(res_ILC_Basal_dict["First Prediction Correct"]+res_ILC_Basal_dict["Top 3 Prediction Correct"]),(res_ILC_HER2E_dict["First Prediction Correct"]+res_ILC_HER2E_dict["Top 3 Prediction Correct"]),(res_ILC_LumA_dict["First Prediction Correct"]+res_ILC_LumA_dict["Top 3 Prediction Correct"]),(res_ILC_LumB_dict["First Prediction Correct"]+res_ILC_LumB_dict["Top 3 Prediction Correct"]),(res_ILC_nl_dict["First Prediction Correct"]+res_ILC_nl_dict["Top 3 Prediction Correct"])]
    
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))
    br1 = np.arange(len(c_vals))
    br2 = [x + barWidth for x in br1]
    plt.bar(br1,c_vals, color ='#00008b', width = barWidth, edgecolor ='black', label ='First Prediction Correct')
    plt.bar(br2,ac_vals, color ='#8d8de5', width = barWidth, edgecolor ='black', label ='Top 3 Prediction Correct')
    #plt.xlabel("Breast Cancer Subtype",fontweight ='bold', fontsize = 15)
    plt.ylabel("Number of Tile Images",fontweight ='bold', fontsize = 15)
    plt.title("ResNet-18 H&E Model Test Set Classification Result",fontweight ='bold', fontsize = 18)
    plt.xticks([r + barWidth*0.5 for r in range(len(c_vals))],
        ['IDC Basal', 'IDC HER2E','IDC LumA','IDC LumB','IDC Normal-like','ILC Basal', 'ILC HER2E','ILC LumA','ILC LumB','ILC Normal-like'],fontsize = 12,rotation = 25)
    plt.yticks(fontsize = 15)
    plt.ylim(0,600)
    plt.legend(fontsize = 15)
    plt.savefig("Resnet_Test_Summary_v5_10STB_1500TpST_50BS_Btest.png")

def run_Test(resnet_model,labels,img_dir):
    '''Run test set (in separate directory) through pre-trained resnet model'''

    from PIL import Image
    from torchvision import transforms

    #Initialize acc/loss dict per image; key=img, value=(actual_label,prediction_label,prediction_percent)
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
        actual_label = img_dir.split("/")[9]
        if i <= 2: #FOR TESTING
            print("Test IMG",i,"- Actual Subtype: ",actual_label)
            print("Test IMG",i,"- Best Prediction: ",pred_label,pred_perc)
            # Print the top 5 scores along with the image label. Sort function is invoked on the torch to sort the scores.
            #_, indices = torch.sort(out, descending=True)
            print("Test IMG",i,"- Top 3 Predictions: ")
            for guess in pred_dict:
                print(pred_dict[guess])
        
        #Print actual image name: - UNCOMMENT ONCE DIRECTORY IS MADE for prettier format?
        # img_name = img.split("/")[-1]
        # actual_hist = img_name.split("_")[2]
        # if actual_hist == "ductal":
        #     actual_hist = "IDC"
        # else:
        #     actual_hist = "ILC"
        # actual_pam50 = img_name.split("_")[4].upper()
        # actual_subtype = actual_hist+"_"+actual_pam50
        # print("Actual Subtype: ",actual_subtype)

        #Add current img stats to dict
        test_dict[img] = (actual_label,first_label,first_perc,second_label,second_perc,third_label,third_perc)

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

model_ft,train_sum_dict,val_sum_dict = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10)

print("Train Summary: ",train_sum_dict)
print("Val Summary: ",val_sum_dict)

#Validation Acc/Loss Stats
plot_Val_Summaries(train_sum_dict,val_sum_dict)

visualize_model(model_ft)

#Running test sets:
#IDC Basal
dir_test_IDC_Basal = "/projects/bgmp/shared/groups/2022/z7t/goecks/balanced_data_test/Invasive_ductal_carcinoma_Basal/"
print("IDC Basal TEST SET:")
dict_IDC_Basal_res = run_Test(model_ft,class_names,dir_test_IDC_Basal)

#IDC HER2E
dir_test_IDC_HER2E = "/projects/bgmp/shared/groups/2022/z7t/goecks/balanced_data_test/Invasive_ductal_carcinoma_HER2E/"
print("IDC HER2E TEST SET:")
dict_IDC_HER2E_res = run_Test(model_ft,class_names,dir_test_IDC_HER2E)

#IDC LumA
dir_test_IDC_LumA = "/projects/bgmp/shared/groups/2022/z7t/goecks/balanced_data_test/Invasive_ductal_carcinoma_LumA/"
print("IDC LumA TEST SET:")
dict_IDC_LumA_res = run_Test(model_ft,class_names,dir_test_IDC_LumA)

#IDC LumB
dir_test_IDC_LumB = "/projects/bgmp/shared/groups/2022/z7t/goecks/balanced_data_test/Invasive_ductal_carcinoma_LumB/"
print("IDC LumB TEST SET:")
dict_IDC_LumB_res = run_Test(model_ft,class_names,dir_test_IDC_LumB)

#IDC normal-like
dir_test_IDC_nl = "/projects/bgmp/shared/groups/2022/z7t/goecks/balanced_data_test/Invasive_ductal_carcinoma_normal-like/"
print("IDC Normal-like TEST SET:")
dict_IDC_nl_res = run_Test(model_ft,class_names,dir_test_IDC_nl)

#ILC Basal
dir_test_ILC_Basal = "/projects/bgmp/shared/groups/2022/z7t/goecks/balanced_data_test/Invasive_lobular_carcinoma_Basal/"
print("ILC Basal TEST SET:")
dict_ILC_Basal_res = run_Test(model_ft,class_names,dir_test_ILC_Basal)

#ILC HER2E
dir_test_ILC_HER2E = "/projects/bgmp/shared/groups/2022/z7t/goecks/balanced_data_test/Invasive_lobular_carcinoma_HER2E/"
print("ILC HER2E TEST SET:")
dict_ILC_HER2E_res = run_Test(model_ft,class_names,dir_test_ILC_HER2E)

#ILC LumA
dir_test_ILC_LumA = "/projects/bgmp/shared/groups/2022/z7t/goecks/balanced_data_test/Invasive_lobular_carcinoma_LumA/"
print("ILC LumA TEST SET:")
dict_ILC_LumA_res = run_Test(model_ft,class_names,dir_test_ILC_LumA)

#ILC LumB
dir_test_ILC_LumB = "/projects/bgmp/shared/groups/2022/z7t/goecks/balanced_data_test/Invasive_lobular_carcinoma_LumB/"
print("ILC LumB TEST SET:")
dict_ILC_LumB_res = run_Test(model_ft,class_names,dir_test_ILC_LumB)

#ILC normal-like
dir_test_ILC_nl = "/projects/bgmp/shared/groups/2022/z7t/goecks/balanced_data_test/Invasive_lobular_carcinoma_normal-like/"
print("ILC Normal-like TEST SET:")
dict_ILC_nl_res = run_Test(model_ft,class_names,dir_test_ILC_nl)

#Plot overall test set results - prediction accuracies by subtype
plot_Test_Summaries(dict_IDC_Basal_res,dict_IDC_HER2E_res,dict_IDC_LumA_res,dict_IDC_LumB_res,dict_IDC_nl_res,dict_ILC_Basal_res,dict_ILC_HER2E_res,dict_ILC_LumA_res,dict_ILC_LumB_res,dict_ILC_nl_res)
print("Test Summary Graph Successfully Created!")