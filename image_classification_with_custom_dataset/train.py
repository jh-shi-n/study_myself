import argparse
import copy
import os
import time
import numpy as np
import random
import torch
import wandb

import torch.optim as optim
import torch.nn.functional as F

from tracemalloc import start
from torch import nn
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data_loader import fake_dataset
from tqdm import tqdm

# Seed fix
random_seed = 2022
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


# Model training Code
def train_model(model, criterion, optimizer, scheduler, num_epochs, save_path):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, _ in tqdm(dataloaders[phase]):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                target = F.one_hot(labels, 2)
                target = target.type('torch.FloatTensor').to(device)
                
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    #loss = criterion(outputs, labels)
                    loss = criterion(outputs, target)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # zero the parameter gradients
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                #print(running_loss)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()
            # print("here", dataloaders[phase] * dataloaders[phase].batch_size)

            epoch_loss = running_loss / \
                (dataset_sizes[phase])  # 확인할 것
            epoch_acc = running_corrects.double(
            ) / (dataset_sizes[phase] * dataloaders[phase].batch_size)
            
            if phase == 'train' : 
                wandb.log({'train_epoch_loss' : epoch_loss,
                            'train_epoch_accuracy' : epoch_acc}, step = epoch)

            if phase == 'valid' : 
                wandb.log({'valid_epoch_loss' : epoch_loss,
                            'valid_epoch_accuracy' : epoch_acc}, step = epoch)

            

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        torch.save(model, os.path.join(
            save_path, "model_epoch_{}.pth".format(epoch)))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_id', type=str, default = "shinhoya")
    parser.add_argument('--lr', type=float, default = 0.0001)
    parser.add_argument('--epochs', type=int, default = 100)

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    # parameter setting as user's set(or default)
    opt = parse_opt()

    # wandb.ai login and init
    wandb.init(project='resnet-18', entity=opt.wandb_id)
    
    # Set transform
    train_valid_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomAffine(45, shear=20),
        transforms.RandomCrop(200),
        transforms.ToTensor()])

    valid_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()])


    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()])

    # Create the dataset (Custom Dataset)
    trainset = fake_dataset(path='./output_V2_1000/train/',
                            transform=train_valid_transform)
    validset = fake_dataset(path='./output_V2_1000/val/',
                            transform=train_valid_transform)
    testset = fake_dataset(path='./output_V2_1000/test/',
                           transform=test_transform)


    # Dataloader 준비 및 데이터셋 사이즈 확인
    dataloaders = {'train': DataLoader(trainset, batch_size=4, shuffle=True),
                   'valid': DataLoader(validset, batch_size=2, shuffle=True),
                   'test': DataLoader(testset, batch_size=1, shuffle=False)}

    dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'valid', 'test']}

    print("Current Dataset : {}".format(dataset_sizes))

    classes = ['fake', 'real']

    # Set default device as gpu, if available
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
    #Parameter Setting
    num_epochs = opt.epochs
    num_classes = len(classes)
    learning_rate = opt.lr

    # Create save directory
    start_time = '{}_{}'.format(time.strftime('%Y%m%d'), time.strftime('%H%M%S'))
    savepath = os.path.join(".", "runs", start_time)
    os.makedirs(savepath, exist_ok=True)
    
    # Load a pretrained model - Resnet18
    print("\nLoading resnet18 for finetuning ...\n")
    model_ft = models.resnet18(pretrained=False)
    
    # Modify fc layers to match num_classes
    model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=2, bias=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
    print(model_ft)
    model_ft = model_ft.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)  # CHN
    
    # Optimizer
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)
    
    # Learning rate decay
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)

    # wandb config setting
    wandb.config = {
    "learning_rate": learning_rate,
    "epochs": num_epochs,
    "batch_size": 4 # train dataset batch_size
    }
    
    # Train the model
    model = train_model(model_ft, criterion, optimizer_ft,
                        exp_lr_scheduler, num_epochs=num_epochs, save_path=savepath)
    
    # Save the entire model
    print("\nSaving the model...")
    torch.save(model, os.path.join(savepath, "model_best.pth"))  # Path
    