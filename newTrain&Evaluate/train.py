import argparse
import os
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import timm
from tqdm import tqdm

trainDataSet = []

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=str, default="Deepfakes")
    parser.add_argument('-e', '--epoch', type=int, default=1)
    parser.add_argument('-b', '--batch', type=int, default=32)
    parser.add_argument('-w', '--workers', type=int, default=1)
    arg = parser.parse_args()
    return arg

class CustomDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path, label = self.data_list[idx]
        img = Image.open(path)
        if self.transform:
            img = self.transform(img)
        return img, label

# def model_load(target, epochNum):
#     if not os.path.isdir("models/" + target):
#         os.mkdir('models/' + target)
#     if epochNum == 1:
#         model = models.xception(pretrained=True)
#     else:
#         model = torch.load("models/" + target + '/epoch' + str(epochNum - 1) + '.pt')
#     return model


def model_load(target, epochNum):
    model_dir = "models/" + target
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, 'epoch' + str(epochNum - 1) + '.pt')
    if os.path.exists(model_path):
        model = torch.load(model_path)
    else:
        # Load pre-trained Xception model
        model = timm.create_model('xception', pretrained=True)

        # Modify the final fully connected layer to match the number of classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(trainDataSet))  # Modify the last layer
    return model



def model_save(model, target, epochNum):
    if not os.path.isdir("models/" + target):
        os.mkdir('models/' + target)
    torch.save(model, "models/" + target + '/epoch' + str(epochNum) + '.pt')

def main(args):
    print("Start time =  ", time.time())

    trainDataFile = open("dataSetDict/" + args.target + "/trainDataSet.txt")
    while True:
        line = trainDataFile.readline()
        if not line:
            break
        record = line.split(" ")
        trainDataSet.append((record[0], int(record[1])))
    print("Training Data Paths loaded: number = ", len(trainDataSet))

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    dataset = CustomDataset(trainDataSet, transform=transform)
    data_loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers)

    startTime = time.time()
    print("Start time =  ", time.time())

    model = model_load(args.target, args.epoch)
    model.fc = nn.Linear(model.fc.in_features, len(trainDataSet))  # Change last layer to match number of classes
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in tqdm(range(args.epoch)):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataset)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, args.epoch, epoch_loss))

        model_save(model, args.target, args.epoch)

    print("Time used = ", time.time() - startTime)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
