import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.models import mobilenet_v2
from tqdm import tqdm
import torchvision.models as models

"""parse arguments from command line"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=str, default="Deepfakes")
    parser.add_argument('-e', '--epoch', type=int, default=1)
    parser.add_argument('-b', '--batch', type=int, default=32)
    parser.add_argument('-w', '--workers', type=int, default=1) 
    arg = parser.parse_args()
    return arg


class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = []
        self.transform = transform

        # Open the train data paths
        with open(data_path, 'r') as f:
            for line in f:
                record = line.split(" ")
                self.data.append((record[0], int(record[1])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label


def main(args):
    print("Start time =", time.time())

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Load the dataset
    dataset = CustomDataset("dataSetDict_224/" + args.target + "/trainDataSet.txt", transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers)

    # Define the model
    model = mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)  # Assuming 2 classes for binary classification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    startTime = time.time()
    print("Start time =", time.time())

    # Start training
    for epoch in tqdm(range(args.epoch)):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch + 1}/{args.epoch}], Loss: {epoch_loss:.4f}")

        # Save the model
        if not os.path.isdir("models_224/" + args.target):
            os.mkdir('models_224/' + args.target)
        torch.save(model, "models_224/" + args.target + '/epoch' + str(epoch + 1) + '.pth')

    print("Time used =", time.time() - startTime)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)