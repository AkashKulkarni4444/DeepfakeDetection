import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.models import resnet152
from torchvision.models.resnet import ResNet152_Weights
from tqdm import tqdm

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
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Load the dataset
    dataset = CustomDataset("dataSetDict_224/" + args.target + "/trainDataSet.txt", transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    # Check for existing saved models
    model_files = [f for f in os.listdir("models_224/" + args.target) if f.endswith('.pth')]
    mx = 0
    if model_files:
        # get model with most epochs
        for mf in model_files:
            temp = int (mf.split('epoch')[1].split('.pth')[0])
            if temp >= mx:
              mx=temp
        # maximum epoch defined in mx
        latest_model_file = 'epoch'+str(mx)+'.pth'
        model = torch.load(os.path.join("models_224/" + args.target, latest_model_file))
        print("Loaded latest model:", latest_model_file)
    else:
        # Define and load ResNet-152 if no saved models found
        model = resnet152(weights=ResNet152_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2) 
        print("No saved models found, loading ResNet-152.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    

    startTime = time.time()
    print("Start time =", time.time())


    # Start training
    for epoch in tqdm(range(mx,args.epoch)):
        model.train()
        # running_loss = 0.0
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device) 
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)     
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        #     running_loss += loss.item() * inputs.size(0)

        # epoch_loss = running_loss / len(dataset)
        # print(f"Epoch [{epoch + 1}/{args.epoch}], Loss: {epoch_loss:.4f}")

        # Save the model
        if not os.path.isdir("models_224/" + args.target):
            os.mkdir('models_224/' + args.target)
        torch.save(model, "models_224/" + args.target + '/epoch' + str(epoch + 1) + '.pth')
        
    print("Time used =", time.time() - startTime)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
