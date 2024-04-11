import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Define the transformations to be applied to the input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

testDataSet = []

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=str, default="Deepfakes")
    arg = parser.parse_args()
    return arg

def dataGen():
    for data in testDataSet:
        path, label = data
        img = Image.open(path)
        img = transform(img)
        img = img.unsqueeze(0)  # Add batch dimension
        yield img

def dataGen2():
    for data in testDataSet:
        path, label = data
        yield torch.tensor([label], dtype=torch.long)  # Convert label to a tensor with appropriate dimensions

def main(args):
    testDataFile = open("dataSetDict_224/" + args.target + "/testDataSet.txt")
    while True:
        line = testDataFile.readline()
        if not line:
            break
        record = line.split(" ")
        testDataSet.append((record[0], int(record[1])))
    print("Test Data Paths loaded: number =", len(testDataSet))

    print("testing epoch=", 1, '\n')
    model = torch.load("models_224/" + args.target + '/epoch' + str(1) + '.pth')
    model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(zip(dataGen(), dataGen2())):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
