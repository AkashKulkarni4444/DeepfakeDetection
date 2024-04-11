import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import csv

# Define the transformations to be applied to the input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

testDataSet = []

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=str, default="Deepfakes")
    parser.add_argument('-o', '--output', type=str, default="test_results.csv")
    parser.add_argument('-a', '--accuracy_output', type=str, default="accuracy_results.csv")
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
    model = torch.load("models_224_mobilenet/" + args.target + '/epoch' + str(1) + '.pth')
    model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    correct = 0
    total = 0
    with torch.no_grad(), open(args.output, 'w', newline='') as csvfile, open(args.accuracy_output, 'w', newline='') as acc_file:
        fieldnames = ['Image_Path', 'True_Label', 'Predicted_Label', 'Correct_Prediction']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for inputs, labels in tqdm(zip(dataGen(), dataGen2())):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct_prediction = int((predicted == labels).sum().item())

            writer.writerow({
                'Image_Path': testDataSet[total-1][0],
                'True_Label': testDataSet[total-1][1],
                'Predicted_Label': predicted.item(),
                'Correct_Prediction': correct_prediction
            })

            correct += correct_prediction

        accuracy = 100 * correct / total
        test_size = len(testDataSet)

        acc_writer = csv.DictWriter(acc_file, fieldnames=['Model_Path', 'Model_Name', 'Accuracy', 'Test_Size'])
        acc_writer.writeheader()
        acc_writer.writerow({
            'Model_Path': "models_224_mobilenet/" + args.target + '/epoch' + str(1) + '.pth',
            'Model_Name': args.target,
            'Accuracy': accuracy,
            'Test_Size': test_size
        })

    print('Accuracy of the network on the test images: %d %%' % accuracy)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
