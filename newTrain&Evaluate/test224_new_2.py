import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import csv
import os

# Define the transformations to be applied to the input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

testDataSet = []

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True, help="Name of the model")
    parser.add_argument('-e', '--epochs', type=int, nargs='+', required=True, help="List of epoch numbers")
    parser.add_argument('-t', '--targets', type=str, nargs='+', required=True, help="List of target directories")
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

def test_model(model, device, model_name, epoch, target):
    output_filename = f"{model_name}_{epoch}_{target}_test_results.csv"
    fieldnames = ['Image_Path', 'True_Label', 'Predicted_Label', 'Correct_Prediction']
    existing_image_paths = set()

    if os.path.exists(output_filename):
        # File exists, read existing image paths
        with open(output_filename, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                existing_image_paths.add(row['Image_Path'])

        mode = 'a'
    else:
        mode = 'w'

    with open(output_filename, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if mode == 'w':
            writer.writeheader()  # Write header only if creating new file

        correct = 0
        total = 0
        for inputs, labels in tqdm(zip(dataGen(), dataGen2())):
            image_path = testDataSet[total][0]
            if image_path in existing_image_paths:
                total += 1
                continue

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct_prediction = int((predicted == labels).sum().item())
            writer.writerow({
                'Image_Path': image_path,
                'True_Label': testDataSet[total-1][1],
                'Predicted_Label': predicted.item(),
                'Correct_Prediction': correct_prediction
            })

            correct += correct_prediction

        accuracy = 100 * correct / total

    return accuracy


# new test model
# def test_model(model, device, model_name, epoch, target):
#     output_filename = f"{model_name}_{epoch}_{target}_test_results.csv"
#     with open(output_filename, 'w', newline='') as csvfile:
#         fieldnames = ['Image_Path', 'True_Label', 'Predicted_Label', 'Correct_Prediction']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()

#         correct = 0
#         total = 0
#         for inputs, labels in tqdm(zip(dataGen(), dataGen2())):
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct_prediction = int((predicted == labels).sum().item())

#             writer.writerow({
#                 'Image_Path': testDataSet[total-1][0],
#                 'True_Label': testDataSet[total-1][1],
#                 'Predicted_Label': predicted.item(),
#                 'Correct_Prediction': correct_prediction
#             })

#             correct += correct_prediction

#         accuracy = 100 * correct / total

#     return accuracy

# def test_model(model, device, model_name, epoch, target):
#     output_filename = f"{model_name}_{epoch}_{target}_test_results.csv"
#     existing_lines = 0
#     if os.path.exists(output_filename):
#         with open(output_filename, 'r') as csvfile:
#             existing_lines = sum(1 for line in csvfile)
#         existing_lines-=1 # subtract one to account for the header row

#     with open(output_filename, 'a', newline='') as csvfile:
#         fieldnames = ['Image_Path', 'True_Label', 'Predicted_Label', 'Correct_Prediction']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         if existing_lines == 0:
#             writer.writeheader()

#         correct = 0
#         total = 0
#         for inputs, labels in tqdm(zip(dataGen(), dataGen2())):
#             if total < existing_lines:
#                 total += 1
#                 continue

#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct_prediction = int((predicted == labels).sum().item())

#             writer.writerow({
#                 'Image_Path': testDataSet[total-1][0],
#                 'True_Label': testDataSet[total-1][1],
#                 'Predicted_Label': predicted.item(),
#                 'Correct_Prediction': correct_prediction
#             })

#             correct += correct_prediction

#         accuracy = 100 * correct / total

#     return accuracy

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for target in args.targets:
        testDataFile = open("dataSetDict_224/" + target + "/testDataSet.txt")
        while True:
            line = testDataFile.readline()
            if not line:
                break
            record = line.split(" ")
            testDataSet.append((record[0], int(record[1])))
        print("Test Data Paths loaded for target:", target, "number =", len(testDataSet))
        
        for epoch in tqdm(args.epochs):
            model_path = f"models_224_{args.model_name}/" + target + '/epoch' + str(epoch) + '.pth'
            model = torch.load(model_path)
            model.eval()  # Set the model to evaluation mode
            model.to(device)

            accuracy = test_model(model, device, args.model_name, epoch, target)

            accuracy_filename = f"{args.model_name}_accuracy_results.csv"
            with open(accuracy_filename, 'a', newline='') as acc_file:
                acc_writer = csv.DictWriter(acc_file, fieldnames=['Model_Path', 'Model_Name', 'Target', 'Epoch', 'Accuracy', 'Test_Size'])
                if os.stat(accuracy_filename).st_size == 0:
                    acc_writer.writeheader()
                acc_writer.writerow({
                    'Model_Path': model_path,
                    'Model_Name': args.model_name,
                    'Target': target,
                    'Epoch': epoch,
                    'Accuracy': accuracy,
                    'Test_Size': len(testDataSet)
                })

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
