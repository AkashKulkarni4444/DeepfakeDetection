import torch
import pandas as pd
import glob
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
import json
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(torch.mean(x_lstm, dim=1)))


class video_dataset(Dataset):
    def __init__(self, video_names, labels, sequence_length=60, transform=None):
        self.video_names = video_names
        self.labels = labels
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100/self.count)
        first_frame = np.random.randint(0, a)
        temp_video = video_path[0]
        label = self.labels.iloc[(self.labels.loc[self.labels["file"] == temp_video].index.values[0]), 1]
        if label == 'FAKE':
            label = 0
        if label == 'REAL':
            label = 1
        for i, frame in enumerate(self.frame_extract(video_path)):
            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames, label

    def frame_extract(self, path):
        path=path[0]
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image


def im_plot(tensor):
    image = tensor.cpu().numpy().transpose(1, 2, 0)
    b, g, r = cv2.split(image)
    image = cv2.merge((r, g, b))
    image = image * [0.22803, 0.22145, 0.216989] + [0.43216, 0.394666, 0.37645]
    image = image * 255.0
    plt.imshow(image.astype(int))
    plt.show()


def number_of_real_and_fake_videos(data_list,args):
    header_list = ["file", "label"]
    # train_labels = pd.read_csv('D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/ResNextLSTM/datasetDict_video_224/fake/'+args.target+'/trainDataSet.csv',names=header_list)
    test_labels = pd.read_csv('D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/ResNextLSTM/datasetDict_video_224/fake/'+args.target+'/testDataSet.csv',names=header_list)
    # lab = pd.concat([train_labels, test_labels], ignore_index=True)
    lab = test_labels
    fake = 0
    real = 0
    for i in data_list:
        temp_video = i[0]
        label = lab.iloc[(lab.loc[lab["file"] == temp_video].index.values[0]), 1]
        if (label == 'FAKE'):
            fake += 1
        if (label == 'REAL'):
            real += 1
    return real, fake


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()
    return 100 * n_correct_elems / batch_size


def evaluate_model(model, test_loader, target_name, epoch):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            _, outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    results = {
        'Model_name': 'Resnext+LSTM',
        'Epoch': epoch,
        'Target': target_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1_score,
        'True_Positive': tp,
        'True_Negative': tn,
        'False_Positive': fp,
        'False_Negative': fn
    }

    return results

def main(args):
    # Load and preprocess data
    header_list = ["file", "label"]
    # train_labels = pd.read_csv('D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/ResNextLSTM/datasetDict_video_224/fake/'+args.target+'/trainDataSet.csv',names=header_list)
    test_labels = pd.read_csv('D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/ResNextLSTM/datasetDict_video_224/fake/'+args.target+'/testDataSet.csv',names=header_list)
    # labels = pd.concat([train_labels, test_labels], ignore_index=True)
    labels=test_labels
    # train_videos=train_labels.to_numpy().tolist()
    valid_videos=test_labels.to_numpy().tolist()
    print("Test samples:", len(valid_videos))
    
    # Calculate number of real and fake videos in train and test sets
    print("TEST: Real - {}, Fake - {}".format(*number_of_real_and_fake_videos(valid_videos, args)))

    # Define image transformations
    im_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = f"models_224/" + args.target + '/epoch' + str(args.epoch) + '.pth'
    print(model_path)
    model = torch.load(model_path)
    # model.eval()  # Set the model to evaluation mode
    # model.to(device)
    num_epochs = args.epoch
    
    # Testing phase
    test_data = video_dataset(valid_videos, labels, sequence_length=4, transform=test_transforms)
    test_loader = DataLoader(test_data, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model.eval()
    
    test_results = evaluate_model(model, test_loader, args.target, num_epochs)

    # Save evaluation results to CSV
    results_df = pd.DataFrame([test_results])
    results_file = 'evaluationresults.csv'

    if not os.path.isfile(results_file):
        results_df.to_csv(results_file, index=False)
    else:
        results_df.to_csv(results_file, mode='a', header=False, index=False)

    print("\nEvaluation results saved to", results_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=str, default="NeuralTextures")
    parser.add_argument('-e', '--epoch', type=int, default=10)
    parser.add_argument('-b', '--batch', type=int, default=4)
    parser.add_argument('-w', '--workers', type=int, default=4)
    arg = parser.parse_args()
    return arg


if __name__ == "__main__":
    args = parse_args()
    main(args)
