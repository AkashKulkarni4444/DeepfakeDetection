import os
from os.path import join
import subprocess
import cv2
from PIL import Image
import numpy as np
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str)
    parser.add_argument('-m', '--model', type=str,default='mobilenet')
    parser.add_argument('-s', '--savePictures', type=bool, default=False)
    parser.add_argument('-e', '--epoch', type=str, default=10)
    return parser.parse_args()

class CustomDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.pictures = os.listdir(self.path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.pictures)

    def __getitem__(self, idx):
        picture_path = os.path.join(self.path, self.pictures[idx])
        img = Image.open(picture_path)
        img = self.transform(img)
        return img

def extract_frames(data_path, video_num = 0, folder_name = "raw", method='cv2'):
    """Method to extract frames, either with ffmpeg or opencv. FFmpeg won't
    start from 0 so we would have to rename if we want to keep the filenames
    coherent."""
    # os.makedirs(output_path, exist_ok=True)
    if method == 'ffmpeg':
        output_path = "picture/raw"
        os.makedirs(output_path, exist_ok=True)
        subprocess.check_output(
            'ffmpeg -i {} {}'.format(
                data_path, join(output_path, '%04d.png')),
            shell=True, stderr=subprocess.STDOUT)
    elif method == 'cv2':
        reader = cv2.VideoCapture(data_path)
        count = 0
        frame_num = 0
        while reader.isOpened():
            success, image = reader.read()
            if not success:
                break
            count += 1
            if count % 4 != 0:
                continue
            frame_num = saveFaces(image, video_num, frame_num, folder_name)
            frame_num += 1
        reader.release()
    else:
        raise Exception('Wrong extract frames method: {}'.format(method))

def detectFaces(img):
    """Method to detect the face from the picture"""
    # img = cv2.imread(image_name)
    face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    result = []
    size = 0
    for (x, y, width, height) in faces:
        if width * height > size:
            if len(result) == 0:
                result.append((x, y, x + width, y + height))
                size = width * height
            else:
                result = result[:-1]
                result.append((x, y, x + width, y + height))
                size = width * height
    return result

def saveFaces(img, video_num, frame_num, full_path):
    """Method to crop the detected face from the picture"""
    img_size = 224
    faces = detectFaces(img)
    os.makedirs(full_path, exist_ok=True)

    if faces:
        frame_num += 1

    for (x1, y1, x2, y2) in faces:
        file_name = '{:05d}.png'.format(frame_num)
        video_name = '{:05d}-'.format(video_num)
        file_name = video_name + file_name

        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        img.crop((x1, y1, x2, y2)).resize((img_size, img_size), Image.LANCZOS).save(
            full_path + "/" + file_name)
    return frame_num

def is_video(name):
    """Method to judge whether the type of the video is right or not"""
    return (name[-4:] in ['.mp4'])

def main(args):
    modelsList = {'Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures'}
    video_name = args.path.split('/')[-1].split('.')[0]
    outputPath = "temp/"+video_name
    if not os.path.isdir('temp'):
        os.mkdir('temp')
    if os.path.isdir(outputPath):
        for i in os.listdir(outputPath):
            os.remove(outputPath+'/'+i)
    else:
        os.mkdir(outputPath)
    extract_frames(args.path, 0, outputPath, 'cv2')

    dataset = CustomDataset(outputPath)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for modelName in tqdm(modelsList):
        fakeCount = 0
        rawCount = 0
        if args.model == 'resnet':
            model_path = "models_224_"+args.model+"/" + modelName+ '/epoch' + str(args.epoch) + '.pth'
        elif args.model == 'mobilenet':
            model_path = "models_224_"+args.model+"/" + modelName+ '/epoch' + str(args.epoch) + '.pth'

        model = torch.load(model_path, map_location=device)
        model.eval()

        for batch in dataloader:
            batch = batch.to(device)
            with torch.no_grad():
                result = model(batch)
            predictions = torch.argmax(result, dim=1)
            fakeCount += torch.sum(predictions == 1).item()
            rawCount += torch.sum(predictions == 0).item()

        finalDecision = 1 if fakeCount > rawCount else 0
        print("model name=", modelName, "\t fakeCount=", fakeCount, "\t rawCount=", rawCount, "\t finalDecision=", finalDecision)
        torch.cuda.empty_cache()
        model = None

if __name__ == '__main__':
    args = parse_args()
    print(args.path)
    main(args)
