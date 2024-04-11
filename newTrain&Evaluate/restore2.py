import os
import random
from tqdm import tqdm

"""This python script will divide the videos into two sets(train & test)"""
"""Paths of pictures from the two sets will be written into file individually"""
"""Picture size used = 224*224*3"""



"""preparing"""
pathsDictionary = {
    'Deepfakes': 'D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/preprocess/picture_224/fake/Deepfakes/',
    'FaceSwap': 'D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/preprocess/picture_224/fake/FaceSwap/',
    'NeuralTextures': 'D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/preprocess/picture_224/fake/NeuralTextures/',
    'Face2Face': 'D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/preprocess/picture_224/fake/Face2Face/',
    'FaceShifter': 'D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/preprocess/picture_224/fake/FaceShifter/',
    'raw': 'D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/preprocess/picture_224/raw/youtube/'
}

def recreate_train_dataset(target='Deepfakes', splitRate=0.8):
    paths = {'raw': 'D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/preprocess/picture_224/raw/youtube/', target: pathsDictionary[target]}
    train = []
    test = []
    
    # Read the paths from testDataSet.txt
    test_paths = []
    with open("D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/newTrain&Evaluate/dataSetDict_224/{}/testDataSet.txt".format(target), 'r') as test_file:
        for line in tqdm(test_file):
            path, label = line.strip().split(',')
            test_paths.append(path)
        print('finished 1st for loop')

    for part in paths:
        partLabel = 0 if (part == 'raw') else 1
        pictureList = os.listdir(paths[part])
        for pictureName in tqdm(pictureList):
            try:
                videoID = int(pictureName.split('-')[0])
                path = os.path.join(paths[part], pictureName)
                if path not in test_paths:  # Exclude paths present in testDataSet.txt
                    train.append((path, partLabel))
            except ValueError as e:
                pass
        random.shuffle(train)
    return train


def recreate_train_dataset_file(target='Deepfakes', splitRate=0.8):
    trainDataSet = recreate_train_dataset(target, splitRate)
    if not os.path.isdir("D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/newTrain&Evaluate/dataSetDict_224/"):
        os.mkdir("D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/newTrain&Evaluate/dataSetDict_224/")
    if not os.path.isdir("D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/newTrain&Evaluate/dataSetDict_224/" + target):
        os.mkdir("D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/newTrain&Evaluate/dataSetDict_224/" + target)
    trainDataFile = open("D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/newTrain&Evaluate/dataSetDict_224/" + target + "/trainDataSet.txt", 'w')
    print(len(trainDataSet))
    for i in trainDataSet:
        pict, label = i
        trainDataFile.write(pict + "," + str(label) + "\n")
    trainDataFile.close()


# Example usage:
recreate_train_dataset_file('NeuralTextures')
