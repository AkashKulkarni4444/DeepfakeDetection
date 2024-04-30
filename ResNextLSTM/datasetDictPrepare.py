import argparse
import os
import random
from PIL import Image
import numpy as np
"""This python script will divide the videos into two sets(train & test)"""
"""Paths of pictures from the two sets will be written into file individually"""
"""Picture size used = 224*224*3"""



"""preparing"""
pathsDictionary = {
    'Deepfakes': 'D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/preprocess/video_224/fake/Deepfakes/',
    'FaceSwap': 'D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/preprocess/video_224/fake/FaceSwap/',
    'NeuralTextures': 'D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/preprocess/video_224/fake/NeuralTextures/',
    'Face2Face': 'D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/preprocess/video_224/fake/Face2Face/',
    'FaceShifter': 'D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/preprocess/video_224/fake/FaceShifter/',
    'raw': 'D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/preprocess/video_224/raw/youtube/'
}
videoNumber = dict()
for path in pathsDictionary:
    videoNumber[path] = 0

"""create indexes of the two set"""
def splitSet(target, splitRate=0.8):
    videoLists = os.listdir(pathsDictionary[target])
    for videoName in videoLists:
        try:
            videoID = int(videoName.split("_")[0].split('.mp4')[0])
            videoNumber[target] = max(videoID, videoNumber[target])
        except ValueError as e:
            print(videoName)
            pass
        
    trainLength = round(videoNumber[target] * splitRate)
    testLength = videoNumber[target] - trainLength

    uList = list(range(1, videoNumber[target] + 1))
    testSet = random.sample(uList, testLength)
    trainSet = []

    for i in uList:
        if i not in testSet:
            trainSet.append(i)
    
    return {"name": target, "testSet": testSet, "trainSet": trainSet, "Num": videoNumber[target]}


"""generate the picture path of the two sets"""

def generateDataSet(target='Deepfakes',splitRate = 0.8):
    paths = {'raw': 'D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/preprocess/video_224/raw/youtube/', target: pathsDictionary[target]}
    train = []
    test = []
    for part in paths:
        splitRes = splitSet(part, splitRate)
        partLabel = ",REAL" if (part == 'raw') else ",FAKE"
        videoList = os.listdir(paths[part])
        for videoName in videoList:
            try:
                videoID = int(videoName.split('_')[0].split('.mp4')[0])
                if videoID in splitRes["trainSet"]:
                    train.append((paths[part] + videoName, partLabel))
                elif videoID in splitRes["testSet"]:
                    test.append((paths[part] + videoName, partLabel))
            except ValueError as e:
                pass
        random.shuffle(train)
    return train, test


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=str, default="Deepfakes")
    parser.add_argument('-r', '--rate', type=int, default=0.8)
    arg = parser.parse_args()
    return arg



"""write the paths to local file"""
if __name__ == "__main__":
    args = parse_args()
    trainDataSet, testDataSet = generateDataSet(args.type,args.rate)
    if not os.path.isdir("datasetDict_video_224/"):
        os.mkdir("datasetDict_video_224/")
    if not os.path.isdir("datasetDict_video_224/"+args.type):
        os.mkdir("datasetDict_video_224/"+args.type)
    trainDataFile = open("datasetDict_video_224/"+args.type+"/trainDataSet.txt", 'w')
    testDataFile = open("datasetDict_video_224/"+args.type+"/testDataSet.txt", 'w')
    print(len(trainDataSet), len(testDataSet))
    for i in trainDataSet:
        vid,label = i
        trainDataFile.write(vid + "" + str(label) + "\n")
    for i in testDataSet:
        vid,label = i
        testDataFile.write(vid + "" + str(label) + "\n")