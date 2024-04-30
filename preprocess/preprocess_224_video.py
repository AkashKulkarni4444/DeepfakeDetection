import os
from os.path import join
import subprocess
import cv2
from PIL import Image
import sys
import os
import glob
import face_recognition
from tqdm import tqdm

def frame_extract(path):
    """Extract the frame from a video file."""
    vidObj = cv2.VideoCapture(path)
    success = True
    while success:
        success, image = vidObj.read()
        if success:
            yield image

# Function to create face videos
def create_face_videos(path_list, out_dir):
    """Creates face videos"""
    already_present_count = len(glob.glob(os.path.join(out_dir, '*.mp4')))
    print("Number of videos already present:", already_present_count)
    
    for path in tqdm(path_list):
        out_path = os.path.join(out_dir, os.path.basename(path))
        out_path = out_path.split('.')[0] + ".mp4"  
        if os.path.exists(out_path):
            print("File already exists:", out_path)
            continue
        
        out = None
        frames_processed = 0
        
        for frame in frame_extract(path):
            if frames_processed > 150:  
                break
                
            face_locations = face_recognition.face_locations(frame)
            if face_locations:  # If faces are detected
                top, right, bottom, left = face_locations[0]
                face_image = frame[top:bottom, left:right]
                face_image_resized = cv2.resize(face_image, (224, 224))
                
                if out is None:
                    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (224, 224))
                
                out.write(face_image_resized)
                
            frames_processed += 1
            
        if out is not None:
            out.release()

# def is_video(name):
#     """Method to judge whether the type of the video is right or not"""
#     return (name[-4:] in ['.mp4'])

if __name__ == '__main__':

    # add the terminal input into the arguments list
    arguments = sys.argv
    type = arguments[1]
    folder = arguments[2]
    directory = "video_224/" + type + "/" + folder

    # executue according to the type
    if (type == "raw"):
        # operate the raw video into faces
        video_files = glob.glob("D:/IIIT/IIIT_Shri_City/BTP/dataset/original_sequences/" + folder + "/c23/videos/*.mp4")
        frame_count = []
        for video_file in video_files:
          cap = cv2.VideoCapture(video_file)
          if(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))<150):
            video_files.remove(video_file)
            continue
          frame_count.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        create_face_videos(video_files,'D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/preprocess/video_224/'+type+'/'+folder)

    if (type == "fake"):
        # operate the fake video into faces
        video_files = glob.glob("D:/IIIT/IIIT_Shri_City/BTP/dataset/manipulated_sequences/" + folder + "/c23/videos/*.mp4")
        frame_count = []
        for video_file in video_files:
          cap = cv2.VideoCapture(video_file)
          if(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))<150):
            video_files.remove(video_file)
            continue
          frame_count.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        create_face_videos(video_files,'D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/preprocess/video_224/'+type+'/'+folder)