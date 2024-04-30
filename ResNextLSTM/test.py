import pandas as pd

if __name__ == "__main__":
    # Read only the first column of trainDataSet.csv into train_videos
    # train_videos = pd.read_csv('D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/ResNextLSTM/datasetDict_video_224/fake/Deepfakes/trainDataSet.csv', usecols=[1])
    # Read only the first column of testDataSet.csv into valid_videos
    # valid_videos = pd.read_csv('D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/ResNextLSTM/datasetDict_video_224/fake/Deepfakes/testDataSet.csv', usecols=[1])

    # Print the first few rows of train_videos and valid_videos
    # print("train_videos:\n", train_videos.head())
    # print("\nvalid_videos:\n", valid_videos.head())

    # # # Check the type of train_videos and valid_videos
    # print("\nType of train_videos:", type(train_videos))
    # print("Type of valid_videos:", type(valid_videos))
    
    # labels = pd.read_csv('D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/ResNextLSTM/datasetDict_video_224/fake/Deepfakes/trainDataSet.csv')
    # labels += pd.read_csv('D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/ResNextLSTM/datasetDict_video_224/fake/Deepfakes/testDataSet.csv')
    train_labels = pd.read_csv('D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/ResNextLSTM/datasetDict_video_224/fake/Deepfakes/trainDataSet.csv')
    # Read the second CSV file
    test_labels = pd.read_csv('D:/IIIT/IIIT_Shri_City/BTP/code/COMP90055_Research_Project-master/ResNextLSTM/datasetDict_video_224/fake/Deepfakes/testDataSet.csv')
    # Concatenate the two DataFrames along the rows (axis=0)
    combined_labels = pd.concat([train_labels, test_labels], ignore_index=True)
    print(combined_labels)