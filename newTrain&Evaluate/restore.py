import pandas as pd

# Read the original CSV file
original_df = pd.read_csv("D:/IIIT/IIIT_Shri_City/BTP/code/mobilenet_resnet/newTrain&Evaluate/testOutput/mobilenet/NeuralTextures/mobilenet_1_NeuralTextures_test_results.csv")

# Create a new DataFrame with only 'Image_Path' and 'True_Label' columns
test_df = original_df[['Image_Path', 'True_Label']]

# Write the new DataFrame to a new CSV file
test_df.to_csv("test.csv", index=False)

print("New CSV file 'test.csv' created successfully.")
