import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('evaluation_results.csv')

# Iterate over each row in the dataset
for index, row in data.iterrows():
    if(row['Epoch']==10):
        # Extract relevant metrics for the confusion matrix
        true_positive = row['True_Positive']
        true_negative = row['True_Negative']
        false_positive = row['False_Positive']
        false_negative = row['False_Negative']
        
        # Create the confusion matrix
        confusion_matrix = [[true_negative, false_positive],
                            [false_negative, true_positive]]
        
        # Plot the heatmap
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Actual Negative', 'Actual Positive'],
                    yticklabels=['Predicted Negative', 'Predicted Positive'])
        
        # Set title and labels
        plt.title(f"Confusion Matrix - Model: {row['Model_name']}, Epoch: {row['Epoch']}, Target: {row['Target']}")
        plt.xlabel('True Label')
        plt.ylabel('Predicted Label')
        
        # Show the plot
        plt.show()
