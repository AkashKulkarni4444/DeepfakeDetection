import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score

def load_data(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    return df['True_Label'], df['Predicted_Label']

def calculate_metrics(true_labels, predicted_labels):
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # ROC curve and AUC
    fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
    auc = roc_auc_score(true_labels, predicted_labels)
    
    return accuracy, precision, recall, f1, cm, fpr, tpr, auc

def main():
    # Path to CSV file
    csv_file = 'your_csv_file.csv'  # Replace with your CSV file path
    
    # Load data
    true_labels, predicted_labels = load_data(csv_file)
    
    # Calculate metrics
    accuracy, precision, recall, f1, cm, fpr, tpr, auc = calculate_metrics(true_labels, predicted_labels)
    
    # Print metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:")
    print(cm)
    print("AUC:", auc)
    
    # You can plot ROC curve if needed
    # import matplotlib.pyplot as plt
    # plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
    # plt.show()

if __name__ == "__main__":
    main()
