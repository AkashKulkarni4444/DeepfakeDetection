import argparse
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(target, epoch, model_name):
    # Load CSV file
    csv_path = f"testOutput/{model_name}/{target}/mobilenet_{epoch}_{target}_test_results.csv"
    df = pd.read_csv(csv_path)

    # Calculate evaluation metrics
    true_labels = df['True_Label']
    predicted_labels = df['Predicted_Label']

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    return accuracy, precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate ML model")
    parser.add_argument('-m', '--model_name', type=str, required=True, help="Name of the model")
    parser.add_argument('-e', '--epochs', type=int, nargs='+', required=True, help="List of epoch numbers")
    parser.add_argument('-t', '--targets', type=str, nargs='+', required=True, help="List of target directories")
    args = parser.parse_args()

    targets = args.targets
    epochs = args.epochs
    model_name = args.model_name
    results_csv_path = "evaluation_results.csv"
    
    for target in targets:
        for epoch in epochs:
            accuracy, precision, recall, f1 = evaluate_model(target, epoch, model_name)
             # Store results in a CSV file
            results_df = pd.DataFrame({
                'Model_name': [model_name],
                'Epoch': [epoch],
                'Target': [target],
                'Accuracy': [accuracy],
                'Precision': [precision],
                'Recall': [recall],
                'F1_Score': [f1]
            })

            if os.path.exists(results_csv_path):
                results_df.to_csv(results_csv_path, mode='a',
                                  header=False, index=False)
            else:
                results_df.to_csv(results_csv_path, index=False)

    print("Results stored in evaluation_results.csv")


if __name__ == "__main__":
    main()
