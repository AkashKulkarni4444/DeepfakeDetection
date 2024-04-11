import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV file
df = pd.read_csv('evaluation_results.csv')

# Iterate over unique target classes
targets = df['Target'].unique()

# Define colors for different metrics
colors = ['blue', 'green', 'orange', 'red']

# Create subplots for each metric
fig, axs = plt.subplots(len(targets), 1, figsize=(10, 8), sharex=True)

for idx, target in enumerate(targets):
    target_df = df[df['Target'] == target]
    
    # Plot each metric
    for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1_Score']):
        axs[idx].plot(target_df['Epoch'], target_df[metric], label=metric, color=colors[i])
    
    # Set title and labels
    axs[idx].set_title(target)
    axs[idx].set_ylabel('Score')
    axs[idx].legend()

# Set common x-label
plt.xlabel('Epoch')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()
