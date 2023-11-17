import torch
import matplotlib.pyplot as plt

readPath = "/home/xingjian.zhang/sleep/4_save/04_ST_FPZ-Cz&EOG/AttnSleep_2CH_S3_1/04_ST_FPZ-Cz&EOG_confusion_matrix.torch"

# Load the confusion matrix
confusion_matrix = torch.load(readPath)

# Define the class labels
labels = ['wake', 'N1', 'N2', 'N3', 'REM']

# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Display the confusion matrix as an image
cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)

# Add a color bar to the right of the plot
plt.colorbar(cax)

# Set the labels for the x and y axes
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)

# Rotate the x-axis labels for better readability
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Display the values in each cell of the matrix
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='r')

# Set labels for the x and y axes
plt.xlabel('Predicted')
plt.ylabel('True')

# Display the plot
plt.show()
