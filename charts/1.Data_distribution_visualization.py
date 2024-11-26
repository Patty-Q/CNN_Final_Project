import matplotlib.pyplot as plt
from collections import Counter
import os

train_data_path = "../data/train/"  # Change to actual path

# Get the category, convert the category name to an integer and then sort
categories = [folder for folder in os.listdir(train_data_path) if os.path.isdir(os.path.join(train_data_path, folder))]
categories = sorted(categories, key=lambda x: int(x))  # Sort by value

# Count the number of samples in each category
category_counts = {category: len(os.listdir(os.path.join(train_data_path, category))) for category in categories}

# Draw a bar chart
plt.figure(figsize=(12, 6))
plt.bar(list(category_counts.keys()), list(category_counts.values())) 
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.title("Training Data Distribution")
plt.xticks(rotation=90)  # Rotate x-axis labels to fit space
plt.grid(axis='y')
plt.savefig("data_distribution.png", dpi=300)
plt.show()
