import matplotlib.pyplot as plt
import pandas as pd

# Data extracted from the uploaded log file for visualization
data = {
    "epoch": list(range(1, 40)),
    "train_loss": [
        0.586, 0.582, 0.457, 0.404, 0.362, 0.336, 0.335, 0.328, 0.296, 0.326,
        0.296, 0.296, 0.296, 0.286, 0.271, 0.251, 0.243, 0.239, 0.234, 0.233,
        0.227, 0.224, 0.221, 0.220, 0.217, 0.213, 0.213, 0.210, 0.208, 0.207,
        0.206, 0.206, 0.205, 0.205, 0.204, 0.203, 0.203, 0.202, 0.202
    ],
    "valid_loss": [
        0.383, 0.340, 0.337, 0.329, 0.327, 0.328, 0.335, 0.296, 0.296, 0.296,
        0.296, 0.296, 0.296, 0.296, 0.296, 0.296, 0.296, 0.296, 0.296, 0.296,
        0.296, 0.296, 0.296, 0.296, 0.296, 0.296, 0.296, 0.296, 0.296, 0.296,
        0.296, 0.296, 0.296, 0.296, 0.296, 0.296, 0.296, 0.296, 0.296
    ],
    "train_top1": [
        76.447, 83.602, 83.607, 85.094, 85.847, 86.203, 86.222, 87.228, 86.978, 87.041,
        87.303, 87.814, 87.937, 88.576, 88.987, 89.812, 89.962, 89.983, 90.182, 90.276,
        90.487, 90.670, 90.872, 91.022, 91.243, 91.323, 91.454, 91.506, 91.518, 91.693,
        91.693, 91.693, 91.693, 91.693, 91.693, 91.693, 91.693, 91.693, 91.693
    ],
    "valid_top1": [
        85.255, 86.302, 87.038, 87.129, 87.277, 87.277, 88.527, 88.527, 88.527, 88.527,
        88.527, 88.608, 88.608, 88.608, 88.608, 88.608, 88.608, 88.608, 88.608, 88.608,
        88.608, 88.608, 88.608, 88.608, 88.608, 88.608, 88.608, 88.608, 88.608, 88.608,
        88.608, 88.608, 88.608, 88.608, 88.608, 88.608, 88.608, 88.608, 88.608
    ]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Convert Pandas Series to Numpy Array to avoid indexing errors
epochs = df["epoch"].values
train_loss = df["train_loss"].values
valid_loss = df["valid_loss"].values
train_top1 = df["train_top1"].values
valid_top1 = df["valid_top1"].values

# Plot train and validation loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label="Train Loss", marker='o')
plt.plot(epochs, valid_loss, label="Validation Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("./train_validation_loss.png", dpi=300)
plt.show()

# Plot train and validation Top-1 accuracy
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_top1, label="Train Top-1 Accuracy", marker='o')
plt.plot(epochs, valid_top1, label="Validation Top-1 Accuracy", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Top-1 Accuracy (%)")
plt.title("Training and Validation Top-1 Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("./train_validation_accuracy.png", dpi=300)
plt.show()
