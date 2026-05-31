'''
Visualize the results from the CSV files. 
'''

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("metrics.csv")

plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.plot(df["epoch"], df["dice"], label="Dice")
plt.plot(df["epoch"], df["iou"], label="IoU")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Segmentation Metrics")
plt.legend()
plt.grid(True)
plt.show()