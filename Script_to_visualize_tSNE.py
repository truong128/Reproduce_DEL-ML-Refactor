import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# === Load t-SNE embedding ===
with open("~/data_embedded_compound.pkl", "rb") as f:
    tsne_data = np.array(pickle.load(f))

# === Load prediction results ===
pred_df = pd.read_csv("~/compound_pred_mlp.csv")  # or use the chemprop one
pred_labels = pred_df['prediction'].values  # 0 = non-binder, 1 = binder

# Safety check: must match number of t-SNE points
assert len(tsne_data) == len(pred_labels), "Mismatch between t-SNE points and predictions!"

# === Separate x, y for plotting ===
x = tsne_data[:, 0]
y = tsne_data[:, 1]

# === Assign colors ===
colors = ['red' if p == 1 else 'blue' for p in pred_labels]

# === Plot ===
plt.figure(figsize=(8, 6))
plt.scatter(x, y, c=colors, s=30, alpha=0.7, edgecolors='k')
plt.title("t-SNE of Compound Space\nRed = Predicted Binder, Blue = Predicted Non-binder")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.grid(True)
plt.tight_layout()
plt.show()
