

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)


def print_classification_report(df, target_names=["NEGATIVE", "POSITIVE"]):
    print("Classification Report:\n")
    print(classification_report(df["label"], df["pred"], target_names=target_names))


def plot_confusion_matrix(df, save_path=None):
    cm = confusion_matrix(df["label"], df["pred"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["NEG", "POS"],
        yticklabels=["NEG", "POS"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
        print(f"📊 Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_roc_curve(df, save_path=None):
    fpr, tpr, _ = roc_curve(df["label"], df["prob_pos"])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
        print(f"📈 ROC curve saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_probability_histogram(df, save_path=None):
    plt.figure(figsize=(6, 5))
    plt.hist(df["prob_pos"][df["label"] == 1], bins=20, alpha=0.6, label="Positive reviews")
    plt.hist(df["prob_pos"][df["label"] == 0], bins=20, alpha=0.6, label="Negative reviews")
    plt.xlabel("Predicted Probability (Positive)")
    plt.ylabel("Count")
    plt.title("Probability Calibration Histogram")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"📉 Probability histogram saved to {save_path}")
    else:
        plt.show()
    plt.close()


def show_misclassified_examples(df, n=5):
    errors = df[df["label"] != df["pred"]]
    print(f"\n Showing {n} misclassified examples:")
    for _, row in errors.sample(min(n, len(errors))).iterrows():
        true = "POS" if row["label"] == 1 else "NEG"
        pred = "POS" if row["pred"] == 1 else "NEG"
        print(f"\nTrue: {true}, Pred: {pred}, Prob: {row['prob_pos']:.2f}")
        print(f"Text: {row['text'][:300]}...")
