from datasets import load_dataset
import pandas as pd
import os, sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.trainer import TextClassifier
import src.analysis as analysis

def main():
    clf = TextClassifier.load("Models/custom")
    dataset = load_dataset("imdb", split="test")
    df = clf.predict_dataset(dataset, batch_size=32)
    df.to_csv("plots/predictions.csv", index=False)

    # Analysis
    analysis.print_classification_report(df)
    analysis.plot_confusion_matrix(df, save_path="plots/confusion_matrix.png")
    analysis.plot_roc_curve(df, save_path="plots/roc_curve.png")
    analysis.plot_probability_histogram(df, save_path="plots/probability_histogram.png")
    analysis.show_misclassified_examples(df, n=5)

if __name__ == "__main__":
    main()
