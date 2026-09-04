import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import joblib

def train_decision_tree(csv_path: str, model_output_path: str, plot_output_path: str):
    """
    Trains a Decision Tree using scikit-learn on a dataset of Hu invariants and labels.
    Plots the tree and saves the trained model to a file.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, header=None)
    df.columns = [f"hu{i+1}" for i in range(df.shape[1]-1)] + ["label"]
    
    # The last column is the target label
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
        
    print(f"Training Decision Tree Classifier on {len(X)} samples...")
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    
    print(f"Saving model to {model_output_path}...")
    joblib.dump(clf, model_output_path)
    
    print(f"Generating and saving tree plot to {plot_output_path}...")
    plt.figure(figsize=(20, 10))
    class_names = [str(cls) for cls in sorted(y.unique())]
    plot_tree(
        clf, 
        filled=True, 
        feature_names=X.columns.tolist(), 
        class_names=class_names,
        rounded=True
    )
    plt.title("Decision Tree for Tetris Piece Classification")
    plt.savefig(plot_output_path, bbox_inches='tight')
    plt.close()
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Decision Tree model on Hu invariants")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--model_output", type=str, default="decision_tree_model.joblib", help="Output path for the trained model")
    parser.add_argument("--plot_output", type=str, default="decision_tree_plot.png", help="Output path for the tree plot")
    
    args = parser.parse_args()
    train_decision_tree(args.csv_path, args.model_output, args.plot_output)
