from utils.trainer import train_decision_tree
from utils.dataset_generator import generate_hu_moments_file

generate_hu_moments_file()
train_decision_tree(csv_path='dataset/tetris-hu-moments.csv', model_output_path='model/decision_tree_model.joblib', plot_output_path='model/decision_tree_plot.png')
