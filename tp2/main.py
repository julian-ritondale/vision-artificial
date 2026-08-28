import sys
from utils.trainer import train_decision_tree
from utils.dataset_generator import generate_hu_moments_file
from utils.demo import run_camera_demo

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_camera_demo()
        return

    generate_hu_moments_file()
    train_decision_tree(
        csv_path='dataset/tetris-hu-moments.csv',
        model_output_path='models/decision_tree_model.joblib',
        plot_output_path='models/decision_tree_plot.png'
    )

    if "--run-demo" in sys.argv:
        run_camera_demo()

if __name__ == "__main__":
    main()
