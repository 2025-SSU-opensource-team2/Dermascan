import argparse
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt

def save_plot(data1, data2, ylabel, title,filename_prefix,experiment_dir):
    """
    Save a plot comparing two datasets to a PDF file.

    Args:
        data1 (list): First data series (e.g., training).
        data2 (list): Second data series (e.g., validation).
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        filename_prefix (str): Prefix for the saved filename.
    """
    
    plt.figure(figsize=(10, 5))
    plt.plot(data1, label=f'Train {ylabel}')
    plt.plot(data2, label=f'Validation {ylabel}')
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    filename = datetime.now().strftime(f"{experiment_dir}/{filename_prefix}.pdf")
    plt.savefig(filename)
    logging.info(f"{title} plot saved")
    plt.close()

def setup_logging(experiment_dir):
    
    log_filename = datetime.now().strftime(f"{experiment_dir}/train_log.txt")
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),  # 로그 파일 저장
            logging.StreamHandler()             # 콘솔에도 출력
        ]
    )
    return log_filename

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--experiment_name", type=str, default=datetime.now().strftime("%y%m%d"), help="Experiment name")
    return parser.parse_args()

def create_experiment_dir(experiment_name: str):
    """Create experiment directory based on the experiment name."""
    experiment_dir = os.path.join("saved_models", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir