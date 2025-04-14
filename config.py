import torch

# 0번 GPU 사용
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 30
LEARNING_RATE = 0.001

train_dir = "dermnet_data/train"
test_dir = "dermnet_data/test"
