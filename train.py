import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import *
from model import get_resnet34
from data_loader import get_dataloaders
from utils import *

import logging
import os
from datetime import datetime


args = parse_args()
experiment_name = args.experiment_name
experiment_dir = create_experiment_dir(experiment_name=experiment_name)

# Set up logging in the experiment directory
log_filename = setup_logging(experiment_dir=experiment_dir)

def train():
    # Load data
    train_dataset, train_loader, test_loader = get_dataloaders(train_dir, test_dir, BATCH_SIZE)

    # Model, criterion, optimizer
    model = get_resnet34(num_classes=len(train_dataset.classes)).to(device)
    print("number of classes:",len(train_dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_accuracy = 0.0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        for batch_idx, (images, labels) in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            loop.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        logging.info(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")

        # Validation
        model.eval()
        correct, total = 0, 0
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(test_loader)
        accuracy = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.2f}%")
        logging.info(f"Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.2f}%")

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = os.path.join(experiment_dir, f"best_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved")
            logging.info(f"Best model saved")

        # Save latest model (optional)
        latest_model_path = os.path.join(experiment_dir, "latest_model.pth")
        torch.save(model.state_dict(), latest_model_path)
        logging.info(f"Latest model saved")

        # Save PDF
        save_plot(train_losses, val_losses, 'Loss', 'Training and Validation Loss', 'loss',experiment_dir=experiment_dir)
        save_plot(train_accuracies, val_accuracies, 'Accuracy (%)', 'Training and Validation Accuracy', 'accuracy',experiment_dir=experiment_dir)

if __name__ == "__main__":
    train()

