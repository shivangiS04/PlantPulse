import torch
import torch.nn as nn
from torchvision import models
from dataloader import test_loader, train_dataset, num_classes
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_and_confusion(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

if __name__ == "__main__":
    # Load model architecture and weights
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load('best_plantpulse_mobilenetv2.pth', map_location=device))
    model = model.to(device)

    # Get all predictions and labels on test set
    true_labels, predictions = evaluate_and_confusion(model, test_loader)

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    classes = train_dataset.classes

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Print classification report
    report = classification_report(true_labels, predictions, target_names=classes)
    print("Classification Report:\n", report)
