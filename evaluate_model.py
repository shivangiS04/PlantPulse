import torch
import torch.nn as nn
from torchvision import models
from dataloader import val_loader, test_loader, num_classes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Evaluation function
def evaluate(model, loader, set_name):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    print(f'{set_name} Accuracy: {accuracy:.2f}%')


if __name__ == "__main__":
    # Load model
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load('best_plantpulse_mobilenetv2.pth', map_location=device))
    model = model.to(device)

    # Evaluate validation and test sets
    evaluate(model, val_loader, "Validation")
    evaluate(model, test_loader, "Test")

