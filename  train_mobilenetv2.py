import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataloader import train_loader, val_loader, num_classes  # Import your dataloaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


if __name__ == "__main__":
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_acc = 0.0
    epochs = 5

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_accuracy = evaluate(model, val_loader)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_plantpulse_mobilenetv2.pth')
            print("Best model saved.")

    print(f'Training Complete. Best Validation Accuracy: {best_val_acc:.2f}%')
