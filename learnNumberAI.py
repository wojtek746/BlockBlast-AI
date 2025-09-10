import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch.nn as nn
import torch.nn.functional as F

class NumberNet(nn.Module):
    def __init__(self):
        super(NumberNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.fc1 = nn.Linear(13440, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((56, 57)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#dataset = datasets.ImageFolder(root=r"C:\Users\Wojtek746\Desktop\AI\png", transform=transform)

train_dataset = datasets.ImageFolder(root=r"C:\Users\Wojtek746\Desktop\AI\train", transform=transform)
test_dataset  = datasets.ImageFolder(root=r"C:\Users\Wojtek746\Desktop\AI\test",  transform=transform)

#train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda")
model = NumberNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "number_net.pth")