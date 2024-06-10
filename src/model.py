import torch.nn as nn
import torch.nn.functional as F

class CIFAR10CNN(nn.Module):
    """
    A CNN model with three convolutional layers for CIFAR-10.
    """
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # First fully connected layer
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # 128 channels * 4x4 feature map size after pooling
        # Second fully connected layer (output layer)
        self.fc2 = nn.Linear(512, 10)  # 10 output classes for CIFAR-10
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply first convolutional layer followed by ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply second convolutional layer followed by ReLU activation and pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Apply third convolutional layer followed by ReLU activation and pooling
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten the feature map into a vector
        x = x.view(-1, 128 * 4 * 4)
        # Apply first fully connected layer followed by ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # Apply output fully connected layer
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = CIFAR10CNN()
    print(model)
