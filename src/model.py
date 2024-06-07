import torch.nn as nn
import torch.nn.functional as F

class CIFAR10CNN(nn.Module):
    """
    Convolutional Layers (conv1, conv2, conv3):

conv1: The first convolutional layer takes 3 input channels (color images) and outputs 32 feature maps using 3x3 kernels with padding of 1 to preserve the spatial dimensions.
conv2: The second layer increases the number of feature maps to 64.
conv3: The third layer further increases the feature maps to 128.
Increasing the number of feature maps in deeper layers helps capture more complex patterns.

Max Pooling Layer (pool):

We use a 2x2 max pooling layer with a stride of 2, which reduces the spatial dimensions of the feature maps by half. This helps in reducing the computational load and prevents overfitting.
Fully Connected Layers (fc1, fc2):

fc1: The first fully connected layer takes the flattened feature maps (128 channels * 4x4 spatial dimensions after pooling) and outputs 512 neurons.
fc2: The second fully connected layer outputs 10 neurons, corresponding to the 10 classes of the CIFAR-10 dataset.
Activation Functions:

We use ReLU (Rectified Linear Unit) activation after each convolutional layer to introduce non-linearity, which helps the network learn more complex patterns.
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
        # Apply output fully connected layer
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = CIFAR10CNN()
    print(model)
