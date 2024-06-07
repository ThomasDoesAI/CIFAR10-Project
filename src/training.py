import torch
import torch.optim as optim
import torch.nn as nn
from data_loading import get_data_loaders
from model import CIFAR10CNN

def train_model(epochs=10, batch_size=64, learning_rate=0.001):
    # Load data
    trainloader, testloader, _ = get_data_loaders(batch_size)
    
    # Initialize the model
    model = CIFAR10CNN()
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics every 200 mini-batches
            running_loss += loss.item()
            if i % 200 == 199:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    print('Finished Training')
    # Save the trained model
    torch.save(model.state_dict(), 'models/cifar10_cnn.pth')

if __name__ == "__main__":
    train_model()
