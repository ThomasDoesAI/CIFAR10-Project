import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from data_loading import get_data_loaders
from model import CIFAR10CNN

def train_model(epochs=10, batch_size=64, learning_rate=0.001, weight_decay=1e-4):
    # Load data
    trainloader, testloader, _ = get_data_loaders(batch_size)
    
    # Initialize the model
    model = CIFAR10CNN()
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
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
        
        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(testloader)
        print(f'Epoch {epoch + 1} validation loss: {val_loss:.3f}')
        
        # Step the scheduler with the validation loss
        scheduler.step(val_loss)
    
    print('Finished Training')
    # Save the trained model
    torch.save(model.state_dict(), 'models/cifar10_cnn.pth')

if __name__ == "__main__":
    train_model()
