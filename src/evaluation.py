import torch
from data_loading import get_data_loaders
from model import CIFAR10CNN

def evaluate_model(batch_size=64):
    # Load the test data
    _, testloader, classes = get_data_loaders(batch_size)
    
    # Initialize the model
    model = CIFAR10CNN()
    
    # Load the trained model parameters
    model.load_state_dict(torch.load('models/cifar10_cnn.pth'))
    
    # Set the model to evaluation mode
    model.eval()
    
    correct = 0
    total = 0
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            
            # Get the class with the highest probability
            _, predicted = torch.max(outputs.data, 1)
            
            # Update the total number of images and correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f}%')

if __name__ == "__main__":
    evaluate_model()
