import torch
import torchvision
import torchvision.transforms as transforms

def get_data_loaders(batch_size=64):
    # Transformations: Convert images to tensors and normalize them
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Download and load training and test datasets
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Classes in CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes

if __name__ == "__main__":
    trainloader, testloader, classes = get_data_loaders()
    # Print first batch of training data to verify
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print('Loaded training batch of size:', images.size())
