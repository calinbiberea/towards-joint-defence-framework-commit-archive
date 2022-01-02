# Dataset processing and loading imports
import torch.utils.data as DataUtils
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


# The good (CIFAR-10) model has batch size 128, but cannot do adversarial attacks with it (out of memory)
def get_CIFAR10_data_loaders(
    DATA_ROOT, batchSize=64, trainSetSize=50000, validationSetSize=10000, testSetSize=10000
):

    # Create a separate transform for each dataset
    # (in case we decide to transform differently)
    trainSetTransform = transforms.Compose([transforms.ToTensor()])
    validationSetTransform = transforms.Compose([transforms.ToTensor()])
    testSetTransform = transforms.Compose([transforms.ToTensor()])

    # Download the dataset (note we technically use the same set for validation
    # and training)
    trainSet = datasets.CIFAR10(
        root=DATA_ROOT, download=True, train=True, transform=trainSetTransform
    )
    validationSet = datasets.CIFAR10(
        root=DATA_ROOT, download=True, train=True, transform=validationSetTransform
    )
    testSet = datasets.CIFAR10(
        root=DATA_ROOT, download=True, train=False, transform=testSetTransform
    )

    # Get the training indices to split into training and validation sets
    indices = np.arange(0, trainSetSize + validationSetSize)
    np.random.shuffle(indices)

    # Construct random samplers (for better training)
    trainSetSampler = SubsetRandomSampler(indices[:trainSetSize])
    validationSetSampler = SubsetRandomSampler(indices[trainSetSize:])
    testSetSampler = SubsetRandomSampler(np.arange(0, testSetSize))

    # Finally, construct the loaders that will be used to get images
    trainSetLoader = DataUtils.DataLoader(
        trainSet, batch_size=batchSize, sampler=trainSetSampler
    )
    validationSetLoader = DataUtils.DataLoader(
        validationSet, batch_size=batchSize, sampler=validationSetSampler
    )
    testSetLoader = DataUtils.DataLoader(
        testSet, batch_size=batchSize, sampler=testSetSampler
    )

    # Return the loaders
    return trainSetLoader, validationSetLoader, testSetLoader


def get_MNIST_data_loaders(
    DATA_ROOT, batchSize=64, trainSetSize=50000, validationSetSize=10000, testSetSize=10000
):

    # Create a separate transform for each dataset
    # (in case we decide to transform differently)
    trainSetTransform = transforms.Compose([transforms.ToTensor()])
    validationSetTransform = transforms.Compose([transforms.ToTensor()])
    testSetTransform = transforms.Compose([transforms.ToTensor()])

    # Download the dataset (note we technically use the same set for validation
    # and training)
    trainSet = datasets.MNIST(
        root=DATA_ROOT, download=True, train=True, transform=trainSetTransform
    )
    validationSet = datasets.MNIST(
        root=DATA_ROOT, download=True, train=True, transform=validationSetTransform
    )
    testSet = datasets.MNIST(
        root=DATA_ROOT, download=True, train=False, transform=testSetTransform
    )

    # Get the training indices to split into training and validation sets
    indices = np.arange(0, trainSetSize + validationSetSize)
    np.random.shuffle(indices)

    # Construct random samplers (for better training)
    trainSetSampler = SubsetRandomSampler(indices[:trainSetSize])
    validationSetSampler = SubsetRandomSampler(indices[trainSetSize:])
    testSetSampler = SubsetRandomSampler(np.arange(0, testSetSize))

    # Finally, construct the loaders that will be used to get images
    trainSetLoader = DataUtils.DataLoader(
        trainSet, batch_size=batchSize, sampler=trainSetSampler
    )
    validationSetLoader = DataUtils.DataLoader(
        validationSet, batch_size=batchSize, sampler=validationSetSampler
    )
    testSetLoader = DataUtils.DataLoader(
        testSet, batch_size=batchSize, sampler=testSetSampler
    )

    # Return the loaders
    return trainSetLoader, validationSetLoader, testSetLoader